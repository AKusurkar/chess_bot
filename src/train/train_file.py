r"""
Stage 5 — training: turn the self-play examples into gradient steps.

This module consumes exactly what Stage 4 produces — a stream of
TrainingExample = (state (20,8,8) f32, pi (1880,) f32, z in {-1,0,+1}) — and
trains MarchHare (network.py) on AlphaZero's loss:

    loss = (z - v)^2  -  pi . log p  +  c * ||theta||^2
           \_______/     \________/     \___________/
            value MSE     policy CE       weight decay
      (regress outcome)  (toward the    (realized as the
                          search-improved  optimizer's decoupled
                          target pi)        decay; see below)

WHY THIS IS THE WHOLE OF STAGE 5
--------------------------------
  * Value target is the game OUTCOME z, not a bootstrapped estimate. Pure
    supervised regression onto {-1,0,+1} — stable, no TD, no target network.
    Both v (network) and z (label) are from the SIDE-TO-MOVE's perspective
    (network.py's value head convention, game.py's finalize labeling), so
    (z - v)^2 compares like with like. No sign reconciliation happens here; if
    you ever see the value head learn the wrong sign, the bug is upstream in
    those conventions, not in this loss.
  * Policy target is the MCTS visit distribution pi (the improvement operator),
    NOT the network's own head. Cross-entropy -pi . log softmax(logits).

POLICY LOSS IS UNMASKED — ON PURPOSE
------------------------------------
The stored example carries pi but NOT a legal-move mask (game.py stores
(state, pi, z) only). So training takes cross-entropy over the FULL 1880-way
softmax, not a legal-only softmax. This is standard AlphaZero and it is fine:
pi is zero on illegal moves, so only legal terms contribute to -pi . log p;
illegal logits are pulled down only indirectly, through the softmax normalizer.
That is exactly enough, because masking is an INFERENCE-time concern and already
happens in mcts.make_net_evaluator (Stage 2) — the net never has to *emit* a
legal distribution, it only has to rank legal moves well. Re-deriving a mask
from the encoded tensor here would be fragile and is unnecessary.

WEIGHT DECAY (the c * ||theta||^2 term)
---------------------------------------
Realized as the optimizer's weight decay rather than an explicit term added to
the loss. Default optimizer is AdamW, whose decay is DECOUPLED — the "cleaner
formulation" the plan calls out. (Plain Adam with weight_decay coerces it back
into the coupled, L2-in-the-loss behaviour the plan's literal formula shows;
pass optimizer="adam" if you want that.) Either way, do not ALSO add an explicit
||theta||^2 term or you double-regularize.

BATCHNORM DISCIPLINE (pairs with self_play.py's note)
-----------------------------------------------------
Self-play left the net in eval() mode (correct there: BN must use running
stats at inference). Training must run in train() mode so BN uses batch stats
and updates its running estimates. Trainer.train_step() calls net.train() every
step, so alternating self-play and training on the SAME net object is safe — you
do not have to remember to flip it back by hand. BN on tiny batches is noisy, so
keep the batch >= a few hundred (the plan's 256-1024).

CUDA
----
Device is auto-selected (cuda if available, else cpu) or forced via
TrainConfig.device. The net is moved once at Trainer construction; each batch is
created on the host and shipped with pinned memory + non_blocking copies when on
cuda. Optional AMP (autocast + GradScaler) is wired and gated behind
TrainConfig.use_amp, off by default. cudnn.benchmark is enabled on cuda (input
shape is fixed at (*,20,8,8), so autotuning pays off).

WHAT THIS MODULE DOES NOT DO
----------------------------
The outer loop (generate self-play -> extend buffer -> train -> eval vs a fixed
checkpoint) is Stage 6. This module owns the buffer, the loss, the optimizer
state, and checkpoint save/load; Stage 6 will call ReplayBuffer.extend(...) and
Trainer.train(...) in a loop. Keeping the optimizer/scheduler inside a persistent
Trainer is deliberate: Adam's moments and the LR schedule must survive ACROSS
outer iterations, not reset every round.

Torch is imported lazily (inside the functions/methods that need it), so
ReplayBuffer and the loss math stay importable and testable without torch — the
same discipline mcts.py uses for its search core.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.utils_file import NUM_MOVES, NUM_PLANES      

from utils.single_game import TrainingExample    # (state (20,8,8) f32, pi (1880,) f32, z float)


# --------------------------------------------------------------------------- #
# Replay buffer  (pure numpy — no torch, so it is importable/testable here)
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """A fixed-capacity ring buffer of TrainingExamples, sized to TURN OVER.

    The plan wants the most recent ~200k-500k positions held, old ones aging
    out so the net always trains on data near its current strength (a stagnant
    buffer trains you against a bygone, weaker self). Backed by a Python list
    with a write cursor: O(1) append and O(1) random indexing (a deque would
    make indexing O(n) and sampling large batches slow).

    Stores references to the small per-example arrays; nothing is copied until
    sample() stacks a batch. Memory ~= len * (state 5KB + pi 7.5KB) ~= 12.5KB
    per position (~2.5GB at 200k, ~6.3GB at 500k) — the dominant RAM cost of the
    whole project, so pick capacity with your host RAM in mind.
    """

    def __init__(self, capacity: int = 500_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: List[TrainingExample] = []
        self._pos = 0                      # next overwrite slot once full

    def __len__(self) -> int:
        return len(self._data)

    def is_full(self) -> bool:
        return len(self._data) >= self.capacity

    def append(self, example: TrainingExample) -> None:
        if len(self._data) < self.capacity:
            self._data.append(example)
        else:
            self._data[self._pos] = example                 # overwrite oldest
            self._pos = (self._pos + 1) % self.capacity

    def extend(self, examples) -> None:
        """Add a batch of examples (e.g. one call's worth of self-play)."""
        for ex in examples:
            self.append(ex)

    def sample(self, batch_size: int,
               rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a minibatch and stack it into contiguous arrays:

            states : (B, 20, 8, 8) float32
            pis    : (B, 1880)     float32
            zs     : (B,)          float32

        Uniform sampling, without replacement when the buffer is large enough
        (batch_size <= len), else with replacement. Returns numpy on purpose —
        Trainer._to_tensors owns the host->device hop, keeping this class
        torch-free.
        """
        n = len(self._data)
        if n == 0:
            raise ValueError("cannot sample from an empty buffer")
        rng = rng or np.random.default_rng()
        replace = batch_size > n
        idx = rng.choice(n, size=batch_size, replace=replace)

        states = np.stack([self._data[i][0] for i in idx]).astype(np.float32, copy=False)
        pis    = np.stack([self._data[i][1] for i in idx]).astype(np.float32, copy=False)
        zs     = np.asarray([self._data[i][2] for i in idx], dtype=np.float32)
        return states, pis, zs


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Training knobs. Defaults follow the plan (Adam ~1e-3, batch 256-1024)."""
    lr: float = 1e-3
    weight_decay: float = 1e-4               # the c in c*||theta||^2
    optimizer: str = "adamw"                 # "adamw" (decoupled) or "adam" (coupled)
    betas: Tuple[float, float] = (0.9, 0.999)
    batch_size: int = 512
    grad_clip_norm: Optional[float] = None   # e.g. 1.0 to clip; None = off
    # Step-decay schedule (plan: "Adam ~1e-3 with step-decay"). Off unless set.
    lr_step_size: Optional[int] = None       # in optimizer steps
    lr_gamma: float = 0.1
    use_amp: bool = False                     # mixed precision (cuda only)
    device: Optional[str] = None              # "cuda" / "cpu" / None = auto
    cudnn_benchmark: bool = True


# --------------------------------------------------------------------------- #
# Device helper
# --------------------------------------------------------------------------- #
def select_device(prefer: Optional[str] = None):
    """Resolve a torch.device. `prefer`='cuda'/'cpu' forces it; None auto-picks
    cuda when available. Torch imported lazily so importing this module needs no
    torch."""
    import torch
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Loss  (standalone so it can be unit-tested / reused by eval)
# --------------------------------------------------------------------------- #
def alphazero_losses(logits, value, pi_target, z_target):
    """Return (total, value_loss, policy_loss) as torch scalars.

        value_loss  = mean (z - v)^2
        policy_loss = mean_b [ -sum_a pi(b,a) * log_softmax(logits)(b,a) ]

    `value` may be (B,) or (B,1); it is squeezed. Weight decay is NOT included
    here — it lives in the optimizer (see module docstring). Torch is imported
    lazily.
    """
    import torch  # noqa: F401
    import torch.nn.functional as F

    value = value.reshape(-1)                       # (B,1) or (B,) -> (B,)
    value_loss = F.mse_loss(value, z_target)
    logp = F.log_softmax(logits, dim=1)             # full 1880-way; see docstring
    policy_loss = -(pi_target * logp).sum(dim=1).mean()
    return value_loss + policy_loss, value_loss, policy_loss


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Owns the network's device placement, optimizer, LR schedule, AMP scaler,
    and checkpoint I/O. Construct ONCE and reuse across Stage-6 outer iterations
    so Adam's moments and the LR schedule persist rather than resetting.
    """

    def __init__(self, net, config: Optional[TrainConfig] = None):
        import torch

        self.config = config or TrainConfig()
        self.device = select_device(self.config.device)
        self.net = net.to(self.device)

        if self.device.type == "cuda" and self.config.cudnn_benchmark:
            # Fixed input geometry (*,20,8,8) -> autotuning the conv algos pays off.
            torch.backends.cudnn.benchmark = True

        self.optimizer = self._build_optimizer()

        self.scheduler = None
        if self.config.lr_step_size:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )

        # GradScaler only does anything for fp16 autocast on cuda.
        self.use_amp = bool(self.config.use_amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None

        self._step = 0

    # -- setup helpers -----------------------------------------------------
    def _build_optimizer(self):
        import torch
        name = self.config.optimizer.lower()
        kwargs = dict(lr=self.config.lr, betas=self.config.betas,
                      weight_decay=self.config.weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(self.net.parameters(), **kwargs)
        if name == "adam":
            return torch.optim.Adam(self.net.parameters(), **kwargs)
        raise ValueError(f"unknown optimizer {self.config.optimizer!r}")

    def _to_tensors(self, states, pis, zs):
        """numpy batch -> device tensors. Pinned + non_blocking on cuda so the
        host->device copy overlaps compute."""
        import torch
        st = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32))
        pt = torch.from_numpy(np.ascontiguousarray(pis, dtype=np.float32))
        zt = torch.from_numpy(np.ascontiguousarray(zs, dtype=np.float32))
        if self.device.type == "cuda":
            st, pt, zt = st.pin_memory(), pt.pin_memory(), zt.pin_memory()
            return (st.to(self.device, non_blocking=True),
                    pt.to(self.device, non_blocking=True),
                    zt.to(self.device, non_blocking=True))
        return st.to(self.device), pt.to(self.device), zt.to(self.device)

    # -- one gradient step -------------------------------------------------
    def train_step(self, states: np.ndarray, pis: np.ndarray, zs: np.ndarray
                   ) -> Dict[str, float]:
        """Single optimizer step on one already-sampled minibatch. Accepts numpy
        (as ReplayBuffer.sample returns) and returns a stats dict with the total
        and both component losses plus the current LR."""
        import torch
        from torch.nn.utils import clip_grad_norm_

        self.net.train()                       # BN in train mode (see docstring)
        st, pt, zt = self._to_tensors(states, pis, zs)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits, value = self.net(st)
            loss, v_loss, p_loss = alphazero_losses(logits, value, pt, zt)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.net.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.grad_clip_norm:
                clip_grad_norm_(self.net.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        self._step += 1

        return {
            "loss": float(loss.detach().item()),
            "value_loss": float(v_loss.detach().item()),
            "policy_loss": float(p_loss.detach().item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "step": self._step,
        }

    # -- many steps over a buffer -----------------------------------------
    def train(self, buffer: ReplayBuffer, num_steps: int,
              batch_size: Optional[int] = None,
              rng: Optional[np.random.Generator] = None,
              on_step=None) -> Dict[str, List[float]]:
        """Sample `num_steps` minibatches from `buffer` and step on each.

        Returns a history dict of per-step lists: loss / value_loss / policy_loss
        / lr — enough for the Stage-5 'both loss terms trend down' check and for
        Stage-6 logging. `on_step(stats)` is an optional per-step hook.
        """
        bs = batch_size or self.config.batch_size
        rng = rng or np.random.default_rng()
        history: Dict[str, List[float]] = {
            "loss": [], "value_loss": [], "policy_loss": [], "lr": [],
        }
        for _ in range(num_steps):
            states, pis, zs = buffer.sample(bs, rng)
            stats = self.train_step(states, pis, zs)
            for k in history:
                history[k].append(stats[k])
            if on_step is not None:
                on_step(stats)
        return history

    # -- inference-side value probe (for the Stage-5 calibration check) ----
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        """Eval-mode value predictions for a batch of encoded states, as a numpy
        (B,) array in [-1,1]. Uses eval() + no_grad (BN running stats), then this
        method restores train() so it does not disturb a training run. Handy for
        the 'won/lost positions get the right sign' spot check."""
        import torch
        self.net.eval()
        try:
            with torch.no_grad():
                st = self._to_tensors(states, np.zeros((len(states), NUM_MOVES), np.float32),
                                      np.zeros((len(states),), np.float32))[0]
                _, value = self.net(st)
                return value.reshape(-1).float().cpu().numpy()
        finally:
            self.net.train()

    # -- checkpoint I/O ----------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Persist model + optimizer + schedule + scaler + config. Needed both
        for crash recovery and for Stage-6's fixed-benchmark eval (you must keep
        past nets to measure against)."""
        import torch
        torch.save({
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "step": self._step,
            "config": asdict(self.config),
        }, path)

    def load_checkpoint(self, path: str, map_location=None) -> None:
        import torch
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.scaler is not None and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self._step = int(ckpt.get("step", 0))