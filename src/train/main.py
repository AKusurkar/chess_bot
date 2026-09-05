"""
Stage 6 — the outer loop & evaluation: where the whole pipeline finally learns.

    init random network
    repeat:
        generate a batch of self-play games  -> append (s, pi, z) to the buffer
        train on N minibatches sampled from the buffer
        every K epochs: play the live net vs a FIXED checkpoint (and the
                        competition baselines), log win-rate / Elo signal

This module owns the loop and the instrumentation; every hard part was solved and
tested upstream. It wires together:
  * batched_self_play.generate_games_batched — the concurrent data pump (Stage 6
    GPU-saturation lift), built on batched_mcts.
  * train.Trainer / ReplayBuffer — the Stage-5 loss/optimizer, constructed ONCE so
    Adam's moments and the LR schedule survive across epochs (the plan is explicit
    that these must not reset every round).
  * arena — the honest strength monitor: the fixed-checkpoint curve (the one test
    that matters most) plus one game against each baseline as an external anchor.

HOW TO RUN
----------
Edit the hyperparameters in the Stage6Config block just below the imports, then:

    python outer_loop.py

No command-line arguments, no wandb. TensorBoard logging stays on by default
(set use_tensorboard = False in the config to turn it off).

THE MEASUREMENT DISCIPLINE (plan, Stage 6)
------------------------------------------
  * Measure against a FIXED old checkpoint, never self-play score — both sides
    improving makes self-play score look flat while you are actually getting
    stronger. The reference defaults to the initial random net and stays fixed
    (set reference_update_every > 0 to advance it later).
  * Force eval diversity (random openings + alternating colours) — done in arena,
    or "win rate over N games" collapses to one game repeated N.
  * Watch the degeneration signals — game length shrinking and draw rate spiking
    usually flag self-play collapse BEFORE the eval curve does. Both are logged
    every epoch.

BATCHNORM DISCIPLINE (pairs with self_play.py / train.py)
---------------------------------------------------------
Self-play and eval run the net in eval() (BN running stats); training runs it in
train() (BN batch stats). We set net.eval() before every self-play and eval phase;
Trainer.train_step sets net.train() itself. Alternating on the SAME net object is
therefore safe — the evaluators read the live weights, so improvement propagates
without rebuilding anything.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getenv("PYTHONPATH"))

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.march_hare import MarchHare
from train.train_file import Trainer, TrainConfig, ReplayBuffer, select_device
from utils.mcts import make_net_evaluator
from utils.batched_mcts import make_batched_net_evaluator
from utils.batched_games import generate_games_batched
from train.eval import (
    make_net_agent,
    load_baseline_agents,
    play_match,
    play_vs_baselines,
)


# =============================================================================
# HYPERPARAMETERS — edit these, then just run:  python outer_loop.py
# =============================================================================
# Defaults target the plan's Stage-7 config (8 blocks / 96 filters / 64 sims).
# For a quick "does the loop actually learn?" proof-of-life run, shrink to
# something like:
#     channels=64, num_blocks=4, num_epochs=40,
#     games_per_epoch=24, games_in_flight=16, num_simulations=30, max_plies=200,
#     buffer_capacity=50_000, min_buffer_to_train=500,
#     train_steps_per_epoch=100, batch_size=256,
#     eval_every=2, eval_games=12, eval_num_simulations=40, eval_max_plies=300
# =============================================================================
@dataclass
class Stage6Config:
    """Every knob for the outer loop. Edit the defaults here."""

    # -- run identity / IO -------------------------------------------------
    run_name: str = "march_hare"
    checkpoint_dir: str = "model_weights"
    tensorboard_dir: str = "runs"
    # Competition repo baselines (each a dir with agent.py): fork lives under /external.
    external_baselines_dir: str = "external/aichessathon-starter/baselines"
    seed: int = 0
    device: Optional[str] = None            # None = auto (cuda if available), or "cpu" / "cuda"

    # -- network -----------------------------------------------------------
    channels: int = 96
    num_blocks: int = 8

    # -- outer loop --------------------------------------------------------
    num_epochs: int = 200
    checkpoint_every: int = 5               # save march_hare_epoch_{n}.pt every N epochs

    # -- self-play (per epoch) --------------------------------------------
    games_per_epoch: int = 64
    games_in_flight: int = 32               # THE parallelism knob (batched leaf evals)
    num_simulations: int = 64
    c_puct: float = 2.0
    fpu: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temp_cutoff_ply: int = 30
    max_plies: int = 512
    max_forward_batch: Optional[int] = None  # cap net forward-pass size (peak memory)

    # -- replay buffer / training -----------------------------------------
    buffer_capacity: int = 500_000
    min_buffer_to_train: int = 2_000        # don't train until there's real data
    train_steps_per_epoch: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    grad_clip_norm: Optional[float] = None
    lr_step_size: Optional[int] = None
    lr_gamma: float = 0.1
    use_amp: bool = False

    # -- evaluation --------------------------------------------------------
    eval_every: int = 5
    eval_games: int = 20                    # vs the fixed reference checkpoint
    eval_num_simulations: int = 100         # inference-strength sims for eval play
    eval_opening_plies: int = 4             # random opening plies for diversity
    eval_move_time_ms: int = 120_000        # generous fixed budget (no wall clock in eval)
    eval_max_plies: int = 600               # matches the platform ply cap
    reference_update_every: int = 0         # 0 = reference stays the initial random net
    play_baselines: bool = True
    baseline_names: Optional[List[str]] = None   # None = auto-discover all in the dir
    baseline_games_each: int = 1            # "just one game is fine"

    # -- logging -----------------------------------------------------------
    use_tensorboard: bool = True
    log_train_steps: bool = False           # also stream per-step loss to tensorboard


# The single config instance the run uses. Edit the defaults above, or override
# individual fields right here, e.g. CONFIG = Stage6Config(num_epochs=300, device="cuda").
CONFIG = Stage6Config()


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class OuterLoop:
    """Holds the persistent state (net, trainer, buffer, evaluators, loggers) and
    runs the Stage-6 cycle. Construct once, call .run()."""

    def __init__(self, config: Optional[Stage6Config] = None):
        import torch

        self.cfg = config or Stage6Config()
        self._seed_everything(self.cfg.seed)

        self.device = select_device(self.cfg.device)
        print(f"[stage6] device = {self.device}")

        # Network + Stage-5 trainer (built ONCE -> optimizer/schedule persist).
        net = MarchHare(channels=self.cfg.channels, num_blocks=self.cfg.num_blocks)
        self.trainer = Trainer(net, self._train_config())     # moves net onto device
        self.net = self.trainer.net

        self.buffer = ReplayBuffer(self.cfg.buffer_capacity)
        self.rng = np.random.default_rng(self.cfg.seed)

        # Evaluators over the LIVE net (read current weights every call):
        #   * batched -> self-play data pump (GPU saturation)
        #   * single  -> arena games (one board at a time)
        self.selfplay_eval = make_batched_net_evaluator(self.net, max_batch=self.cfg.max_forward_batch)
        self.arena_eval = make_net_evaluator(self.net)

        # Fixed reference opponent for the upward curve (initially the random net).
        self.ref_net = None
        self.ref_agent = None
        self._snapshot_reference()

        # Competition baselines, loaded from the forked repo under /external.
        self.baseline_agents = {}
        if self.cfg.play_baselines:
            self.baseline_agents = load_baseline_agents(
                self.cfg.external_baselines_dir, self.cfg.baseline_names)
            print(f"[stage6] baselines loaded: {list(self.baseline_agents) or '(none)'}")

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.writer = None
        self._init_logging()

    # -- setup helpers -----------------------------------------------------
    def _seed_everything(self, seed: int) -> None:
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _train_config(self) -> TrainConfig:
        c = self.cfg
        return TrainConfig(
            lr=c.lr, weight_decay=c.weight_decay, optimizer=c.optimizer,
            batch_size=c.batch_size, grad_clip_norm=c.grad_clip_norm,
            lr_step_size=c.lr_step_size, lr_gamma=c.lr_gamma,
            use_amp=c.use_amp, device=c.device,
        )

    def _snapshot_reference(self) -> None:
        """Freeze the current weights as the fixed eval opponent."""
        import copy
        ref = MarchHare(channels=self.cfg.channels, num_blocks=self.cfg.num_blocks).to(self.device)
        ref.load_state_dict(copy.deepcopy(self.net.state_dict()))
        ref.eval()
        self.ref_net = ref
        ref_eval = make_net_evaluator(ref)
        self.ref_agent = make_net_agent(
            ref_eval, use_mcts=True, num_simulations=self.cfg.eval_num_simulations,
            c_puct=self.cfg.c_puct, fpu=self.cfg.fpu, rng=self.rng)

    def _init_logging(self) -> None:
        if self.cfg.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.tensorboard_dir, self.cfg.run_name))

    def _close_logging(self) -> None:
        if self.writer is not None:
            self.writer.close()

    # -- phase 1: self-play -----------------------------------------------
    def _self_play_phase(self) -> Dict[str, float]:
        self.net.eval()                               # BN running stats for inference
        results = generate_games_batched(
            self.selfplay_eval, self.cfg.games_per_epoch,
            games_in_flight=self.cfg.games_in_flight,
            num_simulations=self.cfg.num_simulations,
            c_puct=self.cfg.c_puct, fpu=self.cfg.fpu,
            dirichlet_alpha=self.cfg.dirichlet_alpha,
            dirichlet_frac=self.cfg.dirichlet_frac,
            temp_cutoff_ply=self.cfg.temp_cutoff_ply,
            max_plies=self.cfg.max_plies,
            add_noise=True, rng=self.rng, progress=True,
        )
        for r in results:
            self.buffer.extend(r.examples)

        lengths = np.array([r.num_plies for r in results], dtype=np.float64)
        winners = Counter(r.winner for r in results)          # True=W, False=B, None=draw
        examples_added = int(sum(len(r.examples) for r in results))
        games = len(results)
        return {
            "selfplay/games": float(games),
            "selfplay/examples_added": float(examples_added),
            "selfplay/buffer_size": float(len(self.buffer)),
            "selfplay/game_len_mean": float(lengths.mean()),
            "selfplay/game_len_min": float(lengths.min()),
            "selfplay/game_len_max": float(lengths.max()),
            "selfplay/white_wins": float(winners.get(True, 0)),
            "selfplay/black_wins": float(winners.get(False, 0)),
            "selfplay/draws": float(winners.get(None, 0)),
            "selfplay/draw_rate": float(winners.get(None, 0) / games) if games else 0.0,
        }

    # -- phase 2: training -------------------------------------------------
    def _train_phase(self, epoch: int) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.min_buffer_to_train:
            print(f"[stage6] buffer {len(self.buffer)} < {self.cfg.min_buffer_to_train}; skipping training")
            return {"train/skipped": 1.0}

        from tqdm.auto import tqdm
        agg = {"loss": [], "value_loss": [], "policy_loss": [], "lr": []}
        bar = tqdm(range(self.cfg.train_steps_per_epoch), desc="train", unit="step", leave=False)
        for _ in bar:
            states, pis, zs = self.buffer.sample(self.cfg.batch_size, self.rng)
            stats = self.trainer.train_step(states, pis, zs)   # sets net.train() (BN batch stats)
            for k in agg:
                agg[k].append(stats[k])
            bar.set_postfix(loss=f"{stats['loss']:.3f}")
            if self.cfg.log_train_steps:
                self._log({f"train_step/{k}": stats[k] for k in ("loss", "value_loss", "policy_loss")},
                          step=stats["step"])
        bar.close()

        return {
            "train/loss": float(np.mean(agg["loss"])),
            "train/value_loss": float(np.mean(agg["value_loss"])),
            "train/policy_loss": float(np.mean(agg["policy_loss"])),
            "train/lr": float(agg["lr"][-1]),
            "train/skipped": 0.0,
        }

    # -- phase 3: evaluation ----------------------------------------------
    def _eval_phase(self) -> Dict[str, float]:
        self.net.eval()                               # honest opinion, BN running stats
        net_agent = make_net_agent(
            self.arena_eval, use_mcts=True, num_simulations=self.cfg.eval_num_simulations,
            c_puct=self.cfg.c_puct, fpu=self.cfg.fpu, rng=self.rng)

        stats: Dict[str, float] = {}

        # The one that matters most: win-rate vs the FIXED reference checkpoint.
        ref = play_match(net_agent, self.ref_agent, games=self.cfg.eval_games,
                         rng=self.rng, opening_plies=self.cfg.eval_opening_plies,
                         max_plies=self.cfg.eval_max_plies,
                         move_time_ms=self.cfg.eval_move_time_ms,
                         progress=True, desc="vs-reference")
        stats["eval/reference_score"] = ref.score
        stats["eval/reference_win_rate"] = ref.win_rate
        stats["eval/reference_wins"] = float(ref.wins)
        stats["eval/reference_draws"] = float(ref.draws)
        stats["eval/reference_losses"] = float(ref.losses)
        stats["_ref_match"] = ref                      # kept for the printout; stripped before logging

        # External anchor: one game vs each competition baseline.
        if self.baseline_agents:
            per = play_vs_baselines(
                net_agent, self.baseline_agents, rng=self.rng,
                games_per_baseline=self.cfg.baseline_games_each,
                max_plies=self.cfg.eval_max_plies,
                move_time_ms=self.cfg.eval_move_time_ms, progress=True)
            baseline_scores = []
            for name, m in per.items():
                stats[f"baseline/{name}_score"] = m.score
                stats[f"baseline/{name}_wins"] = float(m.wins)
                stats[f"baseline/{name}_draws"] = float(m.draws)
                stats[f"baseline/{name}_losses"] = float(m.losses)
                baseline_scores.append(m.score)
            stats["baseline/mean_score"] = float(np.mean(baseline_scores)) if baseline_scores else 0.0
            stats["_baseline_matches"] = per
        return stats

    # -- printing ----------------------------------------------------------
    def _print_epoch(self, epoch: int, stats: Dict[str, float]) -> None:
        from tqdm.auto import tqdm
        lines = [f"===== epoch {epoch}/{self.cfg.num_epochs} ====="]
        lines.append(
            "  self-play : {g:.0f} games | len mean/min/max {lm:.0f}/{ln:.0f}/{lx:.0f} | "
            "W/B/draw {w:.0f}/{b:.0f}/{d:.0f} | draw-rate {dr:.0%} | +{ex:.0f} ex -> buffer {bz:.0f}".format(
                g=stats["selfplay/games"], lm=stats["selfplay/game_len_mean"],
                ln=stats["selfplay/game_len_min"], lx=stats["selfplay/game_len_max"],
                w=stats["selfplay/white_wins"], b=stats["selfplay/black_wins"],
                d=stats["selfplay/draws"], dr=stats["selfplay/draw_rate"],
                ex=stats["selfplay/examples_added"], bz=stats["selfplay/buffer_size"]))
        if stats.get("train/skipped", 0.0) >= 1.0:
            lines.append("  training  : skipped (buffer warming up)")
        else:
            lines.append(
                "  training  : loss {l:.4f} (value {v:.4f} + policy {p:.4f}) | lr {lr:.2e}".format(
                    l=stats["train/loss"], v=stats["train/value_loss"],
                    p=stats["train/policy_loss"], lr=stats["train/lr"]))
        if "eval/reference_score" in stats:
            lines.append(
                "  eval      : vs reference {s:.1%}  (+{w} ={d} -{l})".format(
                    s=stats["eval/reference_score"],
                    w=int(stats["eval/reference_wins"]), d=int(stats["eval/reference_draws"]),
                    l=int(stats["eval/reference_losses"])))
            per = stats.get("_baseline_matches")
            if per:
                anchor = "  baselines : " + " | ".join(
                    f"{name} {m.score:.0%} (+{m.wins}={m.draws}-{m.losses})" for name, m in per.items())
                lines.append(anchor)
        tqdm.write("\n".join(lines))

    # -- logging (tensorboard) --------------------------------------------
    def _log(self, stats: Dict[str, float], step: int) -> None:
        if self.writer is None:
            return
        # Strip private (non-scalar) keys used only for the printout.
        scalars = {k: float(v) for k, v in stats.items()
                   if not k.startswith("_") and isinstance(v, (int, float, bool, np.floating, np.integer))}
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)

    # -- checkpoints -------------------------------------------------------
    def _save_checkpoint(self, epoch: int) -> str:
        path = os.path.join(self.cfg.checkpoint_dir, f"march_hare_epoch_{epoch}.pt")
        self.trainer.save_checkpoint(path)
        print(f"[stage6] saved checkpoint {path}")
        return path

    # -- the loop ----------------------------------------------------------
    def run(self) -> None:
        from tqdm.auto import tqdm

        # epoch 0: persist the random init so there is always a benchmark on disk.
        self._save_checkpoint(0)

        epochs = tqdm(range(1, self.cfg.num_epochs + 1), desc="epochs", unit="epoch")
        for epoch in epochs:
            stats: Dict[str, float] = {}
            stats.update(self._self_play_phase())
            stats.update(self._train_phase(epoch))
            if epoch % self.cfg.eval_every == 0:
                stats.update(self._eval_phase())

            self._print_epoch(epoch, stats)
            self._log(stats, step=epoch)

            if epoch % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(epoch)

            if self.cfg.reference_update_every and epoch % self.cfg.reference_update_every == 0:
                self._snapshot_reference()
                print(f"[stage6] reference advanced to epoch {epoch}")

        self._save_checkpoint(self.cfg.num_epochs)
        self._close_logging()
        print("[stage6] done.")


def run_stage6(config: Optional[Stage6Config] = None) -> None:
    """Convenience entrypoint: build the loop and run it."""
    OuterLoop(config or CONFIG).run()


if __name__ == "__main__":
    run_stage6()