"""
Inference adapter — bridge the competition harness to our model.

The AI Chessathon harness (see agent.py) hands us a FEN string and our clock, and
wants a legal move back in UCI:

    def get_move(fen: str, time_left_ms: int) -> str: ...

This module is the ONLY thing the agent entrypoint needs to call. It owns the
three things that make that bridge correct, and nothing else (no training, no
self-play, no buffer — inference only):

  1. LOAD  — build MarchHare and load trained weights ONCE (at import, off the
     clock), in eval mode with grad disabled. Architecture (channels / #blocks)
     is inferred from the checkpoint so you don't have to hand-match it.
  2. ENCODE — turn the harness FEN into the exact tensor our net expects:
     chess.Board(fen) -> encoding.encode -> (20,8,8), canonicalized to the side
     to move. This is the "correct input for our model" the whole file exists for.
  3. SELECT + UN-FLIP — run the same MCTS we train with (Stage 3), then convert
     the chosen canonical policy index back into a REAL move on the real board
     via moves.index_to_move. This un-flip is the crux: our policy lives in the
     mover-at-bottom frame, but the referee validates the UCI against the actual
     position's legal_moves. index_to_move is the exact inverse of the
     move_to_index the evaluator used, so a legal move is guaranteed for both
     colours (verified for white, black, and promotions).

INFERENCE ≠ SELF-PLAY (what changes vs Stage 4)
-----------------------------------------------
  * NO Dirichlet noise (add_noise=False) — we want the net's honest best move,
    not exploration.
  * GREEDY — take argmax of the visit counts, not a temperature sample.
  * eval() + no_grad — inherited from mcts.make_net_evaluator, which is the
    correct BatchNorm discipline (running stats, never per-call batch stats).

CPU BY DEFAULT
--------------
The platform runs CPU torch (pyproject pins torch==2.13.0 from the cpu index), so
device defaults to "cpu"; pass device="cuda"/"auto" to test locally on a GPU.
Grad is globally disabled and thread count is optionally pinned for steadier CPU
latency.

WARM-UP
-------
Like the numba baseline compiles at import, we run ONE search from the start
position at construction so the first real forward pass / any lazy init is paid
inside the 60 s init budget, not on move one's clock.

PACKAGING (how this reaches the platform)
-----------------------------------------
submission.zip = every root *.py + a weights/ dir (harness/package.py). So drop
these at the fork root: moves.py, encoding.py, network.py, mcts.py, this file,
and an agent.py that calls it; put the checkpoint at weights/<name>.pt. train.py,
self_play.py, game.py and the tests are NOT needed for inference and can be left
out to keep the zip lean.

TORCH-FREE CORE
---------------
Everything except load_model / the ChessInference class body is torch-free (torch
is imported lazily), so encode_fen and select_move can be unit-tested with a mock
evaluator — the same discipline mcts.py uses. That is how this file was verified
here without torch installed.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import chess

from utils.utils_file import encode, NUM_PLANES, move_to_index, index_to_move, NUM_MOVES
from utils.mcts import run_mcts, make_net_evaluator, Evaluator
import torch
from utils.march_hare import MarchHare


# --------------------------------------------------------------------------- #
# Encoding: FEN -> the exact model input
# --------------------------------------------------------------------------- #
def encode_fen(fen: str) -> np.ndarray:
    """Harness FEN -> (20,8,8) float32, canonicalized to the side to move.

    This is precisely encoding.encode applied to chess.Board(fen); the mover is
    always flipped to the bottom, so the tensor matches what the net trained on.
    """
    return encode(chess.Board(fen))


# --------------------------------------------------------------------------- #
# Move selection (evaluator-injected, so it is torch-free testable)
# --------------------------------------------------------------------------- #
def select_move(board: chess.Board,
                evaluate_fn: Evaluator,
                *,
                use_mcts: bool = True,
                num_simulations: int = 100,
                c_puct: float = 2.0,
                fpu: float = 0.0,
                rng: Optional[np.random.Generator] = None) -> chess.Move:
    """Pick a REAL move on `board` (legal in the actual position).

    use_mcts=True  : run Stage-3 MCTS with noise OFF, take the most-visited move
                     (greedy). This is the search-improved choice — the strong one.
    use_mcts=False : skip search and take the network prior's argmax directly.
                     Much faster, much weaker; handy for very low time or debugging.

    Either way the returned object is a chess.Move already un-flipped into the
    board's real frame, so move.uci() is what the harness expects.
    """
    if not use_mcts:
        # priors is {canonical_index -> (real_move, prob)} over the legal moves.
        _value, priors = evaluate_fn(board)
        best_idx = max(priors, key=lambda i: priors[i][1])
        return priors[best_idx][0]

    pi = run_mcts(
        board,
        evaluate_fn=evaluate_fn,
        num_simulations=num_simulations,
        c_puct=c_puct,
        add_noise=False,          # inference: no exploration noise
        fpu=fpu,
        rng=rng,
    )
    # pi is over canonical indices; argmax is the greedy (tau->0) move. Un-flip it
    # back onto the real board — the inverse of the evaluator's move_to_index.
    idx = int(pi.argmax())
    return index_to_move(board, idx)


# --------------------------------------------------------------------------- #
# Checkpoint loading (architecture inferred from the weights)
# --------------------------------------------------------------------------- #
def _infer_architecture(state: Dict[str, "object"]) -> Tuple[int, int]:
    """Read channel count and block count straight off the state_dict, so we
    rebuild the SAME network the checkpoint was trained with without being told.

    channels  : stem conv output channels  = stem.0.weight.shape[0]
    num_blocks : count of spine.<i>.* indices
    """
    import re
    channels = int(state["stem.0.weight"].shape[0])
    block_idxs = set()
    for k in state:
        m = re.match(r"spine\.(\d+)\.", k)
        if m:
            block_idxs.add(int(m.group(1)))
    num_blocks = (max(block_idxs) + 1) if block_idxs else 8
    return channels, num_blocks


def load_model(checkpoint_path: Optional[str] = None,
               *,
               net=None,
               device: str = "cpu",
               channels: Optional[int] = None,
               num_blocks: Optional[int] = None):
    """Build MarchHare and load weights, in eval mode on `device`.

    Pass a `checkpoint_path` (a Trainer.save_checkpoint dict with a "model" key,
    OR a bare state_dict) — architecture is inferred from the weights unless you
    override channels/num_blocks. Or pass an already-built `net` (e.g. for local
    tests) to skip disk entirely.
    """

    if net is not None:
        return net.to(device).eval()

    if checkpoint_path is None:
        raise ValueError("load_model needs either checkpoint_path or net")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if channels is None or num_blocks is None:
        ch, nb = _infer_architecture(state)
        channels = channels or ch
        num_blocks = num_blocks or nb

    net = MarchHare(channels=channels, num_blocks=num_blocks).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def _resolve_device(device: str) -> str:
    """'cpu' / 'cuda' / 'auto'. 'auto' picks cuda when available, else cpu.
    The competition is cpu; this is here for local GPU testing."""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------------------------------------------------------- #
# The inference object the agent holds
# --------------------------------------------------------------------------- #
class ChessInference:
    """Load once, then answer best_move(fen, time_left_ms) per move.

    Construct this at IMPORT time in agent.py (weights load + warm-up happen in
    __init__, inside the init budget). Keep the instance on a module global so it
    survives across the game's moves.
    """

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 *,
                 net=None,
                 device: str = "cpu",
                 num_simulations: int = 100,
                 min_simulations: int = 16,
                 c_puct: float = 2.0,
                 fpu: float = 0.0,
                 use_mcts: bool = True,
                 threads: Optional[int] = None,
                 channels: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 warmup: bool = True):
        import torch

        torch.set_grad_enabled(False)          # inference: never build a graph
        if threads:
            torch.set_num_threads(int(threads))

        self.device = _resolve_device(device)
        self.net = load_model(checkpoint_path, net=net, device=self.device,
                              channels=channels, num_blocks=num_blocks)
        # eval() + no_grad masked evaluator (Stage 2 masking inline). Built once.
        self.evaluate_fn: Evaluator = make_net_evaluator(self.net)

        self.use_mcts = use_mcts
        self.num_simulations = int(num_simulations)
        self.min_simulations = int(min_simulations)
        self.c_puct = float(c_puct)
        self.fpu = float(fpu)

        if warmup:
            self._warmup()

    # -- warm-up: pay first-forward cost off the clock ---------------------
    def _warmup(self) -> None:
        try:
            board = chess.Board()
            # one real search so torch's first forward / any lazy init is paid now
            select_move(board, self.evaluate_fn, use_mcts=self.use_mcts,
                        num_simulations=max(4, min(self.num_simulations, 16)),
                        c_puct=self.c_puct, fpu=self.fpu)
        except Exception as exc:                # never let warm-up sink init
            print(f"[inference] warm-up skipped: {exc}")

    # -- simple, safe time->sims policy -----------------------------------
    def _sims_for_time(self, time_left_ms: Optional[int]) -> int:
        """Coarse guard so we never flag: spend full sims with time in hand, cut
        back when the clock is low. This is deliberately simple — sim count is
        the main strength lever (Stage 7), so tune it here later."""
        base = self.num_simulations
        if time_left_ms is None:
            return base
        if time_left_ms < 2_000:
            return max(self.min_simulations, base // 4)
        if time_left_ms < 10_000:
            return max(self.min_simulations, base // 2)
        return base

    # -- the call the harness ultimately drives ---------------------------
    def best_move(self, fen: str, time_left_ms: Optional[int] = None) -> str:
        """FEN -> legal UCI move. This is what agent.get_move should return."""
        board = chess.Board(fen)
        sims = self._sims_for_time(time_left_ms)
        move = select_move(board, self.evaluate_fn,
                           use_mcts=self.use_mcts,
                           num_simulations=sims,
                           c_puct=self.c_puct,
                           fpu=self.fpu)
        # Safety net: the referee instantly loses us on an illegal string, so if
        # anything ever slips the frame, fall back to a legal move rather than
        # hand back something unplayable.
        if move not in board.legal_moves:
            print(f"[inference] non-legal pick {move.uci()!r}; falling back")
            move = next(iter(board.legal_moves))
        return move.uci()

    # -- inspection helpers (print() is allowed and shown in your log) -----
    def value(self, fen: str) -> float:
        """Net value for the side to move, in [-1,1]."""
        v, _ = self.evaluate_fn(chess.Board(fen))
        return float(v)

    def policy(self, fen: str, top: int = 5) -> List[Tuple[str, float]]:
        """Top-`top` (uci, prior) pairs from the raw policy head (real moves)."""
        _v, priors = self.evaluate_fn(chess.Board(fen))
        ranked = sorted(((mv.uci(), p) for (mv, p) in priors.values()),
                        key=lambda t: t[1], reverse=True)
        return ranked[:top]