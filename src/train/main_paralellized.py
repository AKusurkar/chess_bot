"""
main_parallelized.py — the SAME Stage-6 run as main.py, with self-play spread
across CPU worker processes.

WHY THIS EXISTS
---------------
Self-play is CPU-bound: the PUCT tree-walk, python-chess legal-move generation,
board.copy()/push(), and draw detection all run in single-threaded Python, while
the GPU sits ~80% idle (each batched forward pass is a sub-millisecond blip). The
games are embarrassingly parallel and independent, so the fastest win available
is to stop doing all that CPU work on one core.

DESIGN: centralized training, distributed data generation
---------------------------------------------------------
  * TRAINING stays in the main process, on the GPU, single-owner — exactly as
    main.py does it. Nobody trains in a worker; there are no gradients, no
    optimizer, no shared weights to synchronize.
  * The REPLAY BUFFER lives in the main process only. Workers never touch it.
    They generate GameResult objects and ship them back; the main process does
    buffer.extend(...). No locks, no concurrent writers, no races.
  * Each epoch: main trains -> main broadcasts a fresh CPU weight snapshot to
    every worker -> workers run inference-only self-play on their slice of games
    with those weights -> results funnel back into the one buffer. Repeat.

WHAT CHANGES vs main.py
-----------------------
Only OuterLoop._self_play_phase is overridden. Everything else — the buffer, the
Trainer, the fixed-checkpoint eval curve, baselines, logging, checkpointing, the
whole outer-loop structure — is INHERITED UNCHANGED from main.OuterLoop.

PROGRESS BAR
------------
The live per-game self-play tqdm bar IS preserved: each worker pushes a tick to a
shared Manager queue as every game finishes (via generate_games_batched's on_game
hook), and the main process drains that queue into a bar (total = games_per_epoch)
while starmap_async runs. Same per-game bar main.py shows. The degeneration
signals (game-length mean/min/max, draw rate) are recomputed from the returned
GameResults, byte-for-byte what main.py logs.

RUN
---
    python main_parallelized.py

Edit hyperparameters in main.Stage6Config exactly as before — this file reuses
main.CONFIG. Tune NUM_WORKERS below for your machine.

FILE-DESCRIPTOR / SHARING NOTE  (fixes OSError: [Errno 24] Too many open files)
------------------------------------------------------------------------------
Passing torch tensors through a Pool triggers torch's own tensor-sharing path,
which under the default 'file_descriptor' strategy holds one OPEN FD per shared
tensor. Broadcasting a full state_dict to every worker each epoch exhausts the FD
limit and wedges the pool. Two defenses here:
  1. ROOT FIX — the weight snapshot is converted to plain NUMPY before it crosses
     the process boundary (ordinary pickle, copied into the pipe, no shared
     memory, no FDs) and rebuilt into a tensor state_dict inside the worker.
  2. Belt-and-suspenders — file_system sharing strategy + raising the FD soft
     limit to the hard limit.

GOTCHAS (all handled here)
--------------------------
  * spawn, not fork: CUDA contexts do not survive fork. Pool + Manager use a spawn
    context; the entrypoint sets the spawn start method under a __main__ guard.
  * torch.set_num_threads(1) per worker: otherwise each worker's torch tries to
    grab every core for its tiny forward passes and they fight over the cores we
    are trying to free.
  * Distinct per-worker seeds via SeedSequence(cfg.seed, epoch).spawn(W).
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

import queue as _queue
from collections import Counter
from typing import Dict, Optional

import numpy as np
import torch.multiprocessing as mp

from utils.march_hare import MarchHare
from utils.batched_mcts import make_batched_net_evaluator
from utils.batched_games import generate_games_batched

# Reuse the base loop and its config/entrypoint verbatim — this is what makes the
# run "the exact same thing as main, but parallelized".
from main import OuterLoop, CONFIG


# =============================================================================
# WORKER COUNT — the only machine-specific knob. Edit if your hardware differs.
# =============================================================================
# This box has 16 PHYSICAL cores (32 logical w/ hyperthreading). Self-play is
# CPU-bound on the tree-walk, which scales with physical cores, not hyperthreads.
# 8 workers is the SAFE start on a 10GB card: each worker opens its own CUDA
# context (~0.5-1GB of GPU memory BEFORE any real work), so worker count is gated
# by GPU context memory, not by your cores. Check nvidia-smi after the pool spawns
# and raise toward 12 if there's headroom.
NUM_WORKERS: int = 8


# --------------------------------------------------------------------------- #
# Process-wide FD / sharing hardening (call in main AND in each worker).
# --------------------------------------------------------------------------- #
def _harden_fd_limits() -> None:
    """Avoid 'Too many open files' from torch multiprocessing tensor sharing."""
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # raise soft -> hard
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Worker-side state and functions (top-level so `spawn` can pickle them).
# --------------------------------------------------------------------------- #
# Each worker keeps ONE resident net on the GPU, built once at pool creation, so
# CUDA init and the model object are paid for a single time and reused across all
# epochs. _WORKER_Q is the shared progress queue used to drive the main-process bar.
_WORKER_NET = None
_WORKER_Q = None


def _worker_init(channels: int, num_blocks: int, device: str, progress_q) -> None:
    """Runs once per worker when the pool is created."""
    import torch

    _harden_fd_limits()
    torch.set_num_threads(1)        # don't let each worker grab all the cores
    torch.set_grad_enabled(False)   # workers are inference-only, forever

    global _WORKER_NET, _WORKER_Q
    _WORKER_Q = progress_q
    net = MarchHare(channels=channels, num_blocks=num_blocks).to(device)
    net.eval()
    _WORKER_NET = net


def _tick(_result) -> None:
    """on_game hook: one push per finished game -> one tick on the main bar."""
    try:
        if _WORKER_Q is not None:
            _WORKER_Q.put(1)
    except Exception:
        pass


def _worker_play(state_np, num_games: int, games_in_flight: int,
                 seed_seq, sp_kwargs: dict, max_forward_batch):
    """Runs once per worker per epoch: load the fresh weights (arriving as numpy,
    rebuilt into a tensor state_dict here), then generate this worker's slice of
    self-play games and return the GameResult list. Each finished game pushes a
    tick to the shared progress queue.
    """
    import torch

    global _WORKER_NET
    state_dict = {k: torch.from_numpy(v) for k, v in state_np.items()}
    _WORKER_NET.load_state_dict(state_dict)
    _WORKER_NET.eval()

    evaluate_batch = make_batched_net_evaluator(_WORKER_NET, max_batch=max_forward_batch)
    rng = np.random.default_rng(seed_seq)

    return generate_games_batched(
        evaluate_batch, num_games,
        games_in_flight=games_in_flight,
        rng=rng, progress=False,              # no per-worker tqdm; main draws the bar
        on_game=_tick,                        # per-game progress -> main bar
        **sp_kwargs,
    )


# --------------------------------------------------------------------------- #
# The parallel loop: inherit everything, override only self-play.
# --------------------------------------------------------------------------- #
class ParallelOuterLoop(OuterLoop):
    """main.OuterLoop with the self-play phase fanned out across a worker pool."""

    def __init__(self, config=None):
        _harden_fd_limits()
        super().__init__(config)              # builds trainer, buffer, eval, logging, ...
        self._sp_epoch = 0

        self._ctx = mp.get_context("spawn")
        # Manager queue: proxy-picklable, safe to hand to spawned workers, and used
        # to stream per-game progress back for the self-play bar.
        self._manager = self._ctx.Manager()
        self._progress_q = self._manager.Queue()

        print(f"[parallel] spawning {NUM_WORKERS} self-play workers on {self.device} ...")
        self._pool = self._ctx.Pool(
            processes=NUM_WORKERS,
            initializer=_worker_init,
            initargs=(self.cfg.channels, self.cfg.num_blocks, str(self.device),
                      self._progress_q),
        )
        print("[parallel] worker pool ready.")

    # -- the one overridden phase -----------------------------------------
    def _self_play_phase(self) -> Dict[str, float]:
        from tqdm.auto import tqdm

        # Parity with base: BN running stats (in the state_dict) drive worker
        # inference. Weights cross the pipe as NUMPY (the FD-leak fix).
        self.net.eval()
        state_np = {k: v.detach().cpu().numpy()
                    for k, v in self.net.state_dict().items()}

        # Split games_per_epoch evenly; drop empty chunks if games < workers.
        games = self.cfg.games_per_epoch
        base, rem = divmod(games, NUM_WORKERS)
        chunks = [base + (1 if i < rem else 0) for i in range(NUM_WORKERS)]
        chunks = [c for c in chunks if c > 0]

        # Distinct, reproducible seed per (epoch, worker).
        self._sp_epoch += 1
        seeds = np.random.SeedSequence([self.cfg.seed, self._sp_epoch]).spawn(len(chunks))

        sp_kwargs = dict(
            num_simulations=self.cfg.num_simulations,
            c_puct=self.cfg.c_puct,
            fpu=self.cfg.fpu,
            dirichlet_alpha=self.cfg.dirichlet_alpha,
            dirichlet_frac=self.cfg.dirichlet_frac,
            temp_cutoff_ply=self.cfg.temp_cutoff_ply,
            max_plies=self.cfg.max_plies,
            add_noise=True,                    # self-play: root Dirichlet ON
        )

        # cfg.games_in_flight is now a PER-WORKER concurrency cap; total concurrency
        # across the pool is roughly len(chunks) x that. The GPU has the headroom.
        jobs = [
            (state_np, chunk, min(chunk, self.cfg.games_in_flight),
             seed, sp_kwargs, self.cfg.max_forward_batch)
            for chunk, seed in zip(chunks, seeds)
        ]

        # Dispatch asynchronously, then drain the progress queue into the bar while
        # the workers run. Bar total = total games this epoch (one tick per game).
        async_res = self._pool.starmap_async(_worker_play, jobs)
        bar = tqdm(total=games, desc="self-play", unit="game", leave=False)

        def _drain() -> None:
            try:
                while True:
                    self._progress_q.get_nowait()
                    bar.update(1)
            except _queue.Empty:
                pass

        while not async_res.ready():
            _drain()
            async_res.wait(timeout=0.2)       # brief block so we're not busy-spinning
        _drain()                              # catch any final ticks
        bar.close()

        nested = async_res.get()              # ordered per-job results; re-raises worker errors
        results = [r for sub in nested for r in sub]

        # Funnel every worker's data into the single main-process buffer.
        for r in results:
            self.buffer.extend(r.examples)

        # Recompute the SAME stats base derives from its live on_game stream.
        lengths = np.array([r.num_plies for r in results], dtype=np.float64)
        winners = Counter(r.winner for r in results)          # True=W, False=B, None=draw
        examples_added = int(sum(len(r.examples) for r in results))
        n_games = len(results)
        return {
            "selfplay/games": float(n_games),
            "selfplay/examples_added": float(examples_added),
            "selfplay/buffer_size": float(len(self.buffer)),
            "selfplay/game_len_mean": float(lengths.mean()),
            "selfplay/game_len_min": float(lengths.min()),
            "selfplay/game_len_max": float(lengths.max()),
            "selfplay/white_wins": float(winners.get(True, 0)),
            "selfplay/black_wins": float(winners.get(False, 0)),
            "selfplay/draws": float(winners.get(None, 0)),
            "selfplay/draw_rate": float(winners.get(None, 0) / n_games) if n_games else 0.0,
        }

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> None:
        try:
            super().run()
        finally:
            self._close_pool()

    def _close_pool(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()
            pool.join()
            self._pool = None
        manager = getattr(self, "_manager", None)
        if manager is not None:
            manager.shutdown()
            self._manager = None


def run_stage6_parallel(config=None) -> None:
    """Convenience entrypoint mirroring main.run_stage6."""
    ParallelOuterLoop(config or CONFIG).run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    _harden_fd_limits()
    run_stage6_parallel(CONFIG)