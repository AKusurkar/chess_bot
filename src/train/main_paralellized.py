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
whole outer-loop structure — is INHERITED UNCHANGED from main.OuterLoop. The one
behavioural difference is that the live per-game on_game stream is gone (workers
return their results as a batch); the same degeneration signals (game-length
mean/min/max, draw rate) are recomputed in the main process, so the logger and
printout see identical numbers.

RUN
---
    python main_parallelized.py

Edit hyperparameters in main.Stage6Config exactly as before — this file reuses
main.CONFIG. Tune NUM_WORKERS below for your machine.

GOTCHAS (all handled here)
--------------------------
  * spawn, not fork: CUDA contexts do not survive fork. The pool uses a spawn
    context and the entrypoint sets the spawn start method under a __main__ guard.
  * torch.set_num_threads(1) per worker: otherwise each worker's torch tries to
    grab every core for its tiny forward passes and they fight over the cores we
    are trying to free.
  * CPU weight snapshot: the state_dict is moved to CPU before crossing the
    process boundary (~24MB per worker per epoch — trivial IPC).
  * Distinct per-worker seeds via SeedSequence(cfg.seed, epoch).spawn(W): games
    decorrelate across workers AND epochs, yet a fixed cfg.seed reproduces the run.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

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
# 12 workers leaves ~4 physical cores for the main process (training + eval run
# there on the GPU) and the OS. Raise/lower and watch where wall-clock stops
# improving — that plateau is the CPU (or the GPU) announcing the new bottleneck.
NUM_WORKERS: int = 12


# --------------------------------------------------------------------------- #
# Worker-side state and functions (top-level so `spawn` can pickle them).
# --------------------------------------------------------------------------- #
# Each worker keeps ONE resident net on the GPU, built once at pool creation, so
# CUDA init and the model object are paid for a single time and reused across all
# epochs (not re-spawned every epoch).
_WORKER_NET = None


def _worker_init(channels: int, num_blocks: int, device: str) -> None:
    """Runs once per worker when the pool is created."""
    import torch

    torch.set_num_threads(1)        # don't let each worker grab all the cores
    torch.set_grad_enabled(False)   # workers are inference-only, forever

    global _WORKER_NET
    net = MarchHare(channels=channels, num_blocks=num_blocks).to(device)
    net.eval()
    _WORKER_NET = net


def _worker_play(state_dict, num_games: int, games_in_flight: int,
                 seed_seq, sp_kwargs: dict, max_forward_batch):
    """Runs once per worker per epoch: load the fresh weights, then generate this
    worker's slice of self-play games and return the GameResult list.

    This is just a single-process batched self-play run — it reuses
    generate_games_batched and the whole batched-MCTS stack UNCHANGED. Several of
    these run at once; that concurrency is the entire speedup.
    """
    global _WORKER_NET
    _WORKER_NET.load_state_dict(state_dict)   # in-place copy onto the resident GPU net
    _WORKER_NET.eval()

    evaluate_batch = make_batched_net_evaluator(_WORKER_NET, max_batch=max_forward_batch)
    rng = np.random.default_rng(seed_seq)

    return generate_games_batched(
        evaluate_batch, num_games,
        games_in_flight=games_in_flight,
        rng=rng, progress=False,              # no tqdm from workers (would clobber)
        **sp_kwargs,
    )


# --------------------------------------------------------------------------- #
# The parallel loop: inherit everything, override only self-play.
# --------------------------------------------------------------------------- #
class ParallelOuterLoop(OuterLoop):
    """main.OuterLoop with the self-play phase fanned out across a worker pool.

    Construct once, call .run(). The persistent pool is built in __init__ (CUDA
    init is paid once) and torn down in run()'s finally.
    """

    def __init__(self, config=None):
        super().__init__(config)              # builds trainer, buffer, eval, logging, ...
        self._sp_epoch = 0

        ctx = mp.get_context("spawn")
        print(f"[parallel] spawning {NUM_WORKERS} self-play workers on {self.device} ...")
        self._pool = ctx.Pool(
            processes=NUM_WORKERS,
            initializer=_worker_init,
            initargs=(self.cfg.channels, self.cfg.num_blocks, str(self.device)),
        )
        print("[parallel] worker pool ready.")

    # -- the one overridden phase -----------------------------------------
    def _self_play_phase(self) -> Dict[str, float]:
        # Parity with base: BN running stats (which live in the state_dict) are
        # what the workers will use for inference.
        self.net.eval()

        # Fresh weights this epoch, on CPU for cheap cross-process transfer.
        state_dict = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}

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
            (state_dict, chunk, min(chunk, self.cfg.games_in_flight),
             seed, sp_kwargs, self.cfg.max_forward_batch)
            for chunk, seed in zip(chunks, seeds)
        ]

        print(f"[parallel] {games} games across {len(jobs)} workers "
              f"(chunks={chunks}, in_flight/worker <= {self.cfg.games_in_flight})")

        # Blocks until every worker finishes — a clean barrier before training.
        nested = self._pool.starmap(_worker_play, jobs)
        results = [r for sub in nested for r in sub]

        # Funnel every worker's data into the single main-process buffer.
        for r in results:
            self.buffer.extend(r.examples)

        # Recompute the SAME stats base derives from its live on_game stream, so
        # the printout, the tensorboard log, and the degeneration signals are
        # byte-for-byte the ones main.py would have produced.
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


def run_stage6_parallel(config=None) -> None:
    """Convenience entrypoint mirroring main.run_stage6."""
    ParallelOuterLoop(config or CONFIG).run()


if __name__ == "__main__":
    # CUDA + multiprocessing requires spawn; must be set under the __main__ guard.
    mp.set_start_method("spawn", force=True)
    run_stage6_parallel(CONFIG)