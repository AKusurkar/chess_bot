"""
Stage 4 — self-play game generation: the data pump that feeds training.

This is the driver that finally ties the earlier stages together into the thing
that actually produces training data. It owns no new algorithm; it is the loop
the docstring in game.py sketched, made real:

    while not game.is_over():
        pi  = run_mcts(game.board_copy(), ...)   # Stage 3 is the mover
        tau = temperature_for_ply(len(game))     # play-time exploration schedule
        idx = sample_move_index(pi, tau)         # Stage 0/4: pick from pi
        game.record_and_push(idx, pi)            # Stage 0/4: store (x, pi, mover)
    examples = game.finalize()                   # Stage 0/4: attach z per mover

Everything hard was already solved and tested upstream:
  * MCTS (mcts.run_mcts) is the policy-improvement operator and returns the RAW
    tau=1 visit distribution pi. This module does NOT re-touch temperature inside
    the search; it applies it once, here, at move-selection time via
    game.sample_move_index. That is the whole reason run_mcts returns raw visits
    and sample_move_index owns temperature — so it is applied exactly once.
  * Frame consistency (canonicalize-to-mover) is inherited for free: pi is over
    canonical indices, record_and_push stores in that frame, finalize labels z
    from each position's own mover perspective. No colour reasoning happens here.
  * The hard ply cap lives in SelfPlayGame (max_plies); a capped game finalizes
    as a draw (z = 0 everywhere). We just choose the cap and read the result.

WHAT THIS MODULE ADDS OVER game.py
----------------------------------
  1. The self-play *temperature schedule*: tau = 1 for the first ~30 plies
     (exploration), then tau -> 0 (greedy). Temperature changes only which move
     is PLAYED; the full pi is always stored regardless (game.record_and_push).
  2. Self-play *exploration noise* is on by default (add_noise=True) — Dirichlet
     at the root, applied inside run_mcts. This is non-negotiable for self-play:
     it is what lets the system discover moves the current net does not yet rate.
     (Eval matches, Stage 6, will call with add_noise=False + greedy temperature.)
  3. A thin generate_games() loop that builds the network evaluator ONCE and
     reuses it across games, and surfaces per-game length/winner so Stage 6 can
     watch the degeneration signals (game length shrinking, draw rate spiking)
     the plan calls the best early-warning instruments.

DELIBERATELY DEFERRED
---------------------
  * Resignation (end clearly-lost games early, leaving a fraction un-resigned to
    avoid a blind spot). Defer until compute-starved, per the plan. The ply cap
    is a SEPARATE, earlier necessity and IS implemented (via SelfPlayGame).
  * Concurrency / batched leaf evaluation across many games + virtual loss. That
    is the Stage 6 GPU-saturation lift; it slots in behind the SAME evaluate_fn
    seam this module already threads through, so nothing here has to change when
    it lands — generate_games' sequential loop becomes a concurrent one.

BATCHNORM NOTE
--------------
make_net_evaluator puts the network in eval() mode (correct: self-play inference
must use BatchNorm running stats, never per-batch stats). That is a persistent
side effect on the module. Before you run Stage-5 training on the same object,
call net.train() yourself. This module never flips it back.
"""

from typing import List, NamedTuple, Optional

import numpy as np
import chess

from utils.single_game import SelfPlayGame, sample_move_index, TrainingExample
from utils.mcts import run_mcts, make_net_evaluator, Evaluator

# Plan: tau = 1 for ~the first 30 plies, then tau -> 0. Exposed as a default so
# both play_game and any caller can share one number.
DEFAULT_TEMP_CUTOFF_PLY = 30


def temperature_for_ply(ply: int,
                        cutoff: int = DEFAULT_TEMP_CUTOFF_PLY,
                        hot: float = 1.0,
                        cold: float = 0.0) -> float:
    """Self-play temperature schedule.

    `ply` is 0-indexed and is the number of plies ALREADY played (i.e. len(game)
    just before the move about to be chosen). So with cutoff=30, plies 0..29 —
    the first 30 — are 'hot' (tau=1, sample proportional to visit counts) and ply
    30 onward is 'cold' (tau->0, greedy argmax over visits). A hard switch, not a
    ramp, which matches the plan's 'tau=1 ... then tau->0'. cold=0.0 routes to the
    argmax branch of sample_move_index.
    """
    return hot if ply < cutoff else cold


class GameResult(NamedTuple):
    """The product of one self-play game.

    examples  : the [(state, pi, z), ...] to extend the replay buffer with.
    num_plies : how many plies were actually played (len(game)); a degeneration
                signal to track over training (shrinking games flag trouble).
    winner    : chess.WHITE / chess.BLACK / None. None means draw — by rule OR by
                the ply cap (adjudicated draw). Feeds the draw-rate monitor.
    """
    examples: List[TrainingExample]
    num_plies: int
    winner: Optional[bool]


def play_game(net=None,
              *,
              evaluate_fn: Optional[Evaluator] = None,
              num_simulations: int = 64,
              c_puct: float = 2.0,
              dirichlet_alpha: float = 0.3,
              dirichlet_frac: float = 0.25,
              fpu: float = 0.0,
              add_noise: bool = True,
              temp_cutoff_ply: int = DEFAULT_TEMP_CUTOFF_PLY,
              hot_temp: float = 1.0,
              cold_temp: float = 0.0,
              max_plies: int = 512,
              start_fen: Optional[str] = None,
              rng: Optional[np.random.Generator] = None,
              return_result: bool = False):
    """Play ONE full self-play game and return its training examples.

    Provide either `net` (a MarchHare) or a ready `evaluate_fn` (used by the tests
    here, and by anything that already built a batched evaluator). Building the
    evaluator once and passing evaluate_fn avoids re-wrapping the net every ply.

    The single `rng` is threaded into BOTH run_mcts (root Dirichlet noise) and
    sample_move_index (move sampling) so a seed reproduces the whole game from one
    stream. Returns the flat example list, or a GameResult if return_result=True
    (mirroring run_mcts's return_root idiom).
    """
    rng = rng or np.random.default_rng()
    if evaluate_fn is None:
        if net is None:
            raise ValueError("play_game needs either `net` or `evaluate_fn`")
        evaluate_fn = make_net_evaluator(net)  # eval() mode side-effect: see module docstring

    game = SelfPlayGame.new(fen=start_fen, max_plies=max_plies)

    while not game.is_over():
        # MCTS is the mover. It gets a detached copy WITH history so in-search
        # repetition / fifty-move detection works (game.board_copy keeps the stack).
        pi = run_mcts(
            game.board_copy(),
            num_simulations=num_simulations,
            c_puct=c_puct,
            add_noise=add_noise,            # self-play: root Dirichlet ON
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
            fpu=fpu,
            rng=rng,
            evaluate_fn=evaluate_fn,
        )
        # Temperature is applied HERE, exactly once, on the raw visit distribution.
        # len(game) is the 0-indexed ply about to be played.
        temp = temperature_for_ply(len(game), temp_cutoff_ply, hot_temp, cold_temp)
        idx = sample_move_index(pi, temperature=temp, rng=rng)
        # Store the FULL pi (not the temperature-shaped one) + play the sampled move.
        game.record_and_push(idx, pi)

    examples = game.finalize()  # asserts is_over(); attaches z per position's mover
    if return_result:
        return GameResult(examples, len(game), game.outcome_winner())
    return examples


def generate_games(net=None,
                   num_games: int = 1,
                   *,
                   evaluate_fn: Optional[Evaluator] = None,
                   on_game=None,
                   rng: Optional[np.random.Generator] = None,
                   **play_kwargs) -> List[TrainingExample]:
    """Play `num_games` self-play games sequentially; return one flat example list.

    The evaluator is built ONCE here and reused for every game (so the network is
    wrapped / put in eval mode a single time, not per game). `on_game`, if given,
    is called with each GameResult — the hook Stage 6 uses to log game-length and
    draw-rate distributions without this module prescribing a logging format.

    `**play_kwargs` forwards the tuning knobs to play_game (num_simulations,
    c_puct, temp_cutoff_ply, max_plies, ...). Do NOT pass net/evaluate_fn/rng/
    return_result in play_kwargs — they are managed here.

    NOTE: sequential on purpose. Throughput comes later from Stage 6 (many games
    sharing batched leaf evaluations). That swap happens behind evaluate_fn and
    does not change this function's contract.
    """
    rng = rng or np.random.default_rng()
    if evaluate_fn is None:
        if net is None:
            raise ValueError("generate_games needs either `net` or `evaluate_fn`")
        evaluate_fn = make_net_evaluator(net)

    all_examples: List[TrainingExample] = []
    for _ in range(num_games):
        result = play_game(
            evaluate_fn=evaluate_fn,
            rng=rng,
            return_result=True,
            **play_kwargs,
        )
        all_examples.extend(result.examples)
        if on_game is not None:
            on_game(result)
    return all_examples