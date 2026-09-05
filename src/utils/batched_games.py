"""
Stage 6 (self-play) — batched self-play generation.

self_play.generate_games plays games ONE AT A TIME (correct, but ~3% GPU util).
This module keeps up to `games_in_flight` games alive at once and steps them in
lockstep so their leaf evaluations batch into shared forward passes (see
batched_mcts). It is the concurrent replacement the Stage-4 docstrings promised —
"it slots in behind the SAME evaluate_fn seam" — and it changes nothing about the
algorithm: same temperature schedule, same Dirichlet-at-root exploration, same
per-mover z labeling, same GameResult contract as self_play.play_game.

THE LOOP
--------
    fill up to G slots with fresh games
    while any slot is alive:
        run ONE batched MCTS over all live slots        (batched leaf evals)
        each slot: sample a move from its pi at the ply's temperature, play it
        finalize + emit any slot that just ended; refill from the fresh-game queue

Refilling keeps the batch full instead of letting finished games idle to the tail
(a game that ends at ply 40 would otherwise waste a slot until the 512-ply cap),
which is where most of the throughput actually comes from at low sim counts.

Each game keeps its OWN search tree; trees are never shared, so no virtual loss is
needed (see batched_mcts for why). The only thing shared across games is the
network forward pass.
"""

from typing import Callable, List, Optional

import numpy as np

from utils.single_game import SelfPlayGame, sample_move_index
from utils.self_play import GameResult, temperature_for_ply, DEFAULT_TEMP_CUTOFF_PLY
from utils.batched_mcts import run_batched_mcts, BatchEvaluator


def generate_games_batched(evaluate_batch: BatchEvaluator,
                           num_games: int,
                           *,
                           games_in_flight: int = 32,
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
                           on_game: Optional[Callable[[GameResult], None]] = None,
                           progress: bool = False) -> List[GameResult]:
    """Play `num_games` self-play games, up to `games_in_flight` concurrently, and
    return one GameResult per game (order: completion order, not start order).

    `evaluate_batch` is a batched evaluator (batched_mcts.make_batched_net_evaluator)
    — built ONCE by the caller and reused, so the net is wrapped a single time and
    every game shares its forward passes. `on_game` is called with each GameResult
    as it finishes (the hook Stage 6 uses to stream degeneration signals — game
    length shrinking, draw rate spiking — into the logger without waiting for the
    whole batch). Set `progress=True` for a tqdm bar over completed games.

    Everything else mirrors self_play.play_game exactly: temperature is applied
    HERE, once, on the raw visit distribution; the FULL pi is stored regardless of
    which move temperature picks; Dirichlet noise is on for self-play.
    """
    rng = rng or np.random.default_rng()

    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=num_games, desc="self-play", unit="game", leave=False)

    results: List[GameResult] = []
    slots: List[SelfPlayGame] = []
    started = 0

    def _new_game() -> SelfPlayGame:
        return SelfPlayGame.new(fen=start_fen, max_plies=max_plies)

    # Initial fill.
    while started < num_games and len(slots) < games_in_flight:
        slots.append(_new_game())
        started += 1

    while slots:
        # ONE batched MCTS round over every live game. Each game hands MCTS a
        # detached copy WITH history so in-search repetition/50-move detection works.
        pis = run_batched_mcts(
            [g.board_copy() for g in slots],
            evaluate_batch,
            num_simulations=num_simulations,
            c_puct=c_puct,
            add_noise=add_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
            fpu=fpu,
            rng=rng,
        )

        # Advance each game by one sampled move; store the full pi.
        for game, pi in zip(slots, pis):
            temp = temperature_for_ply(len(game), temp_cutoff_ply, hot_temp, cold_temp)
            idx = sample_move_index(pi, temperature=temp, rng=rng)
            game.record_and_push(idx, pi)

        # Harvest finished games and refill their slots from the fresh-game queue.
        next_slots: List[SelfPlayGame] = []
        for game in slots:
            if game.is_over():
                result = GameResult(game.finalize(), len(game), game.outcome_winner())
                results.append(result)
                if on_game is not None:
                    on_game(result)
                if bar is not None:
                    bar.update(1)
            else:
                next_slots.append(game)

        while started < num_games and len(next_slots) < games_in_flight:
            next_slots.append(_new_game())
            started += 1

        slots = next_slots

    if bar is not None:
        bar.close()
    return results