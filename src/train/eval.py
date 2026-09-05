"""
Stage 6 (evaluation) — the arena: honest strength measurement, in process.

The plan's single most important test is the UPWARD EVAL CURVE: win-rate against a
FIXED checkpoint must tick up, or nothing is learning. This module produces that
number, plus one game against each competition baseline (random/greedy/minimax/
numba) as an external sanity anchor.

WHY IN-PROCESS (and not harness/referee.py)
-------------------------------------------
The competition harness (harness/play.py) runs each side as its own subprocess
through a JSON protocol with a real clock — correct for an HONEST pre-upload check,
but it reloads the model from disk every game and pays subprocess + protocol
overhead. For a per-epoch monitor we call this dozens of times, so we run games in
THIS process against the live net (no reload) and hand each agent a fixed, generous
per-move time budget instead of a wall clock. Same legality/draw/ply-cap rules as
the referee; we are measuring relative strength, not latency, so the wall clock is
deliberately out of the loop. Use `make zip` + `harness/play.py` for the honest,
clock-enforced check before an actual upload.

THE DIVERSITY TRAP (plan, Stage 6)
----------------------------------
Two deterministic nets at tau->0 with no Dirichlet noise play the IDENTICAL game
every time from the start position, so "win rate over N games" silently collapses
to one game repeated N. We break that with a few RANDOM OPENING PLIES per game
(same opening handed to both sides) and by alternating colours. Without this the
one metric the whole project leans on is meaningless.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import chess

from utils.utils_file import index_to_move
from utils.mcts import run_mcts, Evaluator

# An agent is just: (fen, time_left_ms) -> uci string. This is EXACTLY the
# competition's get_move signature, so the baselines drop in unmodified and our
# own submission agent could be scored the same way.
AgentMove = Callable[[str, int], str]

RESULT_WHITE, RESULT_BLACK, RESULT_DRAW = "white", "black", "draw"


# --------------------------------------------------------------------------- #
# Our net, wrapped as an agent (torch-free: torch lives behind evaluate_fn)
# --------------------------------------------------------------------------- #
def _select_move(board: chess.Board,
                 evaluate_fn: Evaluator,
                 *,
                 use_mcts: bool,
                 num_simulations: int,
                 c_puct: float,
                 fpu: float,
                 rng: Optional[np.random.Generator]) -> chess.Move:
    """Greedy move selection — the same rule inference.select_move uses, inlined
    here so the arena core needs no torch (torch is injected via evaluate_fn).

    MCTS with noise OFF (honest opinion), take the most-visited move, then un-flip
    the canonical index back onto the real board via index_to_move.
    """
    if not use_mcts:
        _v, priors = evaluate_fn(board)
        best_idx = max(priors, key=lambda i: priors[i][1])
        return priors[best_idx][0]

    pi = run_mcts(board, evaluate_fn=evaluate_fn, num_simulations=num_simulations,
                  c_puct=c_puct, add_noise=False, fpu=fpu, rng=rng)
    return index_to_move(board, int(pi.argmax()))


def make_net_agent(evaluate_fn: Evaluator,
                   *,
                   use_mcts: bool = True,
                   num_simulations: int = 100,
                   c_puct: float = 2.0,
                   fpu: float = 0.0,
                   rng: Optional[np.random.Generator] = None) -> AgentMove:
    """Turn a single-board Evaluator (mcts.make_net_evaluator) into an AgentMove.

    NOTE on BatchNorm: the caller must ensure the net is in eval() mode before the
    arena runs (self-play/training toggle the mode). The evaluator reads the net
    live, so it always reflects current weights.
    """
    def move(fen: str, time_left_ms: int) -> str:
        board = chess.Board(fen)
        chosen = _select_move(board, evaluate_fn, use_mcts=use_mcts,
                              num_simulations=num_simulations, c_puct=c_puct,
                              fpu=fpu, rng=rng)
        return chosen.uci()
    return move


# --------------------------------------------------------------------------- #
# Competition baselines, loaded straight from the forked repo under /external
# --------------------------------------------------------------------------- #
def load_baseline_agents(baselines_dir: str,
                         names: Optional[List[str]] = None) -> Dict[str, AgentMove]:
    """Import each `baselines/<name>/agent.py` from the competition repo and return
    {name -> get_move}. Any baseline that fails to import (e.g. `numba` when numba
    is not installed) is skipped with a printed note rather than sinking the run —
    the eval is a monitor, not a gate.
    """
    import importlib.util

    base = Path(baselines_dir)
    if names is None:
        names = sorted(p.name for p in base.iterdir()
                       if p.is_dir() and (p / "agent.py").exists())

    agents: Dict[str, AgentMove] = {}
    for name in names:
        agent_path = base / name / "agent.py"
        if not agent_path.exists():
            print(f"[arena] baseline {name!r} has no agent.py at {agent_path}; skipping")
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"baseline_{name}", agent_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)          # runs import-time warm-ups (e.g. numba jit)
            agents[name] = module.get_move
        except Exception as exc:                      # noqa: BLE001 - never let one baseline sink eval
            print(f"[arena] baseline {name!r} failed to load ({exc}); skipping")
    return agents


# --------------------------------------------------------------------------- #
# One game between two agents
# --------------------------------------------------------------------------- #
@dataclass
class GameOutcome:
    result: str            # "white" | "black" | "draw"
    termination: str       # checkmate / stalemate / ... / ply_cap / illegal / crash
    num_plies: int
    pgn: Optional[str] = None


def play_arena_game(white: AgentMove,
                    black: AgentMove,
                    *,
                    start_fen: str = chess.STARTING_FEN,
                    max_plies: int = 600,
                    move_time_ms: int = 120_000,
                    capture_pgn: bool = False) -> GameOutcome:
    """Play one game to completion. Same rules as harness/referee.py minus the wall
    clock: natural terminations and the ply cap end the game; an illegal move or a
    raised exception hands the win to the other side (mirroring the referee's
    'illegal'/'crash')."""
    board = chess.Board(start_fen)
    agents = {chess.WHITE: white, chess.BLACK: black}

    while True:
        finish = board.outcome(claim_draw=True)
        if finish is not None:
            result = RESULT_DRAW if finish.winner is None else (
                RESULT_WHITE if finish.winner == chess.WHITE else RESULT_BLACK)
            return _outcome(board, result, finish.termination.name.lower(), capture_pgn)
        if board.ply() >= max_plies:
            return _outcome(board, RESULT_DRAW, "ply_cap", capture_pgn)

        mover = board.turn
        loser_side = RESULT_WHITE if mover == chess.WHITE else RESULT_BLACK
        winner_side = RESULT_BLACK if mover == chess.WHITE else RESULT_WHITE
        try:
            uci = agents[mover](board.fen(), move_time_ms)
        except Exception as exc:                          # noqa: BLE001
            print(f"[arena] {loser_side} agent crashed: {exc}")
            return _outcome(board, winner_side, "crash", capture_pgn)

        move = _legal_move(board, uci)
        if move is None:
            return _outcome(board, winner_side, "illegal", capture_pgn)
        board.push(move)


def _legal_move(board: chess.Board, uci: str) -> Optional[chess.Move]:
    try:
        move = chess.Move.from_uci(uci)
    except (chess.InvalidMoveError, ValueError):
        return None
    return move if move in board.legal_moves else None


def _outcome(board: chess.Board, result: str, termination: str, capture_pgn: bool) -> GameOutcome:
    pgn = None
    if capture_pgn:
        import chess.pgn
        game = chess.pgn.Game.from_board(board)
        game.headers["Result"] = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}[result]
        game.headers["Termination"] = termination
        pgn = str(game)
    return GameOutcome(result=result, termination=termination,
                       num_plies=board.ply(), pgn=pgn)


# --------------------------------------------------------------------------- #
# Opening diversity
# --------------------------------------------------------------------------- #
def random_opening_fen(rng: np.random.Generator,
                       plies: int,
                       base_fen: str = chess.STARTING_FEN) -> str:
    """Play `plies` uniformly-random legal moves from `base_fen` and return the FEN.
    Backs off if a random line ends the game early (so the returned position always
    has legal moves). plies=0 returns base_fen unchanged."""
    for _ in range(8):                       # a few attempts to avoid an early mate
        board = chess.Board(base_fen)
        ok = True
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                ok = False
                break
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        if ok and not board.is_game_over(claim_draw=True):
            return board.fen()
    return base_fen


# --------------------------------------------------------------------------- #
# Matches (scored from the NET's perspective)
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    games: int
    wins: int
    draws: int
    losses: int
    terminations: Dict[str, int]

    @property
    def score(self) -> float:
        """Standard chess score in [0,1]: win=1, draw=0.5, loss=0."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        p = f"{prefix}/" if prefix else ""
        return {f"{p}score": self.score, f"{p}win_rate": self.win_rate,
                f"{p}wins": self.wins, f"{p}draws": self.draws, f"{p}losses": self.losses}


def play_match(net_agent: AgentMove,
               opponent: AgentMove,
               *,
               games: int,
               rng: np.random.Generator,
               opening_plies: int = 4,
               max_plies: int = 600,
               move_time_ms: int = 120_000,
               progress: bool = False,
               desc: str = "eval") -> MatchResult:
    """Play `games` games, alternating the net's colour and using a fresh random
    opening per game (the same opening for both sides). Result is scored from the
    net's perspective. This is the routine behind both the fixed-checkpoint curve
    and the baseline anchors."""
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=games, desc=desc, unit="game", leave=False)

    wins = draws = losses = 0
    terminations: Dict[str, int] = {}
    for g in range(games):
        net_is_white = (g % 2 == 0)
        start_fen = random_opening_fen(rng, opening_plies)
        white, black = (net_agent, opponent) if net_is_white else (opponent, net_agent)
        outcome = play_arena_game(white, black, start_fen=start_fen,
                                  max_plies=max_plies, move_time_ms=move_time_ms)

        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        if outcome.result == RESULT_DRAW:
            draws += 1
        elif (outcome.result == RESULT_WHITE) == net_is_white:
            wins += 1
        else:
            losses += 1
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return MatchResult(games, wins, draws, losses, terminations)


def play_vs_baselines(net_agent: AgentMove,
                      baseline_agents: Dict[str, AgentMove],
                      *,
                      rng: np.random.Generator,
                      games_per_baseline: int = 1,
                      opening_plies: int = 0,
                      max_plies: int = 600,
                      move_time_ms: int = 120_000,
                      progress: bool = False) -> Dict[str, MatchResult]:
    """One (or a few) games against each competition baseline. Defaults to a single
    clean game from the start position per baseline (opening_plies=0) — the plan's
    'just one game is fine' anchor — but honours games_per_baseline / opening_plies
    if you want more. Returns {baseline_name -> MatchResult}."""
    results: Dict[str, MatchResult] = {}
    items = list(baseline_agents.items())
    iterator = items
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(items, desc="baselines", unit="opp", leave=False)
    for name, opp in iterator:
        results[name] = play_match(net_agent, opp, games=games_per_baseline, rng=rng,
                                   opening_plies=opening_plies, max_plies=max_plies,
                                   move_time_ms=move_time_ms)
    return results