"""
Stage 0/4 bridge — a stateful self-play game driver.

This is a convenience layer over the Stage-0 primitives (moves.py, encoding.py).
It owns ONE ongoing game, hands you a ready-to-use model input at each step,
accepts an externally-chosen move to advance, records the training signal per
move, and finalizes the trajectory into replay-buffer examples once the outcome
is known.

WHERE THIS SITS IN THE ALPHAZERO LOOP
-------------------------------------
The move that gets *played* is NOT the network's argmax. It is sampled from the
MCTS visit distribution pi. This class is deliberately agnostic about how the
move was chosen — it just applies the index you give it. The canonical loop:

    game = SelfPlayGame.new()
    while not game.is_over():
        x    = game.current_input()          # (20,8,8) model input, mover's view
        mask = game.legal_mask()             # (1880,) bool, True = legal
        pi   = run_mcts(game.board_copy(), net, sims=64)   # <- MCTS is the mover
        idx  = sample_move_index(pi, temperature)          # canonical move index
        game.record_and_push(idx, pi)        # store (x, pi, mover), then advance
    examples = game.finalize()               # [(x, pi, z), ...] for the buffer
    buffer.extend(examples)

Note what this class does NOT do: it does not run MCTS and does not touch the
network. MCTS explores hypothetical positions on its own board copies (via
board_copy() and the free functions encode()/move_to_index()); it never calls
record_and_push. Keeping that boundary is what stops the PPO-shaped mistake of
"play the policy's move directly".

FRAME CONSISTENCY
-----------------
Everything is stored in the canonical (mover-at-bottom) frame:
  * current_input() is encoding.encode(board) — already canonicalized.
  * legal_move_indices()/legal_mask() use moves.move_to_index (flips for Black).
  * The move index you pass to record_and_push is a canonical index; it is
    mapped back to the real move via moves.index_to_move.
So the stored (state, pi) pair and the played move all live in one frame, and z
is labeled from each position's own mover perspective. No frame ever leaks.
"""

from typing import List, Optional, Tuple
import numpy as np
import chess

from utils.utils_file import move_to_index, index_to_move, NUM_MOVES, encode, NUM_PLANES

TrainingExample = Tuple[np.ndarray, np.ndarray, float]  # (state (20,8,8), pi (1880,), z)


def sample_move_index(pi: np.ndarray, temperature: float = 1.0,
                      rng: Optional[np.random.Generator] = None) -> int:
    """Pick a canonical move index from a visit distribution pi.

    temperature ~1.0 early (exploration), -> 0 later (greedy). This is the
    self-play move-selection rule; MCTS produces pi, this picks from it. pi is
    assumed to have support only on legal moves (that is how MCTS builds it).
    """
    pi = np.asarray(pi, dtype=np.float64)
    if temperature <= 1e-6:
        return int(pi.argmax())               # tau -> 0 : greedy
    scaled = np.zeros_like(pi)
    nz = pi > 0
    scaled[nz] = pi[nz] ** (1.0 / temperature)
    total = scaled.sum()
    if total <= 0:                            # degenerate guard
        return int(pi.argmax())
    scaled /= total
    rng = rng or np.random.default_rng()
    return int(rng.choice(pi.shape[0], p=scaled))


class SelfPlayGame:
    """One ongoing game plus the per-move training record."""

    def __init__(self, board: Optional[chess.Board] = None, max_plies: int = 512):
        # max_plies is the Stage-4 hard ply cap: with a near-random early net,
        # games otherwise wander until the 50-move/repetition rule fires,
        # cratering throughput. Adjudicated as a draw when hit.
        self.board = board.copy() if board is not None else chess.Board()
        self.max_plies = max_plies
        self._plies = 0
        self._history: List[Tuple[np.ndarray, np.ndarray, bool]] = []  # (x, pi, mover)
        self._input_cache: Optional[np.ndarray] = None

    # -- construction ------------------------------------------------------
    @classmethod
    def new(cls, fen: Optional[str] = None, max_plies: int = 512) -> "SelfPlayGame":
        """Fresh game. Pass a FEN to start from an opening position — useful
        later for forcing diversity in eval matches (Stage 6)."""
        return cls(chess.Board(fen) if fen else chess.Board(), max_plies=max_plies)

    # -- model input side --------------------------------------------------
    def current_input(self) -> np.ndarray:
        """(20,8,8) canonical tensor for the side to move. Cached until push."""
        if self._input_cache is None:
            self._input_cache = encode(self.board)
        return self._input_cache

    def legal_move_indices(self) -> List[int]:
        """Canonical policy indices of the legal moves in this position."""
        return [move_to_index(self.board, m) for m in self.board.legal_moves]

    def legal_mask(self) -> np.ndarray:
        """(1880,) boolean mask, True on legal moves. Stage 2 turns this into an
        additive -inf mask before softmax: logits.masked_fill(~mask, -inf)."""
        mask = np.zeros(NUM_MOVES, dtype=bool)
        mask[self.legal_move_indices()] = True
        return mask

    # -- advancing the game ------------------------------------------------
    def push_index(self, move_index: int) -> Optional[np.ndarray]:
        """Apply a move (chosen externally) without recording it. For eval play
        or scripted opening plies. Returns the next input, or None if over."""
        move = index_to_move(self.board, move_index)
        assert self.board.is_legal(move), f"illegal move {move.uci()} for index {move_index}"
        self.board.push(move)
        self._plies += 1
        self._input_cache = None
        return None if self.is_over() else self.current_input()

    def record_and_push(self, move_index: int, pi: np.ndarray) -> Optional[np.ndarray]:
        """Store (current state, pi, mover) as a pending training example, then
        play move_index. `pi` must be the full canonical visit distribution
        (store it in full even though temperature may pick a different move).
        Returns the next model input, or None if the game is now over."""
        pi = np.asarray(pi, dtype=np.float32)
        assert pi.shape == (NUM_MOVES,), f"pi must be ({NUM_MOVES},), got {pi.shape}"
        self._history.append((self.current_input(), pi, self.board.turn))
        return self.push_index(move_index)

    # -- terminal / outcome ------------------------------------------------
    def is_over(self) -> bool:
        return self._plies >= self.max_plies or self.board.is_game_over(claim_draw=True)

    def outcome_winner(self) -> Optional[bool]:
        """chess.WHITE / chess.BLACK / None. A ply-capped game is a draw (None)."""
        if self.board.is_game_over(claim_draw=True):
            return self.board.outcome(claim_draw=True).winner
        return None  # ply cap (or not over — caller should gate on is_over)

    # -- finalize into training data --------------------------------------
    def finalize(self) -> List[TrainingExample]:
        """Attach z to every recorded position from that position's mover view,
        and return [(state, pi, z)]. Call once the game is over."""
        assert self.is_over(), "finalize() before the game ended"
        winner = self.outcome_winner()
        examples: List[TrainingExample] = []
        for state, pi, mover in self._history:
            if winner is None:
                z = 0.0
            else:
                z = 1.0 if winner == mover else -1.0
            examples.append((state, pi, np.float32(z)))
        return examples

    # -- MCTS convenience --------------------------------------------------
    def board_copy(self) -> chess.Board:
        """A detached copy for MCTS to explore. (copy(stack=False) is faster but
        drops the history MCTS would need for repetition detection.)"""
        return self.board.copy()

    def __len__(self) -> int:
        return self._plies