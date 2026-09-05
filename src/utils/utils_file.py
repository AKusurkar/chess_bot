"""
Stage 0 — the move <-> index bridge (the "pilot light").

Everything here is derived from ONE frozen list so the forward and backward
maps can never drift apart. If you touch this file, re-run tests_stage0.py.

--- Why the list is 1880, not 1858 ---------------------------------------

The plan calls for "1858 possible moves". 1858 is the Leela/AlphaZero number,
and it is only 1858 because those engines let a queen-PROMOTION share the same
policy index as the geometric queen-SLIDE to that square (e.g. e7e8=Q and a
queen sliding e7->e8 are the same index). Reconstructing the actual move from
that shared index then REQUIRES the board (was the mover a pawn on the 7th?).

That directly contradicts the plan's own Stage-0 tests, which want a *pure,
board-independent* bijection  UCI string  <->  index  where the four
promotions e7e8q/r/b/n are four DISTINCT indices. You cannot have both.

Resolution that honors the plan's intent (a bulletproof, context-free bridge):
give every UCI string python-chess can ever emit its own index. That is:

    queen-style slides ......... 1456   (rook 896 + bishop 560)
    knight moves ...............  336
    promotions (22 geom x 4) ...   88   (q,r,b,n all suffixed explicitly)
    -------------------------------------
    total ...................... 1880

The extra 22 over 1858 are exactly the queen-promotions we refuse to fold into
the slide index. The cost is 22 unused policy logits — a rounding error — in
exchange for MOVE_TO_IDX[m.uci()] working for every legal move with zero
special-casing. Worth it.

--- Orientation ----------------------------------------------------------

The list only contains promotions on rank7->rank8 (upward). That is correct
*because* encoding.py canonicalizes every position to the side to move (mover
always at the bottom, moving up). A real black promotion (e.g. e2e1q) is only
ever indexed after flipping the board, at which point it becomes e7e8q. Use
move_to_index / index_to_move below — never MOVE_TO_IDX directly on a raw
black-to-move move.
"""

from typing import Dict, List
import chess
import numpy as np

# 8 queen directions (file delta, rank delta) and 8 knight deltas.
_QUEEN_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_PROMO_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def _generate_move_list() -> List[str]:
    moves = set()

    for from_sq in chess.SQUARES:
        ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)

        # Queen-style rays: rook + bishop lines, up to 7 squares, stop at edge.
        for df, dr in _QUEEN_DIRS:
            for dist in range(1, 8):
                tf, tr = ff + df * dist, fr + dr * dist
                if 0 <= tf < 8 and 0 <= tr < 8:
                    moves.add(chess.Move(from_sq, chess.square(tf, tr)).uci())
                else:
                    break  # ran off the board; no point going further this ray

        # Knight moves.
        for df, dr in _KNIGHT_DELTAS:
            tf, tr = ff + df, fr + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                moves.add(chess.Move(from_sq, chess.square(tf, tr)).uci())

    # Promotions: from rank 7 (index 6) to rank 8 (index 7), straight or a
    # diagonal capture, each with all four promotion pieces suffixed.
    for ff in range(8):
        from_sq = chess.square(ff, 6)
        for df in (-1, 0, 1):
            tf = ff + df
            if 0 <= tf < 8:
                to_sq = chess.square(tf, 7)
                for pp in _PROMO_PIECES:
                    moves.add(chess.Move(from_sq, to_sq, promotion=pp).uci())

    # sorted() makes the index assignment reproducible across runs/machines;
    # a set's iteration order is not guaranteed and would silently reshuffle
    # every index between runs.
    return sorted(moves)


# The one frozen list. Both maps derive from it, so they cannot disagree.
CANONICAL_MOVES: List[str] = _generate_move_list()
NUM_MOVES: int = len(CANONICAL_MOVES)                       # 1880

MOVE_TO_IDX: Dict[str, int] = {uci: i for i, uci in enumerate(CANONICAL_MOVES)}
IDX_TO_MOVE: List[str] = CANONICAL_MOVES                    # index -> uci (identity list)


def flip_move(move: chess.Move) -> chess.Move:
    """Vertically mirror a move (square ^ 56). Promotion piece is unchanged:
    a promotion stays a promotion under a vertical flip."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Canonical policy index for `move` in `board`. Flips into the mover's
    frame when it is Black to move, matching encoding.py's canonicalization."""
    if board.turn == chess.BLACK:
        move = flip_move(move)
    return MOVE_TO_IDX[move.uci()]


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    """Inverse of move_to_index: policy index -> the real move on `board`
    (flipped back out of the canonical frame when it is Black to move)."""
    move = chess.Move.from_uci(IDX_TO_MOVE[index])
    if board.turn == chess.BLACK:
        move = flip_move(move)
    return move

"""
Stage 0 — board -> tensor encoding, canonicalized to the side to move.

Every position is flipped so the mover is at the bottom moving "up" the board
(board.mirror() when it is Black to move). The network therefore only ever
sees positions from one perspective, so it learns each pattern once instead of
twice. This MUST stay consistent with moves.py: moves are indexed in the same
flipped frame (see moves.move_to_index).

Plane layout (20 planes of 8x8), "mover" = side to move (white-like after flip):

    0- 5  mover pieces      P N B R Q K
    6-11  opponent pieces   P N B R Q K
   12     mover   kingside  castling right   (broadcast: whole plane 0/1)
   13     mover   queenside castling right
   14     opponent kingside castling right
   15     opponent queenside castling right
   16     side-to-move: 1 if the mover is *really* White, else 0
   17     en passant target square (only if the capture is actually legal)
   18     halfmove clock  (fifty-move counter), broadcast, /100
   19     fullmove number, broadcast, /100  (crude cap; a tunable, not sacred)

Notes / deliberate simplifications:
  * No position history. The encoding is non-Markovian w.r.t. threefold
    repetition — the net literally cannot see a repetition coming, though
    python-chess still enforces the draw. A single repetition-count plane is
    the cheap partial fix if repetition-blindness shows up later.
  * Plane 16 is near-vestigial once the board is fully canonicalized (chess has
    no color-dependent rules after flipping). AlphaZero keeps a colour feature,
    so it is kept here; dropping it is defensible.
  * The two counter planes are broadcast scalars; the /100 normalization is
    arbitrary and just keeps values ~O(1). Adjust freely.
"""

NUM_PLANES = 20
_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode(board: chess.Board) -> np.ndarray:
    """Return a (20, 8, 8) float32 tensor for `board`, from the mover's view."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    white_to_move = board.turn == chess.WHITE
    # Canonical board: mover is always White-at-the-bottom. mirror() swaps
    # colors, flips vertically, swaps turn, and swaps castling rights + ep.
    cb = board if white_to_move else board.mirror()

    # Piece planes. On cb, the mover is always White, opponent always Black.
    for i, pt in enumerate(_PIECE_ORDER):
        for sq in cb.pieces(pt, chess.WHITE):
            planes[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
        for sq in cb.pieces(pt, chess.BLACK):
            planes[6 + i, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    # Castling rights (mover = White on cb).
    if cb.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if cb.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if cb.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if cb.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0

    # Real color of the mover (computed from the ORIGINAL board).
    if white_to_move:
        planes[16, :, :] = 1.0

    # En passant: mark the target ONLY when the capture is genuinely available.
    # board.ep_square is set on every pawn double-step even when nobody can take
    # it; has_legal_en_passant() filters those phantom targets out.
    if cb.has_legal_en_passant():
        ep = cb.ep_square
        planes[17, chess.square_rank(ep), chess.square_file(ep)] = 1.0

    # Counters (clocks are invariant under mirror(), so read the original).
    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    planes[19, :, :] = min(board.fullmove_number, 100) / 100.0

    return planes