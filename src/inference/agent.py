"""The submission entrypoint. The platform imports this file and calls get_move.

This is the glue between the AI Chessathon harness and our model. All the real
work lives in inference.py; this file just loads the model ONCE at import (inside
the init budget, off your move clock) and forwards each request to it.

Drop at the fork root alongside moves.py, encoding.py, network.py, mcts.py and
inference.py, with the trained checkpoint at weights/model.pt. `make zip` then
packages exactly those.
"""

import chess

from inference.inference_file import ChessInference

# Import-time init: build the net, load weights, warm up the first forward pass —
# all once per game, before the clock starts. Keep the instance on a module global
# so it survives across the game's moves (the process stays alive between calls).
_MODEL = ChessInference(
    "weights/model.pt",
    device="cpu",              # the platform runs cpu torch
    num_simulations=100,       # main strength/latency knob — tune on the ladder
    c_puct=2.0,
    use_mcts=True,
)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    try:
        return _MODEL.best_move(fen, time_left_ms)
    except Exception as exc:
        # A crash or illegal string is an instant loss; degrade to a legal move
        # instead. print() is safe and shows up in your validation log.
        print(f"[agent] fell back after error: {exc}")
        board = chess.Board(fen)
        return next(iter(board.legal_moves)).uci()