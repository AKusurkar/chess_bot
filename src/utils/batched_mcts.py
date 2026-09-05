"""
Stage 6 (core) — batched, synchronous MCTS: the GPU-saturation lift.

The plan is explicit that the throughput win does NOT come from inside one search
(50 sequential sims barely batch — each depends on the last). It comes from
running MANY self-play games at once and folding all of their pending leaf
evaluations into ONE batched forward pass. This module is that mechanism.

WHAT "SYNCHRONOUS" MEANS HERE
-----------------------------
We keep G independent search trees (one per in-flight game) and advance them in
lockstep, one simulation-round at a time:

    for each simulation:
        for each game: descend its OWN tree by PUCT to a leaf   (pure Python)
        collect every non-terminal leaf across all G games
        evaluate them in a SINGLE batched net forward pass       (the GPU win)
        route (value, priors) back, expand + backup each tree

Each tree is still perfectly sequential within itself — one leaf per game per
round — so there is NO need for virtual loss. Virtual loss is only required when
you parallelize *within* a single search (many threads down one tree); we do not.
That keeps this faithful to plain PUCT while still batching the expensive part.

RELATION TO mcts.py
-------------------
This reuses every primitive from the single-position search unchanged — Node,
select_child, expand, add_dirichlet_noise, backup, visit_distribution,
terminal_value — so the tree semantics, the per-ply sign flip, the frame
conventions, and the tau=1 output are IDENTICAL to Stage 3. The only new thing is
the scheduling: which leaves get evaluated together. As a correctness contract,
`run_batched_mcts([board], eval_batch, ...)` with a deterministic evaluator and a
fixed rng produces the same pi as `mcts.run_mcts(board, ...)`.

FRAME / BATCHNORM
-----------------
Identical discipline to mcts.make_net_evaluator: encode() canonicalizes to the
mover, move_to_index flips for Black, and the batched evaluator runs the net in
eval() mode under no_grad so BatchNorm uses running stats (never per-batch stats).
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import chess

from utils.utils_file import move_to_index, encode
from utils.mcts import (
    Node,
    select_child,
    expand,
    add_dirichlet_noise,
    backup,
    visit_distribution,
    terminal_value,
)

# A batched evaluator maps a list of boards -> a list of (value, priors), aligned
# by index. Each element is exactly what the single-board Evaluator returns:
#   value  : float in [-1, 1] from that board's side-to-move perspective.
#   priors : {canonical_move_index -> (chess.Move, prior_prob)} over legal moves.
BatchEvaluator = Callable[[List[chess.Board]], List[Tuple[float, Dict[int, Tuple[chess.Move, float]]]]]


# --------------------------------------------------------------------------- #
# Batched network evaluator (Stage 2 masking done per-board, inline)
# --------------------------------------------------------------------------- #
def make_batched_net_evaluator(net, max_batch: Optional[int] = None) -> BatchEvaluator:
    """Wrap a MarchHare into a BatchEvaluator: stack -> one forward -> per-board
    legal-mask -> softmax. This is the piece that turns G concurrent leaves into a
    single GPU call.

    `max_batch` optionally caps the forward-pass size (peak activation memory
    scales with it); larger batches than the cap are split and concatenated. Torch
    is imported lazily so the search core stays importable without torch.

    BatchNorm discipline: eval() + no_grad => running stats, never batch stats —
    which matters MORE here, because a mis-set mode would let the composition of a
    batch silently change each leaf's evaluation.
    """
    import torch

    device = next(net.parameters()).device
    net.eval()

    @torch.no_grad()
    def _forward(boards: List[chess.Board]):
        xs = np.stack([encode(b) for b in boards]).astype(np.float32)   # (B,20,8,8)
        x = torch.from_numpy(xs).to(device)
        logits, values = net(x)                                         # (B,1880),(B,1)
        # One host copy for the whole batch; the per-board mask/softmax below is
        # cheap on CPU and avoids a GPU sync per board.
        return logits.float().cpu(), values.reshape(-1).float().cpu()

    @torch.no_grad()
    def evaluate_batch(boards: List[chess.Board]):
        if not boards:
            return []

        if max_batch is None or len(boards) <= max_batch:
            logits, values = _forward(boards)
        else:
            chunks_l, chunks_v = [], []
            for s in range(0, len(boards), max_batch):
                lg, vl = _forward(boards[s:s + max_batch])
                chunks_l.append(lg)
                chunks_v.append(vl)
            logits = torch.cat(chunks_l, dim=0)
            values = torch.cat(chunks_v, dim=0)

        out = []
        for k, board in enumerate(boards):
            lg = logits[k]
            legal = list(board.legal_moves)
            idxs = [move_to_index(board, m) for m in legal]

            # Additive -inf mask BEFORE softmax (never "softmax then zero out").
            mask = torch.full_like(lg, float("-inf"))
            mask[idxs] = 0.0
            probs = torch.softmax(lg + mask, dim=0)

            priors = {idx: (mv, float(probs[idx])) for idx, mv in zip(idxs, legal)}
            out.append((float(values[k]), priors))
        return out

    return evaluate_batch


# --------------------------------------------------------------------------- #
# The batched search
# --------------------------------------------------------------------------- #
def run_batched_mcts(boards: List[chess.Board],
                     evaluate_batch: BatchEvaluator,
                     *,
                     num_simulations: int = 64,
                     c_puct: float = 2.0,
                     add_noise: bool = True,
                     dirichlet_alpha: float = 0.3,
                     dirichlet_frac: float = 0.25,
                     fpu: float = 0.0,
                     rng: Optional[np.random.Generator] = None,
                     return_roots: bool = False):
    """Run PUCT MCTS on a *batch* of positions concurrently and return one visit
    distribution per input board.

    Each `boards[i]` should be a detached copy (game.board_copy()); trees never
    mutate the inputs — descents push moves onto per-node copies, exactly as the
    single-position search does. Terminal inputs are handled gracefully: they are
    not searched and receive an all-zero pi (the coordinator never feeds terminal
    boards, but defending here keeps callers simple).

    Returns
        pis : list of np.ndarray (NUM_MOVES,) float32, aligned with `boards`. Each
              is the RAW tau=1 visit distribution (temperature is applied later, at
              move-selection time, exactly once — same contract as run_mcts).
        If return_roots=True, returns (pis, roots) so callers/tests can inspect.
    """
    rng = rng or np.random.default_rng()
    G = len(boards)

    roots: List[Node] = [Node() for _ in range(G)]
    for r, b in zip(roots, boards):
        r.board = b

    # Only non-terminal positions get a tree; terminal ones fall through to a
    # zero pi via visit_distribution on an unexpanded root.
    active = [i for i in range(G) if not roots[i].board.is_game_over(claim_draw=True)]

    # -- Roots: one batched eval, expand, then (self-play only) inject noise -----
    if active:
        root_results = evaluate_batch([roots[i].board for i in active])
        for i, (_v, priors) in zip(active, root_results):
            expand(roots[i], priors)          # root value is unused; only priors matter
        if add_noise:
            # Per-root noise, in a fixed order so a seed reproduces the batch.
            for i in active:
                add_dirichlet_noise(roots[i], dirichlet_alpha, dirichlet_frac, rng)

    # -- Simulations: descend every tree, then evaluate all leaves as one batch --
    for _ in range(num_simulations):
        eval_nodes: List[Node] = []
        eval_paths: List[List[Node]] = []
        eval_boards: List[chess.Board] = []

        for i in active:
            node = roots[i]
            search_path = [node]

            # SELECT: descend by PUCT to an unexpanded node (a leaf).
            while node.expanded():
                idx, child = select_child(node, c_puct, fpu)
                if child.board is None:               # materialize lazily...
                    child.board = node.board.copy()   # ...keeping history so in-search
                    child.board.push(child.move)      # repetition/50-move works
                node = child
                search_path.append(node)

            if node.board.is_game_over(claim_draw=True):
                # Terminal leaf: true outcome, backed up immediately (no net call).
                backup(search_path, terminal_value(node.board))
            else:
                # Defer to the shared batched forward pass below.
                eval_nodes.append(node)
                eval_paths.append(search_path)
                eval_boards.append(node.board)

        # EXPAND + EVALUATE + BACKUP the batch of leaves in one shot.
        if eval_boards:
            results = evaluate_batch(eval_boards)
            for node, search_path, (value, priors) in zip(eval_nodes, eval_paths, results):
                expand(node, priors)
                backup(search_path, value)

    pis = [visit_distribution(roots[i]) for i in range(G)]
    return (pis, roots) if return_roots else pis