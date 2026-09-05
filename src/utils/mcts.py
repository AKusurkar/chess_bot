"""
Stage 3 — Monte Carlo Tree Search (PUCT): the policy-improvement operator.

This is the ONLY algorithmically-changed piece relative to full AlphaZero: 50-100
simulations per move instead of 800. Everything else stays faithful. MCTS turns
the network's raw prior p and value v into a *search-improved* visit distribution
pi. That pi is (a) sampled to choose the move actually played
(game.sample_move_index) and (b) the training target for the policy head (Stage 5).
The gap between pi and the raw policy head p is the improvement operator — it is
what makes this AlphaZero and not PPO-reading-the-policy-head.

FRAME / PERSPECTIVE CONVENTIONS  (must match Stage 0/1 exactly)
---------------------------------------------------------------
  * States are encoded from the side-to-move's view: encoding.encode() flips the
    board for Black. The policy is over the 1880 canonical indices in that flipped
    frame; moves.move_to_index / index_to_move do the flipping. This module only
    ever touches canonical indices via move_to_index, so it inherits that frame
    for free and never has to reason about colour.
  * The network's value v is ALWAYS from the perspective of the side to move at
    the evaluated node. Every node here stores value_sum in its OWN mover frame,
    which is exactly why backup() flips sign every ply and why a child's mean
    value is negated when scored from its parent (puct_score).
  * The returned pi is a (NUM_MOVES,) vector over canonical indices with support
    only on the legal moves — precisely what game.record_and_push /
    sample_move_index consume. It is the RAW visit distribution (tau = 1);
    temperature is applied downstream at sampling time, NOT here, so temperature
    is never applied twice.

WHAT THIS MODULE DELIBERATELY DOES NOT DO (yet)
-----------------------------------------------
Single-position search only. The GPU-saturation work (many concurrent games
sharing batched leaf evaluations + virtual loss) is a separate Stage-6 lift. It
slots in by swapping the per-leaf `evaluate_fn` for a batched one — which is the
reason evaluation is *injected* here rather than hard-wired into the search.
"""

from typing import Callable, Dict, List, Optional, Tuple
import math

import numpy as np
import chess

from utils.utils_file import move_to_index, NUM_MOVES, encode

# An Evaluator maps a board -> (value, priors):
#   value  : float in [-1, 1], from the board's side-to-move perspective.
#   priors : {canonical_move_index -> (chess.Move, prior_prob)} over the legal moves.
Evaluator = Callable[[chess.Board], Tuple[float, Dict[int, Tuple[chess.Move, float]]]]


# --------------------------------------------------------------------------- #
# Tree node
# --------------------------------------------------------------------------- #
class Node:
    """One state in the search tree.

    Stats live *at the node* (MuZero-style) rather than on parent edges:
      prior       P(parent -> this): the network prior for reaching this node.
      visit_count N: number of simulations that have passed through this node.
      value_sum   W: sum of backed-up values, in THIS node's mover frame.
      children    {canonical_move_index -> Node}, created on expansion.
      move        the chess.Move leading INTO this node (None at the root).
      board       the position at this node, materialized lazily on first visit.
    Mean value Q = value_sum / visit_count is from this node's mover's view.
    """

    __slots__ = ("prior", "move", "visit_count", "value_sum", "children", "board")

    def __init__(self, prior: float = 0.0, move: Optional[chess.Move] = None):
        self.prior = prior
        self.move = move
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}
        self.board: Optional[chess.Board] = None

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        """Mean value from THIS node's side-to-move perspective (0 if unvisited)."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def puct_score(parent: Node, child: Node, c_puct: float, fpu: float) -> float:
    """PUCT for selecting `child` from `parent`:

        Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

    Q is NEGATED: child.value() is stored in the child's mover frame, which is the
    opponent of the parent's mover, so a high value for the child is a low value
    for us. Unvisited children take the first-play-urgency constant `fpu` in place
    of -child.value() (which would just be 0); fpu matters disproportionately at
    50 sims because so few nodes are ever visited (Stage 7 tuning knob).
    """
    q = fpu if child.visit_count == 0 else -child.value()
    u = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return q + u


def select_child(node: Node, c_puct: float, fpu: float) -> Tuple[int, Node]:
    """Return (move_index, child) maximizing PUCT among `node`'s children."""
    best_score = -math.inf
    best_idx = -1
    best_child: Optional[Node] = None
    for idx, child in node.children.items():
        score = puct_score(node, child, c_puct, fpu)
        if score > best_score:
            best_score, best_idx, best_child = score, idx, child
    return best_idx, best_child


# --------------------------------------------------------------------------- #
# Expansion / evaluation
# --------------------------------------------------------------------------- #
def expand(node: Node, priors: Dict[int, Tuple[chess.Move, float]]) -> None:
    """Create one child per legal move, carrying its prior and its move object."""
    for idx, (move, p) in priors.items():
        node.children[idx] = Node(prior=p, move=move)


def add_dirichlet_noise(root: Node, alpha: float, frac: float,
                        rng: np.random.Generator) -> None:
    """Mix Dirichlet(alpha) noise into the ROOT priors only (self-play exploration):

        P <- (1 - frac) * P + frac * noise

    This is what lets self-play try moves the current net does not already believe
    in. Do not omit it in self-play, and never apply it at internal nodes or during
    evaluation matches (those want the net's honest opinion).
    """
    idxs = list(root.children.keys())
    if not idxs:
        return
    noise = rng.dirichlet([alpha] * len(idxs))
    for idx, n in zip(idxs, noise):
        c = root.children[idx]
        c.prior = (1.0 - frac) * c.prior + frac * float(n)


def terminal_value(board: chess.Board) -> float:
    """True game value from the side-to-move's perspective for a finished game.
    Checkmate => the side to move has just been mated => -1. Any draw => 0.
    (Assumes board.is_game_over(claim_draw=True) is already True.)"""
    if board.is_checkmate():
        return -1.0
    return 0.0


# --------------------------------------------------------------------------- #
# Backup
# --------------------------------------------------------------------------- #
def backup(search_path: List[Node], value: float) -> None:
    """Propagate `value` (the leaf value, in the LEAF mover's frame) back up the
    path, flipping sign every ply so each node accumulates value in its own mover
    frame. Classic sign-bug site: the flip happens once per edge crossed."""
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


# --------------------------------------------------------------------------- #
# Output distribution
# --------------------------------------------------------------------------- #
def visit_distribution(root: Node) -> np.ndarray:
    """pi(a) = N(a) / sum_b N(b), a (NUM_MOVES,) vector over canonical indices with
    support exactly on the root's legal moves. This is the tau=1 target stored for
    training; game.sample_move_index applies temperature when picking the move to
    play. Falls back to the priors only in the degenerate zero-simulation case."""
    pi = np.zeros(NUM_MOVES, dtype=np.float32)
    total = sum(c.visit_count for c in root.children.values())
    if total == 0:
        for idx, c in root.children.items():
            pi[idx] = c.prior
        return pi
    for idx, c in root.children.items():
        pi[idx] = c.visit_count / total
    return pi


# --------------------------------------------------------------------------- #
# Network-backed evaluator (Stage 2 masking done inline)
# --------------------------------------------------------------------------- #
def make_net_evaluator(net) -> Evaluator:
    """Wrap a MarchHare network into an Evaluator: encode -> forward -> mask ->
    softmax over the legal moves. Torch is imported lazily so the MCTS core above
    stays importable/testable without torch installed.

    BatchNorm discipline (a silent-corruptor site): inference runs in eval mode
    under no_grad, so BN uses its running stats, never per-call batch stats.
    """
    import torch  # local import: keeps the search core torch-free

    device = next(net.parameters()).device
    net.eval()

    @torch.no_grad()
    def _evaluate(board: chess.Board):
        # encode() is already canonicalized to the mover; shape (20, 8, 8).
        x = torch.from_numpy(encode(board)).unsqueeze(0).to(device)   # (1, 20, 8, 8)
        logits, value = net(x)
        logits = logits[0]                                            # (1880,)

        legal = list(board.legal_moves)
        # move_to_index flips into the mover frame for Black, matching encode().
        idxs = [move_to_index(board, m) for m in legal]

        # Additive -inf mask BEFORE softmax (never "softmax then zero out"): the
        # result is a proper distribution over exactly the legal moves.
        mask = torch.full_like(logits, float("-inf"))
        mask[idxs] = 0.0
        probs = torch.softmax(logits + mask, dim=0)

        priors = {idx: (mv, float(probs[idx])) for idx, mv in zip(idxs, legal)}
        return float(value.item()), priors

    return _evaluate


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def run_mcts(board: chess.Board,
             net=None,
             *,
             num_simulations: int = 64,
             c_puct: float = 2.0,
             add_noise: bool = True,
             dirichlet_alpha: float = 0.3,
             dirichlet_frac: float = 0.25,
             fpu: float = 0.0,
             rng: Optional[np.random.Generator] = None,
             evaluate_fn: Optional[Evaluator] = None,
             return_root: bool = False):
    """Run PUCT MCTS from `board` and return the visit distribution pi.

    `board` should be a detached copy (game.board_copy()); it is never mutated in
    place — descents push moves onto copies. Provide either `net` (a MarchHare) or
    a custom `evaluate_fn` (used by the tests here and, later, by the Stage-6
    batched evaluator).

    Returns
        pi : np.ndarray (NUM_MOVES,) float32 — tau=1 normalized visit counts over
             canonical indices (legal support only). Temperature for the move that
             is actually PLAYED is applied later by game.sample_move_index.
        If return_root=True, returns (pi, root) so tests can inspect the tree.
    """
    if evaluate_fn is None:
        if net is None:
            raise ValueError("run_mcts needs either `net` or `evaluate_fn`")
        evaluate_fn = make_net_evaluator(net)
    rng = rng or np.random.default_rng()

    # -- Root: evaluate, expand, then (self-play only) inject exploration noise --
    root = Node()
    root.board = board
    if root.board.is_game_over(claim_draw=True):
        # Nothing to search from a finished position.
        return (visit_distribution(root), root) if return_root else visit_distribution(root)

    _, priors = evaluate_fn(root.board)      # root value is unused; only priors matter
    expand(root, priors)
    if add_noise:
        add_dirichlet_noise(root, dirichlet_alpha, dirichlet_frac, rng)

    # -- Simulations ----------------------------------------------------------
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # SELECT: descend by PUCT until we reach an unexpanded node (a leaf).
        while node.expanded():
            idx, child = select_child(node, c_puct, fpu)
            if child.board is None:              # materialize the child's board lazily
                child.board = node.board.copy()  # copy() keeps history -> in-search
                child.board.push(child.move)     # repetition/50-move detection works
            node = child
            search_path.append(node)

        # EXPAND + EVALUATE the leaf.
        if node.board.is_game_over(claim_draw=True):
            value = terminal_value(node.board)   # true outcome, in the leaf mover's frame
        else:
            value, priors = evaluate_fn(node.board)
            expand(node, priors)

        # BACKUP with per-ply sign flip.
        backup(search_path, value)

    pi = visit_distribution(root)
    return (pi, root) if return_root else pi