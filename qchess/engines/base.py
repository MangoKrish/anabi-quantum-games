"""Shared interface and position helpers for the hybrid quantum-chess engines.

Every engine (search family x evaluator x decision rule x Bayesian layer) implements ``Engine``
and is built from a plain config dict by ``qchess.engines.make_engine``, so the arena can run
any combination from a JSON list.

Conventions used by every module in this package:

* Values are always from the point of view of a fixed colour ``me`` ("w" or "b") and lie in
  [-1, 1]: +1 = I have won (opponent's king probability is 0), -1 = I have lost, 0 = draw/even.
* A *chance node* is a measuring move (capture, blocked move).  ``chance_children`` gives its
  exact outcomes ``[(probability, child_board)]`` using ``ForcedRNG``; nothing is sampled.
* Engines must never mutate the board they are given; they work on ``board.copy()``.
* Boards of another rule set (``board.rules_variant`` set, e.g. ``qchess.engines.ddrules.DDBoard``)
  bring their own ``legal_moves`` / ``outcome_probs`` / ``apply``; the helpers below dispatch to
  them (added 20 October 2026; a ``QuantumBoard`` has no such attribute, so nothing else changes).
* Engines must respect ``Budget``: check ``budget.expired()`` regularly and always return a
  legal move (fall back to the best move found so far).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..board import EPS, MoveResult, QuantumBoard
from ..geometry import BLACK, WHITE, other
from ..moves import MEASURING, Move, MoveType

WIN = 1.0
LOSS = -1.0


# ---- time / node budget ------------------------------------------------------------------------
@dataclass
class Budget:
    """Thinking budget for one move: wall-clock seconds and/or a node cap (whichever ends first).

    ``soft`` is the fraction of the time after which iterative-deepening engines should not start
    a new iteration."""
    seconds: float = 1.0
    nodes: int | None = None
    soft: float = 0.5
    start: float = field(default_factory=time.perf_counter)
    used_nodes: int = 0

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def expired(self) -> bool:
        if self.nodes is not None and self.used_nodes >= self.nodes:
            return True
        return self.elapsed() >= self.seconds

    def soft_expired(self) -> bool:
        if self.nodes is not None and self.used_nodes >= self.soft * self.nodes:
            return True
        return self.elapsed() >= self.soft * self.seconds

    def tick(self, n: int = 1) -> None:
        self.used_nodes += n


class OutOfTime(Exception):
    """Raised inside a search when the budget expires; the engine returns its best-so-far move."""


# ---- root analysis -------------------------------------------------------------------------------
@dataclass
class Candidate:
    """A root move with the statistics a search produced for it (all from the mover's view).

    ``mean``: the search value (expectation over measurement outcomes, backed-up minimax/MCTS
    value).  ``worst``: the lowest value over measurement outcomes / branches seen, if the search
    tracks it (else equal to ``mean``).  ``std``: spread over outcomes/samples (0 if unknown).
    ``visits``: MCTS visits (or 1).  ``p_loss``: probability mass judged lost (0 if unknown)."""
    move: Move
    mean: float
    worst: float | None = None
    std: float = 0.0
    visits: int = 1
    p_loss: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.worst is None:
            self.worst = self.mean


# ---- the engine interface ------------------------------------------------------------------------
class Engine:
    name: str = "engine"

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(cfg or {})
        self.color: str = WHITE
        self.rng = np.random.default_rng(0)
        self._stats: dict = {}

    def new_game(self, color: str, seed: int) -> None:
        """Called once before a game; ``color`` is the side this engine plays."""
        self.color = color
        self.rng = np.random.default_rng(seed)
        self._stats = {}

    def choose(self, board: QuantumBoard, budget: Budget) -> Move:
        raise NotImplementedError

    def observe(self, before: QuantumBoard, move: Move, result: MoveResult, after: QuantumBoard) -> None:
        """Called after every move by either side (``before`` / ``after`` are copies)."""

    def stats(self) -> dict:
        return dict(self._stats)

    def close(self) -> None:
        pass


# ---- position helpers ----------------------------------------------------------------------------
class ForcedRNG:
    """Stand-in for ``board.rng``: the next measurement gives ``outcome``
    (``SparseState.measure`` returns 1 iff ``rng.random() < p1``)."""

    def __init__(self, outcome: int):
        self.outcome = outcome

    def random(self):
        return 0.0 if self.outcome == 1 else 1.0

    def integers(self, *a, **k):  # pragma: no cover - only used by board.copy()
        return 0


def is_quantum(m: Move) -> bool:
    return m.is_split or m.is_merge


def move_type(board: QuantumBoard, m: Move) -> MoveType:
    return board.classify(m)


def is_measuring(board: QuantumBoard, m: Move) -> bool:
    return board.classify(m) in MEASURING


def outcome_probs(board: QuantumBoard, m: Move) -> list[tuple[int | None, float]]:
    """``[(outcome, probability)]`` of a move: ``[(None, 1.0)]`` for a unitary move."""
    if getattr(board, "rules_variant", None) is not None:      # another rule set (qchess.engines.ddrules)
        return board.outcome_probs(m)
    mt = board.classify(m)
    if mt not in MEASURING:
        return [(None, 1.0)]
    p1 = board.state.prob(board.m1_predicate(m, mt))
    return [(o, p) for o, p in ((1, p1), (0, 1.0 - p1)) if p > EPS]


def apply(board: QuantumBoard, m: Move, outcome: int | None = None) -> QuantumBoard:
    """A copy of ``board`` with ``m`` played and (for a measuring move) the given outcome forced."""
    # Same child as board.copy() + ForcedRNG(outcome) + push(m), bitwise (tests/engines/test_fastgen.py),
    # about 12x faster: cached gate matrices and a cheap legality check (qchess.engines.fastgen).
    if getattr(board, "rules_variant", None) is not None:      # another rule set (qchess.engines.ddrules)
        return board.apply(m, outcome)
    from .fastgen import fast_apply
    return fast_apply(board, m, outcome)


def chance_children(board: QuantumBoard, m: Move) -> list[tuple[float, QuantumBoard]]:
    """Exact successors of a move: one child for a unitary move, one per outcome for a
    measuring move, each with its Born probability."""
    return [(p, apply(board, m, o)) for o, p in outcome_probs(board, m)]


def terminal_value(board: QuantumBoard, me: str) -> float | None:
    """+1 / -1 / 0 if the game is over (Cantwell Rule 10) from ``me``'s view, else None."""
    r = board.result()
    if r is None:
        return None
    te = getattr(board, "terminal_expect", None)      # a measured ending (qnrules_r1): its exact expected value
    if te is not None:
        v = te(me)
        if v is not None:
            return v
    if r == "draw":
        return 0.0
    return WIN if (r == "white") == (me == WHITE) else LOSS


def king_balance(board: QuantumBoard, me: str) -> float:
    """P(opponent's king gone) - P(my king gone): a smooth version of the Rule-10 outcome,
    used to adjudicate games that hit the ply cap."""
    return (1.0 - board.king_prob(other(me))) - (1.0 - board.king_prob(me))


def branches(board: QuantumBoard) -> list[tuple[float, int, list]]:
    """Classical boards in the superposition: ``[(Born weight, occupancy bits, types)]``,
    ancilla bits stripped and equal boards merged."""
    mask = (1 << board.nsq) - 1
    acc: dict[int, float] = {}
    for b, a in board.state.amps.items():
        bits = b & mask
        acc[bits] = acc.get(bits, 0.0) + abs(a) ** 2
    out = []
    for bits, w in acc.items():
        types = [board.types[q] if (bits >> q) & 1 else None for q in range(board.nsq)]
        out.append((w, bits, types))
    return out


def position_key(board: QuantumBoard) -> tuple:
    """Hashable key for transposition tables: side to move, piece types and the occupancy
    distribution of the board squares (amplitude phases ignored, probabilities rounded).

    Two positions with the same key can differ in phases, which matter for later interference,
    so engines should treat the key as a heuristic cache key, not as an identity."""
    probs = tuple(sorted((bits, round(w, 9)) for w, bits, _ in branches(board)))
    return (board.turn, tuple(board.types), probs)


def legal_moves(board: QuantumBoard, allow_quantum: bool = True) -> list[Move]:
    # board.legal_moves() (same list, same order; tests/engines/test_fastgen.py), 50-90x faster.
    if getattr(board, "rules_variant", None) is not None:      # another rule set (qchess.engines.ddrules)
        return board.legal_moves(allow_quantum)
    from .fastgen import fast_legal_moves
    if not allow_quantum:
        return fast_legal_moves(board, quantum=False) or fast_legal_moves(board)
    return fast_legal_moves(board)


def other_color(c: str) -> str:
    return other(c)


__all__ = ["Budget", "OutOfTime", "Candidate", "Engine", "ForcedRNG", "is_quantum", "move_type", "is_measuring",
           "outcome_probs", "apply", "chance_children", "terminal_value", "king_balance", "branches",
           "position_key", "legal_moves", "other_color", "WIN", "LOSS", "WHITE", "BLACK"]
