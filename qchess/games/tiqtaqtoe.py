"""Quantum TiqTaqToe: rule sets of Evert van Nieuwenburg's game and of quantumlib/unitary's open version.

State: a superposition of classical 3x3 boards (each square a qutrit: 0 empty, 1 X, 2 O), kept sparse as
``{board tuple: amplitude}``, plus classical bookkeeping: ``touched`` (squares that are not "empty", i.e.
quantumlib's complement of ``empty_squares``: marked, or part of a split), ``measured`` (squares whose content
counts for three-in-a-row) and ``units`` (which squares share a split / entanglement).  X ("w") moves first.

Rule sets (``rules=``):

* ``"q0"``  - classical tic-tac-toe (tiqtaqtoe.com Quantumness 0).
* ``"q1"``  - tiqtaqtoe.com Quantumness 1, "Superpositions", the site's default (public tutorial text, read
              26 Oct 2026): a classical move puts a mark on an empty square and counts at once; a quantum
              move splits the mark 50/50 over two empty squares; "Once the board is full, squares are
              automatically measured. Each pair randomly chooses which of the squares it recombines in";
              the other square of a pair is empty again; "If there is a winner after a measurement, the round
              ends. If not, the game continues."
* ``"q2"``  - tiqtaqtoe.com Quantumness 2, "Entanglement": as q1, plus "normal moves are not immediately
              measured. Only measured squares count for the win", plus the entangling move: from an empty
              square onto an opponent's (unmeasured, unsplit) mark; "if one collapses to X, the other
              collapses to O".  Implemented as flip + sqrt-iSWAP, i.e. (mine, theirs) + i (theirs, mine).
* ``"ql1"`` / ``"ql2"`` / ``"ql3"`` - quantumlib/unitary ``examples/tic_tac_toe`` ``TicTacRules.QUANTUM_V1/V2/V3``
              (Apache-2.0, commit a5eea27, 7 Oct 2026): a move is one square (``QuditFlip(0 <-> mark)`` on an
              empty square) or an ordered pair (a, b): ``TicTacSplit`` = ``QuditXGate(0 <-> mark)`` on b if b is
              empty else on a, then ``QuditSplitGate(mark)`` (V1, V2) or ``QuditISwapPowGate(0.5)`` (V3) on
              (that square, the other).  V1: both squares empty; V2/V3: any two squares.  The whole board is
              measured when no square is empty (``empty_squares`` is empty); only the last measurement counts
              (``eval_board(last_result)``).

Common choices (pre-registered in the report; the site's own code is not public):
* both players with three in a row after one measurement (``BOTH_WIN`` in quantumlib) = draw;
* a full, measured board without a line = draw;
* q2: manual "hold to measure", decoherence and the 3-square / interference levels (Q3, Q4) are not modelled;
* ql2 / ql3 can empty squares again (the 0 <-> mark flip on one's own mark), so games may not end: matches use
  a ply cap with a draw.  q0 / q1 / q2 end within 9 moves: every move ends up as exactly one permanent mark.

Moves are tuples: ``("c", s)`` classical, ``("s", a, b)`` split (q1 / q2 / ql1: a < b; ql2 / ql3: ordered),
``("e", src, dst)`` entangle (q2).  Squares 0..8 row by row (quantumlib letters a..i).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .core import BLACK, WHITE, GameBoard, forced_value

N = 9
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
RULES = ("q0", "q1", "q2", "ql1", "ql2", "ql3", "t1", "t2")
SPLIT_ON_EMPTY = ("q1", "q2", "ql1", "t1", "t2")
ENTANGLING = ("q2", "t2")
MEASURED_AT_ONCE = ("q0", "q1", "t1")
TOURNAMENT = ("t1", "t2")
EMPTY_BOARD = (0,) * N
PRUNE = 1e-14
S2 = 1.0 / math.sqrt(2.0)

# quantumlib QuditSplitGate: g = e^{i pi/4}, coeff = -i g sin(pi/4), diag = g cos(pi/4)
_G = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
SPLIT_COEFF = -1j * _G * S2
SPLIT_DIAG = _G * S2
# QuditISwapPowGate(exponent 0.5): coeff = i sin(pi/4), diag = cos(pi/4)
ISW_COEFF = 1j * S2
ISW_DIAG = S2


def mark_of(color: str) -> int:
    return 1 if color == WHITE else 2


def lines_of(board, value: int) -> bool:
    return any(board[a] == value and board[b] == value and board[c] == value for a, b, c in LINES)


def judge(counted) -> str | None:
    """Result of a board of counted marks (0 = not counted): both lines -> draw (quantumlib BOTH_WIN)."""
    x, o = lines_of(counted, 1), lines_of(counted, 2)
    if x and o:
        return "draw"
    if x:
        return "white"
    if o:
        return "black"
    return None


# ---- sparse qutrit gates ----------------------------------------------------------------------------------
def _flip(amps: dict, s: int, mark: int) -> dict:
    out = {}
    for b, a in amps.items():
        v = b[s]
        if v == 0 or v == mark:
            b = b[:s] + ((mark if v == 0 else 0),) + b[s + 1:]
        out[b] = out.get(b, 0) + a
    return out


def _two(amps: dict, s1: int, s2: int, pairs, coeff: complex, diag: complex) -> dict:
    """Apply a two-qutrit gate that maps basis (x, y) -> diag (x, y) + coeff (y, x) for (x, y) in ``pairs``
    (both orders listed) and leaves every other basis state alone.  (s1, s2) = cirq operand order."""
    out: dict = {}
    for b, a in amps.items():
        x, y = b[s1], b[s2]
        if (x, y) in pairs:
            out[b] = out.get(b, 0) + diag * a
            l = list(b)
            l[s1], l[s2] = y, x
            sb = tuple(l)
            out[sb] = out.get(sb, 0) + coeff * a
        else:
            out[b] = out.get(b, 0) + a
    return {b: a for b, a in out.items() if abs(a) ** 2 > PRUNE}


def split_gate(amps, s1, s2, mark):
    return _two(amps, s1, s2, {(0, mark), (mark, 0)}, SPLIT_COEFF, SPLIT_DIAG)


def iswap_half(amps, s1, s2):
    return _two(amps, s1, s2, {(x, y) for x in range(3) for y in range(3) if x != y}, ISW_COEFF, ISW_DIAG)


# ---- the board -----------------------------------------------------------------------------------------
class TQTBoard(GameBoard):
    rules_variant = "tiqtaqtoe"

    def __init__(self, rules: str = "q1", seed: int = 0):
        if rules not in RULES:
            raise ValueError(f"rules must be one of {RULES}")
        self.rules = rules
        self.amps = {EMPTY_BOARD: 1.0 + 0j}
        self.touched = 0             # bitmask
        self.measured = 0            # bitmask: squares whose (definite) mark counts
        self.units: tuple = ()       # tuple of frozensets of squares in one split / entangled group
        self.counted = EMPTY_BOARD   # the marks that count for three-in-a-row (quantumlib: last_result)
        self.turn = WHITE
        self.history: list = []
        self.collapses = 0
        self._result: str | None = None
        self.rng = np.random.default_rng(seed)

    # ---- protocol ----
    def copy(self) -> "TQTBoard":
        c = TQTBoard.__new__(TQTBoard)
        c.rules = self.rules
        c.amps = dict(self.amps)
        c.touched, c.measured, c.units, c.counted = self.touched, self.measured, self.units, self.counted
        c.turn = self.turn
        c.history = list(self.history)
        c.collapses = self.collapses
        c._result = self._result
        c.rng = self.rng             # shared: engines clone without advancing it (MCTSEngine._clone)
        return c

    def result(self) -> str | None:
        return self._result

    def definite(self, s: int) -> int | None:
        """The square's value if it is the same in every branch, else None."""
        vals = {b[s] for b in self.amps}
        return vals.pop() if len(vals) == 1 else None

    def empty_squares(self) -> list[int]:
        return [s for s in range(N) if not (self.touched >> s) & 1]

    def legal_moves(self, allow_quantum: bool = True) -> list:
        if self._result is not None:
            return []
        empty = self.empty_squares()
        ms = [("c", s) for s in empty]
        if not allow_quantum or self.rules == "q0":
            return ms
        if self.rules in SPLIT_ON_EMPTY:
            ms += [("s", a, b) for i, a in enumerate(empty) for b in empty[i + 1:]]
        else:                                   # ql2 / ql3: any ordered pair of distinct squares
            ms += [("s", a, b) for a in range(N) for b in range(N) if a != b]
        if self.rules in ENTANGLING:
            opp = mark_of(BLACK if self.turn == WHITE else WHITE)
            in_unit = set().union(*self.units) if self.units else set()
            targets = [s for s in range(N) if (self.touched >> s) & 1 and not (self.measured >> s) & 1
                       and s not in in_unit and self.definite(s) == opp]
            ms += [("e", a, t) for a in empty for t in targets]
        if self.rules in TOURNAMENT:            # [i, "M"]: measure an unmeasured non-empty square + its group
            ms += [("m", s) for s in range(N) if (self.touched >> s) & 1 and not (self.measured >> s) & 1]
        return ms

    def _measured_group(self, s) -> tuple:
        for u in self.units:
            if s in u:
                return tuple(sorted(x for x in u if not (self.measured >> x) & 1))
        return (s,)

    def _after_move(self, m):
        """(amps, touched, measured, units) right after ``m``, before any collapse."""
        mark = mark_of(self.turn)
        amps, touched, measured, units = self.amps, self.touched, self.measured, self.units
        kind = m[0]
        if kind == "c":
            s = m[1]
            amps = _flip(amps, s, mark)
            touched |= 1 << s
            if self.rules in MEASURED_AT_ONCE:
                measured |= 1 << s
        elif kind == "s":
            a, b = m[1], m[2]
            empty_b = not (touched >> b) & 1
            s1, s2 = (b, a) if empty_b else (a, b)          # TicTacSplit(square1 = b if b empty else a)
            amps = _flip(amps, s1, mark)
            if self.rules == "ql3":
                amps = iswap_half(amps, s1, s2)
            else:
                amps = split_gate(amps, s1, s2, mark)
            touched |= (1 << a) | (1 << b)
            units = self._merge_units(units, {a, b})
        elif kind == "e":
            src, dst = m[1], m[2]
            amps = _flip(amps, src, mark)
            amps = iswap_half(amps, src, dst)
            touched |= 1 << src
            units = self._merge_units(units, {src, dst})
        else:
            raise ValueError(m)
        return amps, touched, measured, units

    @staticmethod
    def _merge_units(units, new):
        new = set(new)
        keep = []
        for u in units:
            if u & new:
                new |= u
            else:
                keep.append(u)
        return tuple(keep) + (frozenset(new),)

    def _full(self, touched) -> bool:
        return touched == (1 << N) - 1

    def outcome_probs(self, m) -> list:
        if m[0] == "m":                         # outcomes = joint values of the measured group
            g = self._measured_group(m[1])
            acc: dict = {}
            for b, a in self.amps.items():
                k = tuple(b[x] for x in g)
                acc[k] = acc.get(k, 0.0) + abs(a) ** 2
            tot = sum(acc.values())
            return [(k, p / tot) for k, p in sorted(acc.items()) if p / tot > 1e-12]
        amps, touched, _, _ = self._after_move(m)
        if not self._full(touched):
            return [(None, 1.0)]
        acc: dict = {}
        for b, a in amps.items():
            acc[b] = acc.get(b, 0.0) + abs(a) ** 2
        tot = sum(acc.values())
        outs = [(b, p / tot) for b, p in sorted(acc.items()) if p / tot > 1e-12]
        return outs

    def apply(self, m, outcome=None) -> "TQTBoard":
        if m[0] == "m":
            return self._apply_measure(m, tuple(outcome))
        c = self.copy()
        amps, touched, measured, units = self._after_move(m)
        c.history.append(m)
        if self._full(touched):
            if outcome is None:
                raise ValueError("a collapsing move needs its outcome")
            b = tuple(outcome)
            c.amps = {b: 1.0 + 0j}
            c.touched = sum(1 << s for s in range(N) if b[s])
            c.measured = c.touched
            c.units = ()
            c.counted = b
            c.collapses += 1
        else:
            c.amps, c.touched, c.measured, c.units = amps, touched, measured, units
            if m[0] == "c" and self.rules in MEASURED_AT_ONCE:   # measured at once: counts now
                s = m[1]
                c.counted = self.counted[:s] + (mark_of(self.turn),) + self.counted[s + 1:]
        c.turn = BLACK if self.turn == WHITE else WHITE
        c._result = c._judge()
        return c

    def _apply_measure(self, m, vals) -> "TQTBoard":
        """Tournament measurement move: collapse square m[1] and its group to ``vals``; a square found empty is
        empty (unmeasured) again, the others are measured; the turn passes."""
        g = self._measured_group(m[1])
        c = self.copy()
        c.history.append(m)
        kept = {b: a for b, a in self.amps.items() if tuple(b[x] for x in g) == vals}
        nrm = sum(abs(a) ** 2 for a in kept.values()) ** 0.5
        c.amps = {b: a / nrm for b, a in kept.items()}
        cnt = list(self.counted)
        for x, v in zip(g, vals):
            if v:
                c.measured |= 1 << x
            else:
                c.touched &= ~(1 << x)
                c.measured &= ~(1 << x)
            cnt[x] = v
        c.counted = tuple(cnt)
        c.units = tuple(u for u in self.units if not (u & set(g)))
        c.turn = BLACK if self.turn == WHITE else WHITE
        c._result = c._judge()
        return c

    def _judge(self) -> str | None:
        if self.rules in TOURNAMENT:            # more measured lines wins; equal (> 0) = both = draw
            x = sum(all(self.counted[i] == 1 for i in L) for L in LINES)
            o = sum(all(self.counted[i] == 2 for i in L) for L in LINES)
            if x or o:
                return "white" if x > o else "black" if o > x else "draw"
            return "draw" if all(self.counted) else None
        r = judge(self.counted)
        if r is not None:
            return r
        if all(self.counted):
            return "draw"                       # full counted board, no line
        return None

    # ---- evaluator support ----
    def branches(self) -> list:
        acc: dict = {}
        for b, a in self.amps.items():
            acc[b] = acc.get(b, 0.0) + abs(a) ** 2
        return sorted(((p, b) for b, p in acc.items()), key=lambda x: x[1])

    def key(self) -> tuple:
        """Exact state key (amplitudes rounded) for transposition tables."""
        amps = tuple(sorted((b, round(a.real, 10), round(a.imag, 10)) for b, a in self.amps.items()))
        return (self.rules, self.turn, self.touched, self.measured, self.counted, amps, self._result)

    def summary(self) -> dict:
        return {"branches": [["".join(".XO"[v] for v in b), round(p, 6)] for p, b in self.branches()],
                "touched": self.touched, "measured": self.measured, "collapses": self.collapses,
                "result": self._result}

    def __repr__(self):
        return f"TQTBoard({self.rules}, {self.summary()})"


def move_text(m) -> str:
    L = "abcdefghi"
    if m[0] == "m":
        return L[m[1]] + "!"
    if m[0] == "c":
        return L[m[1]]
    if m[0] == "s":
        return L[m[1]] + L[m[2]]
    return L[m[1]] + ">" + L[m[2]]


def move_from_json(a) -> tuple:
    return tuple(a)


# ---- classical tic-tac-toe engine (the "Fairy-Stockfish" of this game) -------------------------------------
@lru_cache(maxsize=None)
def classical_value(board: tuple, mover: int) -> tuple[int, int]:
    """Perfect classical tic-tac-toe on a 9-tuple (0/1/2), ``mover`` to play: (v, plies) with v in {-1, 0, 1}
    from the mover's view (+1 = mover wins) and plies to the end under optimal play (win fast, lose slow)."""
    r = judge(board)
    if r == "draw":
        return 0, 0
    if r is not None:
        return (1 if (r == "white") == (mover == 1) else -1), 0
    empt = [s for s in range(N) if board[s] == 0]
    if not empt:
        return 0, 0
    best = None
    for s in empt:
        nb = board[:s] + (mover,) + board[s + 1:]
        v, d = classical_value(nb, 3 - mover)
        cand = (-v, d + 1)
        if best is None or _better(cand, best):
            best = cand
    return best


def _better(a, b) -> bool:
    if a[0] != b[0]:
        return a[0] > b[0]
    if a[0] > 0:
        return a[1] < b[1]
    if a[0] < 0:
        return a[1] > b[1]
    return False


def classical_branch_value(board: TQTBoard, branch: tuple, me: str) -> float:
    """Value of one classical branch for ``me``: perfect classical tic-tac-toe from that board with the side
    to move to play, every mark counted (as Fairy-Stockfish sees a branch board as ordinary chess)."""
    mover = mark_of(board.turn)
    v, d = classical_value(tuple(branch), mover)
    if v == 0:
        return 0.0
    win = (v > 0) == (board.turn == me)
    return forced_value(win, d)


__all__ = ["TQTBoard", "RULES", "LINES", "judge", "classical_value", "classical_branch_value", "move_text",
           "mark_of"]
