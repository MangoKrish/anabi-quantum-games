"""Quantum TiqTaqToe on the rules of Evert van Nieuwenburg's AI-tournament kit (rule set "kit", 7 November 2026).

The kit's referee (``local-tiqtaqtoe-server.js``, sha256 2d11cece...aabd4, on ``@quantumplayed/quantum-game-engine``
1.0.3) is the truth here.  This module is a line-by-line Python port of its game logic, so that Anabi can search
it at Python speed, and it is validated against the referee's own JS by replaying random games
(``reports/engines/evert_kit_20261107/tiqtaqtoe/validate_kit.py``).  It does NOT change ``tiqtaqtoe.py`` (t1 / t2
keep their behaviour); the differences between t1 / t2 and this referee are listed in that report.

What the referee does (and so this port):

* State = the referee's ``roundState``: a sparse superposition of boards (``quantumState``: board string of
  '0' / 'X' / 'O' -> complex amplitude, entries with |a|^2 < 1e-4 dropped after every step, never renormalised),
  ``measuredSquares``, ``emptySquares`` (a square is empty when it is '0' in every branch), ``probabilities``
  (unnormalised marginals, [pEmpty, pX, pO]), ``pairs`` / ``squareToPair`` (entanglement groups found from the
  joint marginals with a 0.01 dependence threshold, kept in persistent slots), the decoherence counters and
  ``currentTurn``.  Squares 0..8 row by row; the basis index puts square 0 first (base 3, 0 < X < O).
* Moves, as the payload's ``legalMoves`` lists them, in this order: ``[i]`` (classical, on an empty square),
  ``[i, j]`` with i < j (quantumness >= 1: split onto two empty squares; >= 2: entangle an empty square with an
  unmeasured square that holds the opponent's token for sure, >= 3: with any unmeasured square without my
  token; 4: a "phase" move between two non-empty unmeasured squares of the same group, or two ungrouped ones,
  whose first square may hold my token), ``[i, "M"]`` (measure an unmeasured non-empty square and its group;
  legal at EVERY quantumness level, also 0 and 1).
* Gates: a token is placed with ``Shift`` (0 -> X / O); quantum moves apply the engine's ``SqrtSwap``
  ((x, y) -> (1+i)/2 (x, y) + (1-i)/2 (y, x) for x != y).  At quantumness 0 and 1 a classical move is measured at
  once; at >= 2 nothing is measured until a measuring move, a full board or decoherence.
* Measurement: ``measureTarget`` per square, one after the other, outcome drawn by the engine's cumulative rule
  (the last possible value takes any probability mass the dropped amplitudes left over).
* Full board (no empty square) with unmeasured tokens: squares whose OLD probabilities (from before this move)
  were a sure X / O are marked measured without a measurement, then every other unmeasured token is measured
  with its group, square by square.
* Win rule: only measured squares count, read from the FIRST branch of the state (lowest basis index); more
  three-in-a-rows wins, an equal positive number = BOTH (each player gets a point), none and no '0' in that
  branch = draw.  Rounds: round 1 and 3 start with X, round 2 with O; 3 rounds by default.

Moves in this module are tuples: ``(i,)``, ``(i, j)``, ``(i, "M")`` (``list(move)`` is the payload form).
A measuring step's outcome is the tuple of ``(square, value)`` pairs drawn, in the referee's order.
"""
from __future__ import annotations

import math

import numpy as np

from .core import BLACK, WHITE, GameBoard

N = 9
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
UNFINISHED, X_WINS, O_WINS, BOTH, DRAW = 0, 1, 2, 3, 4
CHAR = "0XO"
VAL = {"0": 0, "X": 1, "O": 2}
UI_STATES = ((None, (1, 0, 0)), ("X", (0, 1, 0)), ("O", (0, 0, 1)), ("half-x", (0.5, 0.5, 0)),
             ("half-o", (0.5, 0, 0.5)), ("xo", (0, 0.5, 0.5)), ("exO", (0.25, 0.25, 0.5)),
             ("eXo", (0.25, 0.5, 0.25)), ("eX", (0.25, 0.75, 0)), ("eO", (0.25, 0, 0.75)))
FAST_DERIVE = True       # derive_entanglements skips pairs with a sure square (identical result, see there)
KEEP_MAG2 = 0.0001       # serializeEngineState
CLEAN_EPS = 1e-10        # cleanAmplitudes


def near(a: float, b: float) -> bool:
    return abs(a - b) < 0.01


class NeedChoice(Exception):
    """Raised by a chooser to stop a simulation at a measurement (``idx``, ``options`` = [(value, p)])."""

    def __init__(self, idx, options):
        super().__init__(idx)
        self.idx, self.options = idx, options


# ---- amplitude helpers (amps: dict board-tuple -> (re, im), kept in ascending basis order) ---------------------
def _sorted(d: dict) -> dict:
    return {k: d[k] for k in sorted(d)}


def _clean(d: dict) -> dict:
    return {k: a for k, a in d.items() if abs(a[0]) >= CLEAN_EPS or abs(a[1]) >= CLEAN_EPS}


def serialize(amps: dict) -> dict:
    return {k: a for k, a in amps.items() if a[0] * a[0] + a[1] * a[1] >= KEEP_MAG2}


def apply_shift(amps: dict, s: int, val: int) -> dict:
    out = {}
    for b, (re, im) in amps.items():
        nb = b[:s] + ((b[s] + val) % 3,) + b[s + 1:]
        out[nb] = (0.0 + re, 0.0 + im)
    return _sorted(_clean(out))


def apply_sqrtswap(amps: dict, a: int, b: int) -> dict:
    acc: dict = {}

    def add(k, re, im):
        o = acc.get(k)
        acc[k] = (re + 0.0, im + 0.0) if o is None else (o[0] + re, o[1] + im)

    for bd, (re, im) in amps.items():
        x, y = bd[a], bd[b]
        if x == y:
            add(bd, re, im)
        else:
            add(bd, re * 0.5 - im * 0.5, im * 0.5 + re * 0.5)          # amp * (0.5 + 0.5i)
            l = list(bd)
            l[a], l[b] = y, x
            add(tuple(l), re * 0.5 + im * 0.5, im * 0.5 - re * 0.5)    # amp * (0.5 - 0.5i)
    return _sorted(_clean(_clean(acc)))


def selection_probs(marg: dict) -> list:
    """The engine's draw: cumulative over the values in ascending order, ``rand <= cum``; the last value also
    takes 1 - total (dropped mass).  Returns [(value, probability)]."""
    keys = sorted(marg)
    out, cum = [], 0.0
    for i, k in enumerate(keys):
        prev = cum
        cum = cum + marg[k]
        out.append((k, (cum - prev) if i < len(keys) - 1 else 1.0 - prev))
    return [(k, p) for k, p in out if p > 0.0]


def measure_target(amps: dict, idx: int, chooser) -> tuple[dict, int]:
    marg: dict = {}
    for b, (re, im) in amps.items():
        v = b[idx]
        marg[v] = marg.get(v, 0) + (re * re + im * im)
    sel = chooser(idx, selection_probs(marg))
    prob = marg.get(sel, 0)
    if prob > 1e-12:
        f = 1.0 / math.sqrt(prob)
        out = {b: (a[0] * f, a[1] * f) for b, a in amps.items() if b[idx] == sel}
    else:                                   # the engine's zero-probability branch (never drawn by its own rule)
        best, bm = None, -1.0
        for b, (re, im) in amps.items():
            m = re * re + im * im
            if m > bm:
                best, bm = b, m
        base = list(best) if best is not None else [0] * N
        base[idx] = sel
        out = {tuple(base): (1.0, 0.0)}
    return _sorted(_clean(out)), sel


LINE_MASKS = tuple(sum(1 << i for i in L) for L in LINES)


def winner(state: tuple, measured) -> int:
    # bitmask form of the referee's count of measured three-in-a-rows per player (same counts, 11 Nov 2026)
    xm = om = 0
    for i in range(N):
        if measured[i]:
            v = state[i]
            if v == 1:
                xm |= 1 << i
            elif v == 2:
                om |= 1 << i
    x = o = 0
    for L in LINE_MASKS:
        if xm & L == L:
            x += 1
        elif om & L == L:
            o += 1
    if x > o and x > 0:
        return X_WINS
    if o > x and o > 0:
        return O_WINS
    if x > 0 and o > 0 and x == o:
        return BOTH
    if x == 0 and o == 0:
        return UNFINISHED if 0 in state else DRAW
    return UNFINISHED


def compute_probs(amps: dict) -> list:
    pr = [[0, 0, 0] for _ in range(N)]
    for b, (re, im) in amps.items():
        p = re * re + im * im
        for j in range(N):
            pr[j][b[j]] += p
    return pr


_NZ: dict = {}


def empty_flags(amps) -> list:
    """[square is '0' in every branch] - the referee's emptySquares, via a cached per-board bitmask."""
    m = 0
    for b in amps:
        z = _NZ.get(b)
        if z is None:
            z = 0
            for i in range(N):
                if b[i]:
                    z |= 1 << i
            if len(_NZ) < 500_000:
                _NZ[b] = z
        m |= z
    return [not (m >> i) & 1 for i in range(N)]


_DERIVE: dict = {}


def derive_entanglements(amps: dict, previous_pairs) -> tuple[list, dict]:
    """The referee's deriveEntanglements; results cached on (amplitudes, previous pairs) - a pure function."""
    key = (tuple(amps.items()), tuple(None if x is None else tuple(x) for x in (previous_pairs or ())))
    got = _DERIVE.get(key)
    if got is not None:
        return [None if x is None else list(x) for x in got[0]], dict(got[1])
    out = _derive_entanglements(amps, previous_pairs)
    if len(_DERIVE) > 100_000:
        _DERIVE.clear()
    _DERIVE[key] = ([None if x is None else tuple(x) for x in out[0]], dict(out[1]))
    return out


def _derive_entanglements(amps: dict, previous_pairs) -> tuple[list, dict]:
    tot = 0
    for re, im in amps.values():
        tot += re * re + im * im
    if tot == 0:
        tot = 1
    items = [(b, (re * re + im * im) / tot) for b, (re, im) in amps.items()]
    marg = [[0, 0, 0] for _ in range(N)]
    for b, p in items:
        for i in range(N):
            marg[i][b[i]] += p
    adj = [[] for _ in range(N)]
    # Speed-up (11 Nov 2026), same decisions: a square whose marginal has a value of probability >= 1 - 1e-9 is
    # independent of every other square to within 2e-9 << 0.01, so the referee's test is false for every pair
    # that contains it; only pairs of two uncertain squares are tested, with the referee's own arithmetic.
    fuzzy = [i for i in range(N) if max(marg[i]) < 1.0 - 1e-9] if FAST_DERIVE else list(range(N))
    for ii, i in enumerate(fuzzy):
        for j in fuzzy[ii + 1:]:
            joint = [[0, 0, 0] for _ in range(3)]
            for b, p in items:
                joint[b[i]][b[j]] += p
            dep = any(abs(joint[x][y] - marg[i][x] * marg[j][y]) > 0.01 for x in range(3) for y in range(3))
            if dep:
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * N
    comps = []
    for i in range(N):
        if seen[i]:
            continue
        comp, queue = [], [i]
        seen[i] = True
        while queue:
            c = queue.pop(0)
            comp.append(c)
            for nb in adj[c]:
                if not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
        if len(comp) >= 2:
            comps.append(sorted(comp))
    prev = list(previous_pairs or [])
    assigned: set = set()
    slots = []
    for comp in comps:
        best, mx = -1, 0
        for slot, pc in enumerate(prev):
            if pc and slot not in assigned:
                ov = sum(1 for x in comp if x in pc)
                if ov > mx:
                    mx, best = ov, slot
        if best != -1:
            slots.append([comp, best])
            assigned.add(best)
        else:
            slots.append([comp, None])
    for it in slots:
        if it[1] is None:
            s = 0
            while s in assigned:
                s += 1
            it[1] = s
            assigned.add(s)
    pairs: list = [None] * 4
    s2p: dict = {}
    for comp, slot in slots:
        while len(pairs) <= slot:
            pairs.append(None)
        pairs[slot] = comp
        for x in comp:
            s2p[x] = slot
    return pairs, s2p


def closest_ui_state(p) -> str | None:
    best, md = UI_STATES[0][0], math.inf
    for name, q in UI_STATES:
        d = abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2])
        if d < md:
            md, best = d, name
    return best


# ---- the referee's roundState ------------------------------------------------------------------------------------
class RoundState:
    __slots__ = ("q", "dec", "amps", "measured", "empty", "probs", "pairs", "s2p", "turn", "winner", "counters",
                 "just_projected")

    def __init__(self, quantumness: int = 2, decoherence: int = 0, first: str = "X"):
        self.q, self.dec = int(quantumness), int(decoherence)
        self.amps = {(0,) * N: (1.0, 0.0)}
        self.measured = [False] * N
        self.empty = [True] * N
        self.probs = [[1, 0, 0] for _ in range(N)]
        self.pairs: list = [None, None, None, None]
        self.s2p: dict = {}
        self.turn = first
        self.winner = UNFINISHED
        self.counters = [0] * N
        self.just_projected = False

    def clone(self) -> "RoundState":
        c = RoundState.__new__(RoundState)
        c.q, c.dec, c.amps = self.q, self.dec, self.amps          # amps are never mutated in place
        c.measured, c.empty = list(self.measured), list(self.empty)
        c.probs = [list(p) for p in self.probs]
        c.pairs = [None if p is None else list(p) for p in self.pairs]
        c.s2p = dict(self.s2p)
        c.turn, c.winner, c.counters, c.just_projected = self.turn, self.winner, list(self.counters), \
            self.just_projected
        return c

    # -- derived views
    def first_branch(self) -> tuple:
        return next(iter(self.amps))

    def result(self) -> int:
        return winner(self.first_branch(), self.measured)

    def board_labels(self) -> list:
        out = []
        for i in range(N):
            name = closest_ui_state(self.probs[i])
            if name is not None:
                slot = self.s2p.get(i)
                if slot is not None and slot < len(self.pairs) and self.pairs[slot]:
                    pos = self.pairs[slot].index(i) if i in self.pairs[slot] else -1
                    if pos != -1:
                        name = name + str(pos + 1)
            out.append(name)
        return out

    def group_of(self, i: int) -> list:
        sq = [i]
        slot = self.s2p.get(i)
        if slot is not None and slot < len(self.pairs) and self.pairs[slot]:
            sq += [x for x in self.pairs[slot] if x != i and not self.measured[x]]
        return sq


def is_legal(rs: RoundState, move) -> bool:
    if len(move) == 2 and move[1] in ("M", "measure"):
        s = int(move[0])
        if s < 0 or s > 8:
            return False
        return not rs.empty[s] and not rs.measured[s]
    mine, opp = (1, 2) if rs.turn == "X" else (2, 1)
    if len(move) == 1:
        return bool(rs.empty[int(move[0])])
    if len(move) != 2:
        return False
    s1, s2 = int(move[0]), int(move[1])
    if rs.q == 0 and s1 != s2:
        return False
    if rs.measured[s1] or rs.measured[s2]:
        return False
    e1, e2 = rs.empty[s1], rs.empty[s2]
    if not e1 and not e2:
        if rs.q < 4:
            return False
        if near(rs.probs[s1][mine], 0):
            return False
        if rs.s2p.get(s1) != rs.s2p.get(s2):
            return False
    if e1 != e2:
        if rs.q < 2:
            return False
        ne = s2 if e1 else s1
        if not near(rs.probs[ne][mine], 0):
            return False
        if not near(rs.probs[ne][opp], 1) and rs.q < 3:
            return False
    return True


def legal_moves(rs: RoundState) -> list:
    ms = [(i,) for i in range(N) if is_legal(rs, (i,))]
    if rs.q > 0:
        ms += [(i, j) for i in range(N) for j in range(i + 1, N) if is_legal(rs, (i, j))]
    ms += [(i, "M") for i in range(N) if is_legal(rs, (i, "M"))]
    return ms


def _derive(rs: RoundState) -> None:
    rs.pairs, rs.s2p = derive_entanglements(rs.amps, rs.pairs)


def do_classical(rs: RoundState, t: int, token: str, now: bool) -> None:
    rs.amps = serialize(apply_shift(rs.amps, t, 1 if token == "X" else 2))
    rs.empty[t] = False
    if now:
        rs.measured[t] = True
    _derive(rs)


def do_quantum(rs: RoundState, s1: int, s2: int, token: str) -> None:
    val = 1 if token == "X" else 2
    amps = rs.amps
    e1, e2 = rs.empty[s1], rs.empty[s2]
    if e1 and e2:
        amps = apply_shift(amps, s1, val)
        amps = apply_sqrtswap(amps, s1, s2)
    elif e1 != e2:
        es, ne = (s1, s2) if e1 else (s2, s1)
        amps = apply_shift(amps, es, val)
        amps = apply_sqrtswap(amps, ne, es)
    else:
        amps = apply_sqrtswap(amps, s1, s2)
    rs.amps = serialize(amps)
    rs.empty = empty_flags(rs.amps)
    _derive(rs)


def measure_squares(rs: RoundState, squares, chooser) -> None:
    amps = rs.amps
    for idx in squares:
        amps, _ = measure_target(amps, idx, chooser)
    rs.amps = serialize(amps)
    for idx in squares:
        rs.measured[idx] = True
    rs.empty = empty_flags(rs.amps)
    for i in range(N):
        if rs.empty[i]:
            rs.measured[i] = False
    _derive(rs)
    rs.winner = rs.result()
    rs.probs = compute_probs(rs.amps)


def compute_decoherence(rs: RoundState, limit: int, touched, chooser) -> bool:
    if limit <= 0:
        return False
    for i in touched:
        rs.counters[i] = 0
    todo = []
    for i in range(N):
        if not rs.measured[i] and not rs.empty[i]:
            if i not in touched:
                rs.counters[i] += 1
            if rs.counters[i] >= limit:
                todo.append(i)
    if not todo:
        return False
    uniq = list(todo)
    for sq in todo:
        slot = rs.s2p.get(sq)
        if slot is not None and slot < len(rs.pairs) and rs.pairs[slot]:
            for x in rs.pairs[slot]:
                if x not in uniq and not rs.measured[x]:
                    uniq.append(x)
    measure_squares(rs, uniq, chooser)
    for x in uniq:
        rs.counters[x] = 0
    return True


def _toggle(rs: RoundState) -> None:
    rs.turn = "O" if rs.turn == "X" else "X"


def do_move(rs: RoundState, move, chooser, project: bool = True) -> None:
    """The referee's ``doMove(settings, roundState, move, token, project)`` on ``rs`` (mutated in place)."""
    if not is_legal(rs, move):
        raise ValueError(f"illegal move {move!r}")
    token = rs.turn
    now = rs.q in (0, 1)
    if len(move) == 1:
        do_classical(rs, int(move[0]), token, now)
    elif move[1] in ("M", "measure"):
        measure_squares(rs, rs.group_of(int(move[0])), chooser)
    else:
        a, b = int(move[0]), int(move[1])
        if a == b:
            do_classical(rs, a, token, now)
        else:
            do_quantum(rs, a, b, token)
    rs.just_projected = False
    unmeasured = any(not m and not e for m, e in zip(rs.measured, rs.empty))
    if all(not e for e in rs.empty) and project and unmeasured:
        for i in range(N):                                   # resolveDeterministicSquares (old probabilities)
            if not rs.empty[i]:
                p = rs.probs[i]
                if near(p[1], 1) or near(p[2], 1):
                    rs.measured[i] = True
        for i in range(N):
            if not rs.measured[i] and not rs.empty[i]:
                measure_squares(rs, rs.group_of(i), chooser)
        rs.probs = compute_probs(rs.amps)
        rs.winner = rs.result()
        rs.just_projected = True
        _toggle(rs)
        return
    rs.probs = compute_probs(rs.amps)
    if rs.dec > 0:
        touched = [int(x) for x in move if not isinstance(x, str)]
        if compute_decoherence(rs, rs.dec, touched, chooser):
            rs.probs = compute_probs(rs.amps)
            rs.winner = rs.result()
            rs.just_projected = True
            _toggle(rs)
            return
    rs.winner = rs.result()
    _toggle(rs)


# ---- payload <-> state ------------------------------------------------------------------------------------------
def board_string(b: tuple) -> str:
    return "".join(CHAR[v] for v in b)


def state_from_payload(p: dict, counters=None) -> RoundState:
    st = p.get("settings") or {}
    rs = RoundState(int(st.get("quantumness", 2)), int(st.get("decoherence", 0) or 0), p.get("currentTurn")
                    or p.get("player") or "X")
    amps = {}
    for s, a in p["quantumState"].items():
        amps[tuple(VAL[c] for c in s)] = (float(a.get("real", 0.0)), float(a.get("imaginary", 0.0)))
    rs.amps = _sorted(amps)
    # every field the referee sends is read as sent; a field that is MISSING (not the referee's format) is derived
    # from quantumState the way the referee derives it (measuredSquares cannot be derived: then none measured)
    rs.measured = [bool(x) for x in p["measuredSquares"]] if "measuredSquares" in p else [False] * N
    rs.empty = [bool(x) for x in p["emptySquares"]] if "emptySquares" in p else empty_flags(rs.amps)
    rs.probs = ([[float(v) for v in x] for x in p["probabilities"]] if "probabilities" in p
                else compute_probs(rs.amps))
    if "pairs" in p or "squareToPair" in p:
        rs.pairs = [None if x is None else [int(v) for v in x] for x in (p.get("pairs") or [None] * 4)]
        rs.s2p = {int(k): int(v) for k, v in (p.get("squareToPair") or {}).items()}
    else:
        rs.pairs, rs.s2p = derive_entanglements(rs.amps, None)
    if counters is not None:
        rs.counters = list(counters)
    rs.winner = rs.result()
    return rs


def payload_from_state(rs: RoundState, round_number: int = 1, timeout_ms: int = 5000, game_id="local-test-game"):
    """The request the referee's ``fetchRemoteAiMove`` would send for this state."""
    return {"gameId": game_id, "roundId": f"round-{round_number}", "player": rs.turn, "currentTurn": rs.turn,
            "board": rs.board_labels(), "probabilities": [list(p) for p in rs.probs],
            "measuredSquares": list(rs.measured), "emptySquares": list(rs.empty),
            "quantumState": {board_string(b): {"real": a[0], "imaginary": a[1]} for b, a in rs.amps.items()},
            "pairs": [None if x is None else list(x) for x in rs.pairs],
            "squareToPair": {str(k): v for k, v in sorted(rs.s2p.items())},
            "settings": {"quantumness": rs.q, "decoherence": rs.dec, "timerEnabled": True,
                         "moveTime": timeout_ms / 1000},
            "legalMoves": [list(m) for m in legal_moves(rs)]}


def move_from_json(m) -> tuple:
    return tuple(m)


# ---- the search board -------------------------------------------------------------------------------------------
_RESULT = {UNFINISHED: None, X_WINS: "white", O_WINS: "black", BOTH: "draw", DRAW: "draw"}


def _forcer(prefix):
    taken = []

    def chooser(idx, options):
        k = len(taken)
        if k < len(prefix):
            sq, v = prefix[k]
            if sq != idx:
                raise ValueError(f"outcome prefix square {sq} != measured square {idx}")
            taken.append(prefix[k])
            return v
        raise NeedChoice(idx, options)
    return chooser


class KitTQTBoard(GameBoard):
    """``core.GameBoard`` on the kit referee's rules (X = "w", O = "b"; one board = one round)."""

    rules_variant = "tiqtaqtoe-kit"
    rules = "kit"

    def __init__(self, rs: RoundState | None = None, seed: int = 0):
        self.rs = rs if rs is not None else RoundState()
        self.history: list = []
        self.rng = np.random.default_rng(seed)
        self._kids: dict = {}

    @classmethod
    def new(cls, quantumness: int = 2, decoherence: int = 0, round_number: int = 1, seed: int = 0):
        return cls(RoundState(quantumness, decoherence, "X" if round_number % 2 == 1 else "O"), seed)

    @classmethod
    def from_payload(cls, payload: dict, counters=None, seed: int = 0):
        return cls(state_from_payload(payload, counters), seed)

    @property
    def turn(self) -> str:
        return WHITE if self.rs.turn == "X" else BLACK

    def copy(self) -> "KitTQTBoard":
        c = KitTQTBoard.__new__(KitTQTBoard)
        c.rs = self.rs.clone()
        c.history = list(self.history)
        c.rng = self.rng
        c._kids = {}
        return c

    def result(self) -> str | None:
        return _RESULT[self.rs.winner]

    def legal_moves(self, allow_quantum: bool = True) -> list:
        if self.rs.winner != UNFINISHED:
            return []
        ms = legal_moves(self.rs)
        if not allow_quantum:
            cl = [m for m in ms if len(m) == 1]
            return cl or ms
        return ms

    def _simulate(self, m, prefix=()) -> RoundState:
        rs = self.rs.clone()
        do_move(rs, m, _forcer(prefix))
        return rs

    def _outcomes(self, m) -> list:
        got = self._kids.get(m)
        if got is not None:
            return got
        outs, stack = [], [((), 1.0)]
        while stack:
            pre, p = stack.pop()
            try:
                rs = self._simulate(m, pre)
            except NeedChoice as nc:
                for v, q in reversed(nc.options):
                    stack.append((pre + ((nc.idx, v),), p * q))
                continue
            outs.append((pre, p, rs))
        if len(outs) == 1:
            outs = [(None, 1.0, outs[0][2])]
        else:
            tot = sum(p for _, p, _ in outs)
            outs = [(o, p / tot, rs) for o, p, rs in sorted(outs, key=lambda x: x[0])]
        self._kids[m] = outs
        return outs

    def outcome_probs(self, m) -> list:
        return [(o, p) for o, p, _ in self._outcomes(m)]

    def apply(self, m, outcome=None) -> "KitTQTBoard":
        rs = None
        for o, _, r in self._outcomes(m):
            if o == outcome or (outcome is not None and o is not None and tuple(map(tuple, outcome)) == o):
                rs = r
                break
        if rs is None:
            raise ValueError(f"no outcome {outcome!r} for move {m!r}")
        c = KitTQTBoard.__new__(KitTQTBoard)
        c.rs = rs.clone()
        c.history = self.history + [m]
        c.rng = self.rng
        c._kids = {}
        return c

    def apply_with(self, m, chooser) -> "KitTQTBoard":
        """Play ``m`` with measurement values from ``chooser(square, options) -> value`` (referee replays)."""
        rs = self.rs.clone()
        do_move(rs, m, chooser)
        c = KitTQTBoard(rs)
        c.history = self.history + [m]
        c.rng = self.rng
        return c

    def branches(self) -> list:
        return [(re * re + im * im, b) for b, (re, im) in self.rs.amps.items()]

    def summary(self) -> dict:
        return {"branches": [[board_string(b), round(p, 6)] for p, b in self.branches()],
                "measured": self.rs.measured, "winner": self.rs.winner, "turn": self.rs.turn}

    def payload(self, round_number: int = 1, timeout_ms: int = 5000) -> dict:
        return payload_from_state(self.rs, round_number, timeout_ms)

    def __repr__(self):
        return f"KitTQTBoard(q{self.rs.q}, {self.summary()})"


def make_kit_evaluator(**kw):
    """Anabi's TiqTaqToe evaluator (``opponents.make_evaluator('tqt:...')``): Born-averaged perfect classical
    tic-tac-toe over the branches."""
    from .core import BranchAveragedEvaluator
    from .tiqtaqtoe import classical_branch_value
    return BranchAveragedEvaluator(classical_branch_value, name="tqt-perfect-classical", **kw)


__all__ = ["KitTQTBoard", "RoundState", "do_move", "legal_moves", "is_legal", "state_from_payload",
           "payload_from_state", "make_kit_evaluator", "NeedChoice", "UNFINISHED", "X_WINS", "O_WINS", "BOTH",
           "DRAW"]
