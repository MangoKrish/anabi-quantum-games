"""Cheqqers (quantum checkers): a port of the reference rules for Anabi.

Source: Marien Raat et al., "Quantum Checkers: The Development and Analysis of a Quantum Combinatorial Game",
IEEE CoG 2025, arXiv:2506.05962; code github.com/raatmarien/cheqqers, tag ``cog-paper-version`` (commit
a271182, AGPL-3.0-or-later), files ``game.py``, ``board.py``, ``quantum_state.py``, ``moves.py``, ``piece.py``.
This module re-implements those rules (it copies no code) so that Anabi can enumerate measurement outcomes
exactly; the reference ``Game`` stays the referee in matches against the reference agents.

What is reproduced, including the reference's quirks (all checked against the reference, see the report):

* Board: dark squares ``(x + y) % 2 == 0`` indexed row by row from y = 0; White on rows ``< start_rows`` moving
  to +y, Black on the top rows; White moves first.  Levels ``GameType``: 0 classic, 1 superposition, 2
  entanglement, 3 interference (merges).
* Moves in the reference's generation order: if any take exists, only takes (from any piece); otherwise
  plain moves, then splits (every pair of a piece's plain moves), then merges (two squares of one superposition,
  same crowned flag, onto a common target).  Level 0 keeps plain moves; levels 1-2 plain + splits; level 3 all.
* A take continues the mover's turn while ``get_take_moves`` has a take *from the landing square* and the
  piece did not land on the last row; the continuation may then be any take of the mover (as in the reference).
* Quantum state as in the reference: per piece superposition a move history, rebuilt into a circuit whenever a
  square is measured: X on the start qubit, ``moves_since_measure % 4`` S gates, plain moves relabel the qubit
  and add S, splits apply the quantum-chess split-jump matrix on (to1, to2, from), merges the merge-jump matrix
  on (from1, from2, to); an entangling take inserts CCX(taker-from, taken, taker-to), CX(to, taken),
  CX(to, from), S(to).  Merges leave the source squares in place (they become empty in the state, still
  marked quantum on the board), as in the reference.
* Takes: classical x quantum unentangled = entangle (levels 2, 3) or measure the taken square (level 1);
  classical x entangled = measure taken; quantum x classical = measure the taker first; quantum x quantum = taker
  then taken.  A measurement measures every qubit of the circuit; squares found empty are cleared, the
  superposition (and its entangled partner) is dropped; a failed take is a pass that keeps the turn rule.
* Draw when ``moves_since_take >= 40`` (plies), checked before "no legal move = loss".

Outcomes are enumerated exactly with a small state-vector simulator (complex128; the reference samples from
cirq's complex64 state, so probabilities agree to ~1e-7).  An outcome is the sequence of measurement results,
each = (squares found occupied, squares found empty, taken flag).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from itertools import product as _product

import numpy as np

from .core import BLACK, WHITE, GameBoard, forced_value, heuristic_value

CLASSIC, SUPERPOSITION, ENTANGLEMENT, INTERFERENCE = 0, 1, 2, 3
LEVEL_NAMES = {0: "classic", 1: "superposition", 2: "entanglement", 3: "interference"}
EMPTY, OCC, QUANT = 0, 1, 2
DRAW_PLIES = 40
PRUNE = 1e-12

_r2 = 1 / np.sqrt(2)
SPLIT_JUMP = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1j, 0, 0, 0],
    [0, 1j * _r2, _r2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -_r2, 1j * _r2, 0],
    [0, 1j * _r2, -_r2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1j * _r2, -_r2, 0],
    [0, 0, 0, 1j, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=complex)
MERGE_JUMP = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1j * _r2, 0, -1j * _r2, 0, 0, 0],
    [0, 0, _r2, 0, -_r2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1j, 0],
    [0, -1j, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -_r2, 0, -1j * _r2, 0, 0],
    [0, 0, 0, -1j * _r2, 0, -_r2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=complex)


# ---- moves: tuples, hashable -----------------------------------------------------------------------------
# ("m", take, from, to)   ("s", from, to1, to2)   ("g", from1, from2, to)      (g = merge)
def move_text(m) -> str:
    if m[0] == "m":
        return f"{m[2]}{'x' if m[1] else '-'}{m[3]}"
    if m[0] == "s":
        return f"{m[1]}^{m[2]},{m[3]}"
    return f"{m[1]},{m[2]}>{m[3]}"


# ---- superposition records -------------------------------------------------------------------------------
@dataclass
class Sup:
    sid: int
    occ: list                 # occupied_squares (order kept as in the reference)
    moves: list               # moves; None = entanglement placeholder
    msm: int                  # moves_since_measure at creation

    def clone(self):
        return Sup(self.sid, list(self.occ), list(self.moves), self.msm)


# ---- tiny state-vector simulator ---------------------------------------------------------------------------
class _Circ:
    def __init__(self):
        self.nq = 0
        self.ops = []

    def new(self) -> int:
        self.nq += 1
        return self.nq - 1

    def run(self) -> np.ndarray:
        psi = np.zeros(1 << self.nq, dtype=complex)
        psi[0] = 1.0
        n = self.nq
        for op in self.ops:
            kind, qs = op[0], op[1]
            t = psi.reshape([2] * n)
            if kind == "X":
                t = np.flip(t, axis=qs[0]).copy()
            elif kind == "S":
                sl = [slice(None)] * n
                sl[qs[0]] = 1
                t = t.copy()
                t[tuple(sl)] *= 1j
            elif kind in ("CX", "CCX"):
                *ctl, tg = qs
                t = t.copy()
                sl = [slice(None)] * n
                for c in ctl:
                    sl[c] = 1
                sub = t[tuple(sl)]
                ax = tg - sum(1 for c in ctl if c < tg)
                t[tuple(sl)] = np.flip(sub, axis=ax)
            elif kind == "M3":
                u = op[2]
                t = np.moveaxis(t, qs, [0, 1, 2]).reshape(8, -1)
                t = (u @ t).reshape([2, 2, 2] + [2] * (n - 3))
                t = np.moveaxis(t, [0, 1, 2], qs)
            psi = t.reshape(-1)
        return psi


@dataclass
class CheqqersBoard(GameBoard):
    size: int = 5
    start_rows: int = 1
    level: int = INTERFERENCE
    allow_draws: bool = True
    pieces: list = field(default_factory=list)        # (color, crowned, msm) or None
    occ: list = field(default_factory=list)            # EMPTY / OCC / QUANT
    turn: str = WHITE
    since_take: int = 0
    sups: list = field(default_factory=list)
    ents: list = field(default_factory=list)          # (taken_sid, from_sid)
    history: list = field(default_factory=list)
    next_sid: int = 0
    rng: object = None

    rules_variant = "cheqqers"

    # ---- setup ----
    @classmethod
    def new(cls, size: int = 5, start_rows: int = 1, level: int = INTERFERENCE, allow_draws: bool = True):
        b = cls(size=size, start_rows=start_rows, level=level, allow_draws=allow_draws)
        b._geometry()
        for i in range(b.nsq):
            x, y = b.ixy[i]
            if y < start_rows:
                b.pieces.append((WHITE, False, 0)); b.occ.append(OCC)
            elif y >= size - start_rows:
                b.pieces.append((BLACK, False, 0)); b.occ.append(OCC)
            else:
                b.pieces.append(None); b.occ.append(EMPTY)
        b.rng = np.random.default_rng(0)
        return b

    def _geometry(self):
        key = self.size
        g = _GEOM.get(key)
        if g is None:
            xyi, ixy = {}, {}
            i = 0
            for y in range(self.size):
                for x in range(self.size):
                    if (x + y) % 2 == 1:
                        continue
                    xyi[(x, y)] = i
                    ixy[i] = (x, y)
                    i += 1
            g = _GEOM[key] = (xyi, ixy, i)
        self.xyi, self.ixy, self.nsq = g

    def copy(self) -> "CheqqersBoard":
        c = CheqqersBoard.__new__(CheqqersBoard)
        c.size, c.start_rows, c.level, c.allow_draws = self.size, self.start_rows, self.level, self.allow_draws
        c.xyi, c.ixy, c.nsq = self.xyi, self.ixy, self.nsq
        c.pieces = list(self.pieces)
        c.occ = list(self.occ)
        c.turn = self.turn
        c.since_take = self.since_take
        c.sups = [s.clone() for s in self.sups]
        c.ents = list(self.ents)
        c.history = list(self.history)
        c.next_sid = self.next_sid
        c.rng = self.rng
        return c

    # ---- move generation (board.py) ----
    def _gen(self, color, take: bool):
        moves = []
        d = 1 if color == WHITE else -1
        for y in range(self.size):
            for x in range(self.size):
                i = self.xyi.get((x, y))
                if i is None or self.occ[i] == EMPTY or self.pieces[i][0] != color:
                    continue
                cand = [(x - 1, y + d), (x + 1, y + d)]
                if self.pieces[i][1]:
                    cand += [(x - 1, y - d), (x + 1, y - d)]
                for (x2, y2) in cand:
                    j = self.xyi.get((x2, y2))
                    if j is None:
                        continue
                    if take:
                        if self.occ[j] == EMPTY or self.pieces[j][0] == color:
                            continue
                        k = self.xyi.get((2 * x2 - x, 2 * y2 - y))
                        if k is None or self.occ[k] != EMPTY:
                            continue
                        moves.append(("m", True, i, k))
                    else:
                        if self.occ[j] != EMPTY:
                            continue
                        moves.append(("m", False, i, j))
        if not take:
            moves += self._split_merge(moves)
        return moves

    def _split_merge(self, moves):
        splits = []
        seen = set()
        for m in moves:
            f = m[2]
            if f in seen:
                continue
            rel = [r for r in moves if r[2] == f]
            if len(rel) < 2:
                continue
            seen.add(f)
            for a in range(len(rel)):
                for b in range(a + 1, len(rel)):
                    splits.append(("s", f, rel[a][3], rel[b][3]))
        merges = []
        seen_t = set()
        for m in moves:
            t = m[3]
            if t in seen_t:
                continue
            rel = [r for r in moves if r[3] == t]
            if len(rel) < 2:
                continue
            seen_t.add(t)
            for a in range(len(rel)):
                for b in range(a + 1, len(rel)):
                    f1, f2 = rel[a][2], rel[b][2]
                    if self._same_sup(f1, f2) and self.pieces[f1][1] == self.pieces[f2][1]:
                        merges.append(("g", f1, f2, t))
        # (the reference skips a source / target once it produced a split / merge; re-scanning one that
        # produced none yields none again, so marking it seen here gives the same list)
        return splits + merges

    def _same_sup(self, a, b) -> bool:
        return any(a in s.occ and b in s.occ for s in self.sups)

    def take_moves(self, color):
        return self._gen(color, True)

    def possible_moves(self, color):
        tk = self.take_moves(color)
        if tk:
            return tk
        ms = self._gen(color, False)
        if self.level == INTERFERENCE:
            return ms
        if self.level in (SUPERPOSITION, ENTANGLEMENT):
            return [m for m in ms if m[0] in ("m", "s")]
        return [m for m in ms if m[0] == "m"]

    def legal_moves(self, allow_quantum: bool = True):
        if self.result() is not None:
            return []
        ms = self.possible_moves(self.turn)
        if not allow_quantum:
            cl = [m for m in ms if m[0] == "m"]
            return cl or ms
        return ms

    def result(self):
        if self.allow_draws and self.since_take >= DRAW_PLIES:
            return "draw"
        if not self.possible_moves(self.turn):
            return "white" if self.turn == BLACK else "black"
        return None

    # ---- superposition lookup ----
    def _sup_on(self, sq):
        for s in self.sups:
            if sq in s.occ:
                return s
        return None

    def _sup_by_id(self, sid):
        for s in self.sups:
            if s.sid == sid:
                return s
        raise KeyError(sid)

    def _ent_of(self, sup):
        if sup is None:
            return None
        for e in self.ents:
            if self._sup_eq(e[0], sup) or self._sup_eq(e[1], sup):
                return e
        return None

    def _sup_eq(self, sid, sup) -> bool:
        # the reference compares superpositions by their occupied_squares lists (PieceSuperposition.__eq__)
        return self._sup_by_id(sid).occ == sup.occ

    def _is_entangled(self, sq) -> bool:
        return self._ent_of(self._sup_on(sq)) is not None

    def _crown(self, piece, to):
        if not piece[1]:
            y = self.ixy[to][1]
            if (piece[0] == WHITE and y == self.size - 1) or (piece[0] == BLACK and y == 0):
                return (piece[0], True, piece[2])
        return piece

    def _new_sup(self, occ, move, msm) -> Sup:
        s = Sup(self.next_sid, occ, [move], msm)
        self.next_sid += 1
        self.sups.append(s)
        return s

    # ---- circuit of a square (game._get_circuit_for_square) ----
    def _circuit(self, sq):
        sup = self._sup_on(sq)
        sup_from = None
        ent = self._ent_of(sup)
        if ent is not None:
            sup = self._sup_by_id(ent[0])
            sup_from = self._sup_by_id(ent[1])
        c = _Circ()
        P, T = "c", "t"
        q = {}
        start = self._from_of(sup.moves[0])
        q[(P, start)] = c.new()
        c.ops.append(("X", (q[(P, start)],)))
        for _ in range(sup.msm % 4):
            c.ops.append(("S", (q[(P, start)],)))

        def handle(m, pre):
            if m[0] == "m" and not m[1]:
                qb = q.pop((pre, m[2]))
                q[(pre, m[3])] = qb
                c.ops.append(("S", (qb,)))
            elif m[0] == "s":
                qf = q.pop((pre, m[1]))
                q[(pre, m[2])] = c.new()
                q[(pre, m[3])] = c.new()
                c.ops.append(("M3", (q[(pre, m[2])], q[(pre, m[3])], qf), SPLIT_JUMP))
            elif m[0] == "g":
                q[(pre, m[3])] = c.new()
                c.ops.append(("M3", (q[(pre, m[1])], q[(pre, m[2])], q[(pre, m[3])]), MERGE_JUMP))

        for m in sup.moves:
            if m is None:
                tm = sup_from.moves[0]
                f, t = tm[2], tm[3]
                q[(T, f)] = c.new()
                c.ops.append(("X", (q[(T, f)],)))
                for _ in range(sup_from.msm % 4):
                    c.ops.append(("S", (q[(T, f)],)))
                q[(T, t)] = c.new()
                tx, ty = self.ixy[f]
                ux, uy = self.ixy[t]
                taken = self.xyi[((tx + ux) // 2, (ty + uy) // 2)]
                c.ops.append(("CCX", (q[(T, f)], q[(P, taken)], q[(T, t)])))
                c.ops.append(("CX", (q[(T, t)], q[(P, taken)])))
                c.ops.append(("CX", (q[(T, t)], q[(T, f)])))
                c.ops.append(("S", (q[(T, t)],)))
                del q[(P, taken)]
            else:
                handle(m, P)
        if sup_from is not None:
            for m in sup_from.moves[1:]:
                handle(m, T)
        return c, q, ent, sup, sup_from

    @staticmethod
    def _from_of(m):
        return m[2] if m[0] == "m" else m[1]

    def _measure_dist(self, sq):
        """Exact distribution of game.measure(sq): [(found_tuple, empty_tuple, taken, p)] grouped."""
        c, q, ent, sup, sup_from = self._circuit(sq)
        psi = c.run()
        probs = np.abs(psi) ** 2
        items = list(q.items())
        out: dict = {}
        n = c.nq
        for idx in np.nonzero(probs > PRUNE)[0]:
            found, s1 = {}, 0
            for (pre, s), qb in items:
                bit = (int(idx) >> (n - 1 - qb)) & 1
                if bit:
                    s1 += 1
                    found[s] = True
                elif s not in found:
                    found[s] = False
            taken = s1 == 1 and ent is not None
            key = (tuple(sorted(k for k, v in found.items() if v)), tuple(sorted(k for k, v in found.items() if not v)),
                   taken)
            out[key] = out.get(key, 0.0) + float(probs[idx])
        tot = sum(out.values())
        return [(k, p / tot) for k, p in sorted(out.items())], (ent, sup, sup_from)

    def _apply_measure(self, sq, oracle):
        """game.measure(sq): a square that is not quantum is read classically (no draw); otherwise the result
        ``(found, empty, taken)`` comes from ``oracle(sq)``.  Returns (is_there, taken)."""
        if self.occ[sq] != QUANT:
            return self.occ[sq] == OCC, False
        result = oracle(sq)
        _, (ent, sup, sup_from) = self._measure_dist_meta(sq)
        found, empty, taken = result
        for s in found:
            self.occ[s] = OCC
        for s in empty:
            self.occ[s] = EMPTY
            self.pieces[s] = None
        self._remove_sup(sup)
        if ent is not None:
            self.ents.remove(ent)
            self._remove_sup(sup_from)
        return sq in found, taken

    def _measure_dist_meta(self, sq):
        sup = self._sup_on(sq)
        sup_from = None
        ent = self._ent_of(sup)
        if ent is not None:
            sup = self._sup_by_id(ent[0])
            sup_from = self._sup_by_id(ent[1])
        return None, (ent, sup, sup_from)

    def _remove_sup(self, sup):
        for k, s in enumerate(self.sups):                 # list.remove: the first equal one
            if s.occ == sup.occ:
                del self.sups[k]
                return
        raise ValueError("superposition not found")

    # ---- applying moves (game.apply_move) with an outcome oracle ----
    def _run(self, m, oracle):
        """Mutates self.  ``oracle(sq) -> result`` resolves each measurement."""
        self.since_take += 1
        canceled = taken = False
        if m[0] == "m":
            if m[1]:
                canceled, taken = self._take(m, oracle)
            else:
                self._plain(m)
        elif m[0] == "s":
            self._split(m)
        else:
            self._merge(m)
        self.history.append(m)
        if taken or (not canceled and m[0] == "m" and m[1]):
            self.since_take = 0
        last_row = 0 if self.turn == BLACK else self.size - 1
        is_take = m[0] == "m" and m[1]
        if (not is_take or not any(t[2] == m[3] for t in self.take_moves(self.turn))
                or self.ixy[m[3]][1] == last_row):
            self.turn = BLACK if self.turn == WHITE else WHITE

    def _plain(self, m):
        f, t = m[2], m[3]
        piece = self.pieces[f]
        st = self.occ[f]
        if st == QUANT:
            sup = self._sup_on(f)
            sup.moves.append(m)
            sup.occ.remove(f)
            sup.occ.append(t)
        else:
            piece = (piece[0], piece[1], piece[2] + 1)
        piece = self._crown(piece, t)
        self.pieces[t] = piece
        self.occ[t] = st
        self.pieces[f] = None
        self.occ[f] = EMPTY

    def _take(self, m, oracle):
        f, t = m[2], m[3]
        piece = self.pieces[f]
        fs = self.occ[f]
        fx, fy = self.ixy[f]
        tx, ty = self.ixy[t]
        k = self.xyi[((fx + tx) // 2, (fy + ty) // 2)]
        ks = self.occ[k]
        piece = self._crown(piece, t)
        if fs == OCC and ks == QUANT and not self._is_entangled(k):
            if self.level in (ENTANGLEMENT, INTERFERENCE):
                self.occ[f] = QUANT
                self.occ[t] = QUANT
                self.pieces[t] = piece
                staken = self._sup_on(k)
                staken.moves.append(None)
                sfrom = self._new_sup([f, t], m, piece[2])
                self.ents.append((staken.sid, sfrom.sid))
                self.occ[k] = EMPTY
                self.pieces[k] = None
                staken.occ.remove(k)
                return True, False
            there, tk = self._apply_measure(k, oracle)
            if not there:
                return True, tk
        elif fs == OCC and ks == QUANT:
            there, tk = self._apply_measure(k, oracle)
            if not there:
                return True, tk
        elif fs == QUANT and ks == OCC:
            there, tk = self._apply_measure(f, oracle)
            if not there:
                return True, tk
        elif fs == QUANT and ks == QUANT:
            there, t1 = self._apply_measure(f, oracle)
            if not there:
                return True, t1
            there2, t2 = self._apply_measure(k, oracle)
            if not there2:
                return True, t1 or t2
        piece = (piece[0], piece[1], piece[2] + 1)
        self.pieces[k] = None
        self.occ[k] = EMPTY
        self.pieces[t] = piece
        self.occ[t] = OCC
        self.pieces[f] = None
        self.occ[f] = EMPTY
        return False, True

    def _split(self, m):
        f, t1, t2 = m[1], m[2], m[3]
        piece = self.pieces[f]
        if self.occ[f] == QUANT:
            sup = self._sup_on(f)
            sup.moves.append(m)
            sup.occ.remove(f)
            sup.occ += [t1, t2]
        else:
            self._new_sup([t1, t2], m, piece[2])
        self.occ[f] = EMPTY
        self.pieces[f] = None
        piece = self._crown(piece, t1)
        for i in (t1, t2):
            self.occ[i] = QUANT
            self.pieces[i] = piece

    def _merge(self, m):
        f1, f2, t = m[1], m[2], m[3]
        piece = self.pieces[f1]
        sup = self._sup_on(f1)
        sup.moves.append(m)
        sup.occ.append(t)
        piece = self._crown(piece, t)
        self.occ[t] = QUANT
        self.pieces[t] = piece

    # ---- chance interface ----
    def children(self, m):
        """Exact outcomes of ``m``: [(outcome_key, p, board)] (outcome_key = tuple of measurement results)."""
        out = []

        def rec(prefix, p):
            b = self.copy()
            k = [0]
            need = []

            def oracle(sq):
                if k[0] < len(prefix):
                    r = prefix[k[0]]
                    k[0] += 1
                    return r
                dist, _ = b._measure_dist(sq)
                need.append(dist)
                raise _Branch

            try:
                b._run(m, oracle)
            except _Branch:
                for r, q in need[0]:
                    rec(prefix + (r,), p * q)
                return
            out.append((prefix if prefix else None, p, b))

        rec((), 1.0)
        return out

    def outcome_probs(self, m):
        return [(o, p) for o, p, _ in self.children(m)]

    def apply(self, m, outcome=None):
        b = self.copy()
        res = list(outcome or ())

        def oracle(sq):
            if not res:
                raise ValueError(f"move {m} needs a measurement outcome")
            return res.pop(0)

        b._run(m, oracle)
        return b

    # ---- branches for the evaluator ----
    def groups(self):
        """Independent measurement groups: [(squares, [(found, p)])] - one per unentangled superposition and one
        per entangled pair."""
        done = set()
        out = []
        for s in self.sups:
            if s.sid in done:
                continue
            sq = next((x for x in s.occ if self.occ[x] == QUANT), None)
            if sq is None:
                done.add(s.sid)
                continue
            dist, (ent, sup, sup_from) = self._measure_dist(sq)
            done.add(sup.sid)
            if sup_from is not None:
                done.add(sup_from.sid)
            out.append([(set(f), set(e), p) for (f, e, _t), p in dist])
        return out

    def branches(self):
        """Classical boards in the superposition: [(p, key)] with key = tuple of (color, crowned) or None per
        square.  Exact product over groups (callers sample when there are too many)."""
        gs = self.groups()
        base = [None if self.occ[i] == EMPTY else (self.pieces[i][0], self.pieces[i][1]) for i in range(self.nsq)]
        if not gs:
            return [(1.0, tuple(base))]
        size = 1
        for g in gs:
            size *= len(g)
        if size > 4096:                              # sample the product with a key-seeded generator
            rng = np.random.default_rng(zlib.crc32(repr((base, self.occ, self.turn)).encode()))
            combos = []
            for _ in range(64):
                combos.append(tuple(int(rng.choice(len(g), p=[x[2] for x in g])) for g in gs))
            from collections import Counter
            cnt = Counter(combos)
            items = [(c / 64.0, idxs) for idxs, c in cnt.items()]
        else:
            items = []
            for idxs in _product(*[range(len(g)) for g in gs]):
                p = 1.0
                for g, i in zip(gs, idxs):
                    p *= g[i][2]
                if p > PRUNE:
                    items.append((p, idxs))
        out = []
        for p, idxs in items:
            b = list(base)
            for g, i in zip(gs, idxs):
                f, e, _ = g[i]
                for s in e:
                    b[s] = None
            out.append((p, tuple(b)))
        return out

    # ---- misc ----
    def chances(self) -> dict:
        """Probability of every quantum square (game.get_all_chances)."""
        out = {}
        for i, st in enumerate(self.occ):
            if st == QUANT and i not in out:
                c, q, *_ = self._circuit(i)
                psi = c.run()
                pr = np.abs(psi) ** 2
                n = c.nq
                for (pre, s), qb in q.items():
                    p1 = float(sum(pr[j] for j in range(len(pr)) if (j >> (n - 1 - qb)) & 1))
                    out[s] = p1                     # dict(zip(squares, ev_list)): the last qubit of a square wins
        return out

    def signature(self) -> dict:
        return {"turn": self.turn, "occ": list(self.occ), "since_take": self.since_take,
                "pieces": [None if p is None else [p[0], bool(p[1])] for p in self.pieces]}

    def summary(self) -> dict:
        s = self.signature()
        s["chances"] = {int(k): round(v, 6) for k, v in self.chances().items()}
        return s

    def display(self) -> str:
        rows = []
        for y in range(self.size - 1, -1, -1):
            r = f"{y} "
            for x in range(self.size):
                i = self.xyi.get((x, y))
                if i is None:
                    r += "  "
                elif self.occ[i] == EMPTY:
                    r += "- "
                else:
                    c, k = self.pieces[i][0], self.pieces[i][1]
                    ch = ("W" if k else "w") if c == WHITE else ("B" if k else "b")
                    r += (ch if self.occ[i] == OCC else ch + "?")[:2].ljust(2)
            rows.append(r)
        return "\n".join(rows)


class _Branch(Exception):
    pass


_GEOM: dict = {}


# ---- classical checkers engine per branch (the "Fairy-Stockfish" of this game) -----------------------------
MAN, KING = 1.0, 1.5
ADV = 0.05


class ClassicalCheckers:
    """Alpha-beta on a classical board (the reference's move rules incl. forced takes and the continuation
    rule), depth ``depth`` plies with takes searched until quiet (bounded by ``qdepth``).  Score = material
    (man 1, king 1.5) + 0.05 per row a man has advanced, from the side to move; no move = loss."""

    def __init__(self, size: int, depth: int = 2, qdepth: int = 6, scale: float = 3.0):
        self.size, self.depth, self.qdepth, self.scale = size, depth, qdepth, scale
        b = CheqqersBoard.new(size, 0, CLASSIC)
        self.xyi, self.ixy, self.nsq = b.xyi, b.ixy, b.nsq
        self.cache: dict = {}
        self.nodes = 0

    def moves(self, sq, color):
        """Classical moves of ``color`` on a tuple board (None / (color, crowned)): takes if any, else plain."""
        size, xyi = self.size, self.xyi
        takes, plain = [], []
        d = 1 if color == WHITE else -1
        for i in range(self.nsq):
            p = sq[i]
            if p is None or p[0] != color:
                continue
            x, y = self.ixy[i]
            cand = ((x - 1, y + d), (x + 1, y + d)) + (((x - 1, y - d), (x + 1, y - d)) if p[1] else ())
            for x2, y2 in cand:
                j = xyi.get((x2, y2))
                if j is None:
                    continue
                q = sq[j]
                if q is None:
                    plain.append((i, j, None))
                elif q[0] != color:
                    k = xyi.get((2 * x2 - x, 2 * y2 - y))
                    if k is not None and sq[k] is None:
                        takes.append((i, k, j))
        return takes if takes else plain

    def play(self, sq, color, mv):
        f, t, k = mv
        b = list(sq)
        p = b[f]
        b[f] = None
        if k is not None:
            b[k] = None
        y = self.ixy[t][1]
        last = self.size - 1 if color == WHITE else 0
        if not p[1] and y == last:
            p = (p[0], True)
        b[t] = p
        b = tuple(b)
        nxt = BLACK if color == WHITE else WHITE
        if k is not None and y != last and any(m[0] == t for m in self.moves(b, color) if m[2] is not None):
            nxt = color                       # the mover continues (any take, as the reference allows)
        return b, nxt

    def static(self, sq, color):
        s = 0.0
        for i, p in enumerate(sq):
            if p is None:
                continue
            y = self.ixy[i][1]
            v = KING if p[1] else MAN + ADV * (y if p[0] == WHITE else self.size - 1 - y)
            s += v if p[0] == color else -v
        return s

    def search(self, sq, color, depth, alpha, beta, ply):
        """Negamax value (from ``color``'s view) and plies; returns float score or ('forced', win, plies)."""
        self.nodes += 1
        ms = self.moves(sq, color)
        if not ms:
            return -1000.0 + ply
        is_take = ms[0][2] is not None
        if depth <= 0 and (not is_take or depth <= -self.qdepth):
            return self.static(sq, color)
        best = -math.inf
        for mv in ms:
            nb, nxt = self.play(sq, color, mv)
            if nxt == color:
                v = self.search(nb, color, depth - (0 if is_take else 1), alpha, beta, ply + 1)
            else:
                v = -self.search(nb, nxt, depth - 1, -beta, -alpha, ply + 1)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def value(self, sq: tuple, color: str) -> float:
        key = (sq, color)
        v = self.cache.get(key)
        if v is None:
            v = self.search(sq, color, self.depth, -math.inf, math.inf, 0)
            if len(self.cache) > 200000:
                self.cache.clear()
            self.cache[key] = v
        return v

    def branch_value(self, board: CheqqersBoard, key: tuple, me: str) -> float:
        v = self.value(key, board.turn)
        sgn = 1.0 if board.turn == me else -1.0
        if abs(v) > 500:                      # forced: no move within the horizon
            plies = int(1000 - abs(v))
            return forced_value((v > 0) == (board.turn == me), plies)
        return sgn * heuristic_value(v, self.scale)


__all__ = ["CheqqersBoard", "ClassicalCheckers", "move_text", "LEVEL_NAMES", "CLASSIC", "SUPERPOSITION",
           "ENTANGLEMENT", "INTERFERENCE"]
