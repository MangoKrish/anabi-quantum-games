"""Tie-break tables against a uniformly random opponent, for the TiqTaqToe arena bot (12 November 2026).

The exact tables (``tqt_kit_solver``) give each position's value against a perfect opponent.  Often several moves
share the best value ("optimal" moves, within ``TIE`` exactly as ``tqt_perfect.ExactPlayer`` decides it).  Picking
among those can never lose value against a perfect player; this module picks the one with the highest expected
round score against an opponent that plays uniformly at random over the legal moves (the arena's House bot).

Two numbers per state, both from the side to move (scores: win 1, draw or BOTH 0.5, loss 0):

* ``A(s)``: the mover is our bot (plays the optimal move with the best A/B score), the other side is random;
* ``B(s)``: the mover is random, the other side is our bot.

  A(s) = max over optimal moves m of  E_outcomes[ score(c) if c is over, A(c) if c's mover is us, else 1 - B(c) ]
  B(s) = mean over legal moves m of   E_outcomes[ score(c) if c is over, B(c) if c's mover is random, else 1 - A(c) ]

Keys are the exact tables' keys (mover's view, 8 symmetries, colour swap), so one solve from an X-first start
covers both colours and both round starts.  The optimal sets are computed from the loaded (float32) exact table
with the same arithmetic as the bot, so they are the bot's own optimal sets.  Only complete tables (every move of
every node stored) can be used.

  python -m qchess.games.tqt_rand --table tqt_tables/kit_q2_d0.npz --out tqt_tables/rand_q2_d0.npz

Output (``.npz``): sorted uint64 ``keys`` and float32 ``a``, ``b`` (NaN where that role never occurs), ``q``,
``dec`` and the root values ``a_root`` (we move first) / ``b_root`` (random moves first).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .tqt_kit import UNFINISHED, RoundState, legal_moves
from .tqt_kit_solver import VALUE, KitSolver, outcomes, state_key

TIE = 1e-6        # same as tqt_perfect.TIE


def term_score(c, mover: str) -> float:
    """Round score of ``mover`` in the finished state ``c`` (draw and BOTH 0.5)."""
    v = VALUE[c.winner]
    return (1.0 + (v if mover == "X" else -v)) / 2.0


class Missing(Exception):
    """A state the exact table does not have (the table is not complete for this state)."""


def look_x(t: KitSolver, rs):
    """Exact value from X's view (same as ``ExactPlayer._look``)."""
    if rs.winner != UNFINISHED:
        return VALUE[rs.winner]
    v = t.lookup(state_key(rs, t.sym))
    if v is None:
        return None
    return v if rs.turn == "X" else -v


def optimal_moves(t: KitSolver, rs, kids: dict | None = None):
    """(optimal moves in legal order, {move: outcomes}) from table lookups, as ``ExactPlayer.choose`` phase 1 does
    for one hypothesis.  Raises ``Missing`` if a child is not in the table."""
    moves = legal_moves(rs)
    if kids is None:
        kids = {m: outcomes(rs, m) for m in moves}
    sg = 1.0 if rs.turn == "X" else -1.0
    vals = {}
    for m in moves:
        tot = 0.0
        for p, c in kids[m]:
            v = look_x(t, c)
            if v is None:
                raise Missing()
            tot += p * v / 1
        vals[m] = tot
    best = max(sg * v for v in vals.values())
    return [m for m in moves if sg * vals[m] >= best - TIE], kids


class RandSolver:
    def __init__(self, table: KitSolver):
        self.t = table
        self.A: dict = {}
        self.B: dict = {}
        self.t0 = time.time()
        self.next_report = 250_000

    def _tick(self):
        n = len(self.A) + len(self.B)
        if n >= self.next_report:
            self.next_report += 250_000
            print(f"  A {len(self.A)} B {len(self.B)} {time.time() - self.t0:.0f} s", flush=True)

    def a(self, rs) -> float:
        k = state_key(rs, self.t.sym)
        v = self.A.get(k)
        if v is not None:
            return v
        me = rs.turn
        opt, kids = optimal_moves(self.t, rs)
        best = -1.0
        for m in opt:
            e = 0.0
            for p, c in kids[m]:
                if c.winner != UNFINISHED:
                    e += p * term_score(c, me)
                elif c.turn == me:
                    e += p * self.a(c)
                else:
                    e += p * (1.0 - self.b(c))
            best = max(best, e)
        self.A[k] = best
        self._tick()
        return best

    def b(self, rs) -> float:
        k = state_key(rs, self.t.sym)
        v = self.B.get(k)
        if v is not None:
            return v
        me = rs.turn
        moves = legal_moves(rs)
        tot = 0.0
        for m in moves:
            for p, c in outcomes(rs, m):
                if c.winner != UNFINISHED:
                    tot += p * term_score(c, me)
                elif c.turn == me:
                    tot += p * self.b(c)
                else:
                    tot += p * (1.0 - self.a(c))
        v = tot / len(moves)
        self.B[k] = v
        self._tick()
        return v

    def save(self, path: str, a_root: float, b_root: float) -> None:
        ks = np.array(sorted(set(self.A) | set(self.B)), dtype=np.uint64)
        a = np.array([self.A.get(int(k), math.nan) for k in ks], dtype=np.float32)
        b = np.array([self.B.get(int(k), math.nan) for k in ks], dtype=np.float32)
        tmp = path + ".tmp.npz"
        np.savez(tmp, keys=ks, a=a, b=b, q=self.t.q, dec=self.t.dec, a_root=a_root, b_root=b_root)
        os.replace(tmp, path)


class RandTable:
    """Loaded ``rand_qQ_dD.npz``: ``a(rs)`` / ``b(rs)`` or None when the state (or role) is not stored."""

    def __init__(self, path: str):
        z = np.load(path)
        self.q, self.dec = int(z["q"]), int(z["dec"])
        self.keys, self.av, self.bv = z["keys"], z["a"], z["b"]
        self.a_root, self.b_root = float(z["a_root"]), float(z["b_root"])
        self.sym = True

    def _get(self, arr, rs):
        k = state_key(rs, self.sym)
        i = int(np.searchsorted(self.keys, np.uint64(k)))
        if i < len(self.keys) and int(self.keys[i]) == k:
            v = float(arr[i])
            return None if math.isnan(v) else v
        return None

    def a(self, rs):
        return self._get(self.av, rs)

    def b(self, rs):
        return self._get(self.bv, rs)

    def after_our_move(self, c, me: str):
        """Our expected score after our move led to state ``c`` (None if not stored)."""
        if c.winner != UNFINISHED:
            return term_score(c, me)
        if c.turn == me:
            return self.a(c)
        v = self.b(c)
        return None if v is None else 1.0 - v

    def __len__(self):
        return len(self.keys)


def main(argv=None):
    import argparse
    import resource
    import sys
    sys.setrecursionlimit(20000)
    ap = argparse.ArgumentParser(description="tie-break table against a uniformly random opponent")
    ap.add_argument("--table", required=True, help="a COMPLETE exact table (kit_qQ_dD.npz)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = KitSolver.load(a.table)
    s = RandSolver(t)
    root = RoundState(t.q, t.dec, "X")
    ar = s.a(root)
    br = s.b(root)
    s.save(a.out, ar, br)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform.startswith("linux") else 2 ** 20)
    print(f"DONE rand q{t.q} dec{t.dec}: we first {ar:.6f}, we second {1 - br:.6f}, round mean {(ar + 1 - br) / 2:.6f}; "
          f"A {len(s.A)} B {len(s.B)} states, {time.time() - s.t0:.0f} s, maxrss {rss:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
