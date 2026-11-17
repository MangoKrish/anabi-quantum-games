"""Full anti-random tables for the TiqTaqToe arena bot's opt-in ``--anti-random`` mode (13 November 2026).

``tqt_rand`` (day 2) breaks ties among the moves that are optimal against a perfect opponent by the expected
round score against a uniformly random opponent.  This module drops the "optimal only" restriction: it is the
full expectimax against a uniformly random opponent over ALL legal moves (the day-2 upper bound,
``reports/engines/arena_20261112/day2/upper_bound_vs_random.py``, now stored).  The bot uses it only after the
opponent has shown it is not perfect (see ``tqt_tournament_bot.DeviationTracker``).

  A*(s) = max over legal moves m of  E_outcomes[ score(c) if c is over, else 1 - B*(c) ]
  B*(s) = mean over legal moves m of E_outcomes[ score(c) if c is over, else 1 - A*(c) ]

(the kit rules always pass the turn, so a child is never the same mover's state again; the ``tqt_rand`` code that
handles that case is kept).  No exact table is needed: the values come from the game tree alone.  Keys, file
format and the loader are the ones of ``tqt_rand`` (``RandTable``); files are named ``arand_qQ_dD.npz`` so the
day-2 loader (``rand_q*_d*.npz``) never picks them up.

  python -m qchess.games.tqt_arand --q 2 --dec 0 --out <dir>/arand_q2_d0.npz
"""
from __future__ import annotations

import time

from .tqt_kit import UNFINISHED, RoundState, legal_moves
from .tqt_kit_solver import outcomes, state_key
from .tqt_rand import RandSolver, RandTable, term_score


class _NoTable:
    """Stands in for the exact table: only ``q``, ``dec`` and ``sym`` are used by the unrestricted solve."""

    def __init__(self, q: int, dec: int):
        self.q, self.dec, self.sym = int(q), int(dec), True


class ArandSolver(RandSolver):
    def __init__(self, q: int, dec: int):
        super().__init__(_NoTable(q, dec))

    def a(self, rs) -> float:
        k = state_key(rs, self.t.sym)
        v = self.A.get(k)
        if v is not None:
            return v
        me = rs.turn
        best = -1.0
        for m in legal_moves(rs):
            e = 0.0
            for p, c in outcomes(rs, m):
                if c.winner != UNFINISHED:
                    e += p * term_score(c, me)
                elif c.turn == me:
                    e += p * self.a(c)
                else:
                    e += p * (1.0 - self.b(c))
            if e > best:
                best = e
        self.A[k] = best
        self._tick()
        return best


ArandTable = RandTable


def main(argv=None):
    import argparse
    import resource
    import sys
    sys.setrecursionlimit(20000)
    ap = argparse.ArgumentParser(description="full anti-random table (all legal moves) for --anti-random")
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--dec", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = ArandSolver(a.q, a.dec)
    root = RoundState(a.q, a.dec, "X")
    ar = s.a(root)
    br = s.b(root)
    s.save(a.out, ar, br)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform.startswith("linux") else 2 ** 20)
    print(f"DONE arand q{a.q} dec{a.dec}: we first {ar:.6f}, we second {1 - br:.6f}, round mean {(ar + 1 - br) / 2:.6f}; "
          f"A {len(s.A)} B {len(s.B)} states, {time.time() - s.t0:.0f} s, maxrss {rss:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
