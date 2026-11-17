"""Exact expectiminimax on the kit referee's own rules (``tqt_kit``), for the TiqTaqToe arena (11 November 2026).

Values are from X's view: +1 X wins the round, -1 O wins, 0 for a draw and for BOTH (each player gets a point, so
the round leaves the score difference unchanged; the arena leaderboard counts it as a draw).  Chance nodes are the
referee's own measurement outcomes with its own probabilities (``tqt_kit`` quirks included: dropped amplitudes,
the last value taking the leftover mass).

Levels 0-3 always end (every placing move adds one token to every branch, at most 9; between placing moves every
measuring move resolves an unmeasured square), so the full tree can be solved.  Level 4 has phase moves that place
nothing and can cycle; it is only searched to a node cap (``solve_limited``), never fully.

The state key is the referee's whole state up to labels: side to move, measured squares, decoherence counters
(when on) and the amplitudes rounded to 1e-9.  Entanglement groups and probabilities are functions of the
amplitudes; group slot numbers only ever enter by equality, so they are not part of the key.  Keys are 8-byte
blake2b digests (stable across machines and Python versions).

Tables: ``KitSolver.save(path)`` writes sorted uint64 keys + float32 values (``.npz``); ``load`` reads them into
a lookup that uses ``np.searchsorted`` (12 bytes per state).
"""
from __future__ import annotations

import hashlib
import struct
import time

import numpy as np

from .tqt_kit import (BOTH, DRAW, O_WINS, UNFINISHED, X_WINS, NeedChoice, RoundState, _forcer, do_move,
                      legal_moves)

VALUE = {X_WINS: 1.0, O_WINS: -1.0, BOTH: 0.0, DRAW: 0.0}


def _r(x: float) -> float:
    return round(x, 9) + 0.0


# the 8 symmetries of the square as permutations p: new square of old square s is p[s]
def _syms():
    rot = (2, 5, 8, 1, 4, 7, 0, 3, 6)          # quarter turn: old s -> new rot[s]
    ref = (2, 1, 0, 5, 4, 3, 8, 7, 6)
    out, r = set(), tuple(range(9))
    for _ in range(4):
        out.add(r)
        out.add(tuple(ref[r[s]] for s in range(9)))
        r = tuple(rot[r[s]] for s in range(9))
    return sorted(out)


SYMS = _syms()
INV = [tuple(sorted(range(9), key=lambda s: p[s])) for p in SYMS]   # inv[new] = old
POW3 = [3 ** (8 - i) for i in range(9)]
PROB_KEY = True
SWAPC = (0, 2, 1)


def state_key(rs: RoundState, sym: bool = True) -> int:
    """Key of the state seen from the side to move (colours swapped when O moves), minimised over the 8
    symmetries of the square when ``sym``.  Values are stored from the mover's view (see ``KitSolver``)."""
    sw = rs.turn == "O"
    if rs.q <= 3 and PROB_KEY:
        # levels 0-3: no two branches ever meet in a gate (every SqrtSwap pairs a freshly placed token of the mover
        # with a square that never holds the mover's token), so only |a|^2 can matter: key on probabilities
        amps = [(tuple(SWAPC[v] for v in b) if sw else b, round(re * re + im * im, 12) + 0.0, 0.0)
                for b, (re, im) in rs.amps.items()]
    else:
        amps = [(tuple(SWAPC[v] for v in b) if sw else b, _r(re), _r(im)) for b, (re, im) in rs.amps.items()]
    best = None
    for inv in (INV if sym else [tuple(range(9))]):
        meas = 0
        for n in range(9):
            if rs.measured[inv[n]]:
                meas |= 1 << n
        parts = [struct.pack("<H", meas)]
        if rs.dec > 0:
            parts.append(bytes(min(255, rs.counters[inv[n]]) for n in range(9)))
        br = []
        for b, re, im in amps:
            idx = 0
            for n in range(9):
                idx += b[inv[n]] * POW3[n]
            br.append((idx, re, im))
        br.sort()
        parts += [struct.pack("<Hdd", *x) for x in br]
        k = b"".join(parts)
        if best is None or k < best:
            best = k
    return int.from_bytes(hashlib.blake2b(best, digest_size=8).digest(), "little")


def outcomes(rs: RoundState, m) -> list:
    """[(probability, child RoundState)] of move ``m``: every measurement outcome the referee can draw."""
    outs, stack = [], [((), 1.0)]
    while stack:
        pre, p = stack.pop()
        c = rs.clone()
        try:
            do_move(c, m, _forcer(pre))
        except NeedChoice as nc:
            for v, q in nc.options:
                stack.append((pre + ((nc.idx, v),), p * q))
            continue
        outs.append((p, c))
    tot = sum(p for p, _ in outs)
    return [(p / tot, c) for p, c in outs]


class Budget(Exception):
    pass


class KitSolver:
    """Memoised exact values.  ``tt``: key -> value (a dict while solving; a sorted array after ``load``)."""

    def __init__(self, q: int, dec: int = 0, sym: bool = True):
        self.q, self.dec, self.sym = int(q), int(dec), bool(sym)
        self.shuffle = None         # random.Random: search moves in a random order (parallel partial solves)
        self.cut = True             # stop at a sure win (smaller table; other moves of that node not stored)
        self.tt: dict = {}
        self.keys = None            # loaded table: sorted uint64
        self.vals = None
        self.nodes = 0
        self.deadline = None
        self.node_cap = None
        self.progress = None

    # ---- lookup
    def lookup(self, k: int):
        v = self.tt.get(k)
        if v is not None:
            return v
        if self.keys is not None:
            i = int(np.searchsorted(self.keys, np.uint64(k)))
            if i < len(self.keys) and int(self.keys[i]) == k:
                return float(self.vals[i])
        return None

    def __len__(self):
        return len(self.tt) + (0 if self.keys is None else len(self.keys))

    # ---- solving
    def value(self, rs: RoundState) -> float:
        w = rs.winner
        if w != UNFINISHED:
            return VALUE[w]
        k = state_key(rs, self.sym)
        sg = 1.0 if rs.turn == "X" else -1.0
        v = self.lookup(k)
        if v is not None:
            return sg * v
        self.nodes += 1
        if self.node_cap is not None and (self.nodes > self.node_cap or
                                          ((self.nodes & 63) == 0 and time.perf_counter() > self.deadline)):
            raise Budget()
        ms = legal_moves(rs)
        if self.shuffle is not None and len(ms) > 1:
            ms = list(ms)
            self.shuffle.shuffle(ms)      # search order only; values do not depend on it
        if not ms:                      # never happens on the referee's rules (an unfinished round has a move)
            v = 0.0
        else:
            x = rs.turn == "X"
            best = -2.0 if x else 2.0
            for m in ms:
                mv = self.move_value(rs, m)
                if x:
                    if mv > best:
                        best = mv
                        if best >= 1.0 and self.cut:
                            break
                elif mv < best:
                    best = mv
                    if best <= -1.0 and self.cut:
                        break
            v = best
        self.tt[k] = sg * v
        if self.progress is not None and len(self.tt) % 250000 == 0:
            self.progress(len(self.tt))
        return v

    def move_value(self, rs: RoundState, m) -> float:
        return sum(p * self.value(c) for p, c in outcomes(rs, m))

    def move_values(self, rs: RoundState) -> dict:
        return {m: self.move_value(rs, m) for m in legal_moves(rs)}

    def solve_limited(self, rs: RoundState, seconds: float, node_cap: int = 2_000_000):
        """Exact move values from ``rs`` if the remaining tree fits in ``seconds`` / ``node_cap`` new nodes, else
        None.  Values found before the budget ran out stay in the table (they are exact)."""
        self.nodes, self.node_cap, self.deadline = 0, node_cap, time.perf_counter() + seconds
        try:
            return self.move_values(rs)
        except (Budget, RecursionError):
            return None
        finally:
            self.node_cap = self.deadline = None

    def solve_all(self, report_every: float = 60.0, log=print) -> float:
        """Solve both round starts (X first and O first)."""
        t0 = time.time()
        vals = []
        for first in ("X", "O"):
            rs = RoundState(self.q, self.dec, first)
            vals.append(self.value(rs))
            log(f"q{self.q} dec{self.dec} start {first}: value {vals[-1]:+.6f}, {len(self.tt)} states, "
                f"{time.time() - t0:.0f} s")
        return vals

    # ---- tables
    def save(self, path: str) -> None:
        from . import tqt_kit
        tqt_kit._DERIVE.clear()                 # free the rules' caches first (memory-capped workers)
        tqt_kit._NZ.clear()
        n = len(self.tt)
        ks = np.fromiter(self.tt.keys(), dtype=np.uint64, count=n)
        vs = np.fromiter(self.tt.values(), dtype=np.float32, count=n)
        if self.keys is not None:
            ks, vs = np.concatenate([ks, self.keys]), np.concatenate([vs, self.vals])
        o = np.argsort(ks, kind="stable")
        ks, vs = ks[o], vs[o]
        tmp = path + ".tmp.npz"
        np.savez(tmp, keys=ks, vals=vs, q=self.q, dec=self.dec)
        import os
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "KitSolver":
        z = np.load(path)
        s = cls(int(z["q"]), int(z["dec"]))
        s.keys, s.vals = z["keys"], z["vals"]
        return s


def main(argv=None):
    import argparse
    import resource
    import sys
    sys.setrecursionlimit(10000)
    ap = argparse.ArgumentParser(description="solve a kit TiqTaqToe setting exactly and save its table")
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--dec", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-states", type=int, default=0, help="stop and save the (exact) partial table here")
    ap.add_argument("--max-rss-mb", type=int, default=0, help="stop and save the partial table at this RSS")
    ap.add_argument("--shuffle", type=int, default=None, help="random move order with this seed")
    ap.add_argument("--checkpoint-min", type=float, default=0, help="also save the partial table this often")
    ap.add_argument("--no-cut", action="store_true", help="store every move's subtree (complete table)")
    a = ap.parse_args(argv)
    s = KitSolver(a.q, a.dec)
    s.cut = not a.no_cut
    if a.shuffle is not None:
        import random
        s.shuffle = random.Random(a.shuffle)
    last_ck = [time.time()]
    t0 = time.time()

    class Full(Exception):
        pass

    def prog(n):
        print(f"  {n} states {time.time() - t0:.0f} s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}",
              flush=True)
        if a.checkpoint_min and time.time() - last_ck[0] > 60 * a.checkpoint_min:
            last_ck[0] = time.time()
            s.save(a.out.replace(".npz", "_partial.npz"))
        if n % 5_000_000 == 0:
            s.save(a.out.replace(".npz", "_partial.npz"))     # checkpoint (every value in it is exact)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / 1024 if sys.platform.startswith("linux") else rss / 2 ** 20
        if (a.max_states and n >= a.max_states) or (a.max_rss_mb and rss_mb >= a.max_rss_mb):
            raise Full()
    s.progress = prog
    try:
        v = s.solve_all()
    except Full:
        s.save(a.out.replace(".npz", "_partial.npz"))
        print(f"PARTIAL q{a.q} dec{a.dec} states {len(s.tt)} time {time.time() - t0:.0f} s", flush=True)
        return
    s.save(a.out)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"DONE q{a.q} dec{a.dec} values X-first {v[0]:+.6f} O-first {v[1]:+.6f} states {len(s.tt)} "
          f"time {time.time() - t0:.0f} s maxrss {rss}", flush=True)


if __name__ == "__main__":
    main()
