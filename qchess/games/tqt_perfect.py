"""Exact play for the TiqTaqToe arena bot (11 November 2026): precomputed kit tables + a capped endgame search.

``ExactPlayer.choose(states, deadline)`` returns ``(move, info)`` or ``(None, info)``:

* ``states`` is a list of ``RoundState`` hypotheses for the same payload (one without decoherence; with
  decoherence one per counter vector still consistent with the game, see ``DecoherenceTracker``); move values are
  averaged over them (uniform weights).
* Values come from the kit table of the payload's (quantumness, decoherence) when one was loaded
  (``<dir>/kit_q{q}_d{dec}.npz``, full or ``_partial``: every stored value is exact), and states not in it are
  searched exactly by ``KitSolver.value`` with a node cap and the deadline.  Without a table the same capped search
  runs on an in-memory solver per setting (kept between requests, so a round's endgame gets faster).  If any
  value cannot be had in time the answer is None and the caller uses Anabi.
* Among moves within 1e-6 of the best value it prefers the one that gives the opponent the most room to go
  wrong: the Born-weighted mean, over the opponent's replies, of how much each reply loses against the
  opponent's best reply (``spread``); computed only while time remains, else the first optimal move.
* 12 November 2026: when ``<dir>/rand_q{q}_d{dec}.npz`` exists (``tqt_rand``, built from a complete table) the
  tie-break is instead the expected round score against a uniformly random opponent (the arena's House bot);
  ``spread`` stays the fallback when a state is not in that table.  Only ever chooses among the optimal moves.
* 13 November 2026 (opt-in, ``anti_random=True``; off by default, then nothing below runs): ``<dir>/arand_q{q}_d{dec}.npz``
  (``tqt_arand``: full expectimax against a uniformly random opponent over ALL legal moves) is loaded, and
  ``choose(..., anti=True)`` plays the best move against a random opponent even when it is not optimal.  The caller
  (``tqt_tournament_bot``) asks for that only after the opponent has played a move the exact table proves
  non-optimal (``deviation``), which a perfect opponent never does.
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time

from .tqt_kit import UNFINISHED, legal_moves
from .tqt_kit_solver import VALUE, Budget, KitSolver, outcomes, state_key

TIE = 1e-6
DEV_MARGIN = 1e-3        # a move counts as a deviation only if it loses more than this (X-view value units)
RUNTIME_TT_MAX = 3_000_000


class ExactPlayer:
    def __init__(self, table_dir: str | None = None, node_cap: int = 300_000, log=None, rand: bool = True,
                 anti_random: bool = False):
        self.node_cap = int(node_cap)
        self.solvers: dict = {}           # (q, dec) -> KitSolver
        self.locks: dict = {}
        self.loaded: dict = {}            # (q, dec) -> description
        self.rand: dict = {}              # (q, dec) -> tqt_rand.RandTable (tie-break against a random opponent)
        self.arand: dict = {}             # (q, dec) -> full anti-random table (only with anti_random=True)
        self.glock = threading.Lock()
        self.log = log or (lambda *a: None)
        if table_dir and os.path.isdir(table_dir):
            # names: kit_qQ_dD.npz / kit_qQ_dD_full.npz (complete or cut; every move stored or not) and
            # *_partial.npz (a checkpoint: exact values, not all states).  Per setting the bot loads ONE file:
            # a non-partial one if any (largest first), else the largest partial.  Every file is logged.
            cands: dict = {}
            for path in sorted(glob.glob(os.path.join(table_dir, "kit_q*_d*.npz"))):
                base = os.path.basename(path)
                m = re.fullmatch(r"kit_q(\d)_d(\d)(_full)?(_partial)?\.npz", base)
                if not m:
                    self.log(f"[tqt-exact] skipped {base} (name not a table name)")
                    continue
                key = (int(m.group(1)), int(m.group(2)))
                partial = bool(m.group(4))
                cands.setdefault(key, []).append((0 if not partial else 1, -os.path.getsize(path), path, partial))
            for key in sorted(cands):
                for rank, (_, _, path, partial) in enumerate(sorted(cands[key])):
                    base = os.path.basename(path)
                    if key in self.loaded:
                        self.log(f"[tqt-exact] skipped {base}: q{key[0]} dec{key[1]} already has {self.loaded[key]}")
                        continue
                    try:
                        s = KitSolver.load(path)
                    except Exception as e:       # noqa: BLE001
                        self.log(f"[tqt-exact] could not load {base}: {e!r}")
                        continue
                    self.solvers[key] = s
                    self.locks[key] = threading.Lock()
                    self.loaded[key] = f"{len(s.keys)} states" + (" partial" if partial else "")
                    self.log(f"[tqt-exact] table q{key[0]} dec{key[1]}: {self.loaded[key]} ({base})")
            if rand:
                from .tqt_rand import RandTable
                for path in sorted(glob.glob(os.path.join(table_dir, "rand_q*_d*.npz"))):
                    base = os.path.basename(path)
                    m = re.fullmatch(r"rand_q(\d)_d(\d)\.npz", base)
                    if not m:
                        self.log(f"[tqt-exact] skipped {base} (name not a rand table name)")
                        continue
                    key = (int(m.group(1)), int(m.group(2)))
                    if key not in self.loaded:
                        self.log(f"[tqt-exact] skipped {base}: no exact table for q{key[0]} dec{key[1]}")
                        continue
                    try:
                        rt = RandTable(path)
                    except Exception as e:       # noqa: BLE001
                        self.log(f"[tqt-exact] could not load {base}: {e!r}")
                        continue
                    if (rt.q, rt.dec) != key:
                        self.log(f"[tqt-exact] skipped {base}: it holds q{rt.q} dec{rt.dec}")
                        continue
                    self.rand[key] = rt
                    self.log(f"[tqt-exact] tie-break vs random q{key[0]} dec{key[1]}: {len(rt)} states ({base})")
            if anti_random:
                from .tqt_rand import RandTable
                for path in sorted(glob.glob(os.path.join(table_dir, "arand_q*_d*.npz"))):
                    base = os.path.basename(path)
                    m = re.fullmatch(r"arand_q(\d)_d(\d)\.npz", base)
                    key = (int(m.group(1)), int(m.group(2))) if m else None
                    if key is None or key not in self.loaded or key[1] != 0 or self.loaded[key].endswith("partial"):
                        self.log(f"[tqt-exact] skipped {base} (needs a complete exact table, decoherence off)")
                        continue
                    try:
                        rt = RandTable(path)
                    except Exception as e:       # noqa: BLE001
                        self.log(f"[tqt-exact] could not load {base}: {e!r}")
                        continue
                    if (rt.q, rt.dec) != key:
                        self.log(f"[tqt-exact] skipped {base}: it holds q{rt.q} dec{rt.dec}")
                        continue
                    self.arand[key] = rt
                    self.log(f"[tqt-exact] anti-random after deviation q{key[0]} dec{key[1]}: {len(rt)} states ({base})")

    def _solver(self, q: int, dec: int):
        key = (q, dec)
        with self.glock:
            s = self.solvers.get(key)
            if s is None:
                s = self.solvers[key] = KitSolver(q, dec)
                self.locks[key] = threading.Lock()
            if len(s.tt) > RUNTIME_TT_MAX:
                s.tt = {}
            return s, self.locks[key]

    def _val(self, s: KitSolver, rs) -> float:
        if rs.winner != UNFINISHED:
            return VALUE[rs.winner]
        return s.value(rs)

    def _look(self, s: KitSolver, rs):
        """Table value from X's view, or None (lookup only, no search)."""
        if rs.winner != UNFINISHED:
            return VALUE[rs.winner]
        v = s.lookup(state_key(rs, s.sym))
        if v is None:
            return None
        return v if rs.turn == "X" else -v

    def choose(self, states: list, deadline: float, search_s: float | None = None, anti: bool = False):
        """``deadline``: perf_counter time to give up by; ``search_s``: at most this long for the capped search
        (phase 2) when the table lookups (phase 1) do not settle the move."""
        info: dict = {}
        if not states:
            return None, info
        rs0 = states[0]
        s, lock = self._solver(rs0.q, rs0.dec)
        info["table"] = self.loaded.get((rs0.q, rs0.dec), "none")
        if not lock.acquire(timeout=max(0.0, min(0.2, deadline - time.perf_counter()))):
            info["exact"] = "busy"
            return None, info
        try:
            moves = legal_moves(rs0)
            sg = 1.0 if rs0.turn == "X" else -1.0
            n = len(states)
            kids = {m: [outcomes(rs, m) for rs in states] for m in moves}
            # phase 1: table lookups only.  The solves cut off once a side has a sure win, so a non-best move may be
            # missing; the parent's stored value tells which known moves are optimal.
            pv = [self._look(s, rs) for rs in states]
            known = {}
            for m in moves:
                tot = 0.0
                for outs in kids[m]:
                    for p, c in outs:
                        v = self._look(s, c)
                        if v is None:
                            tot = None
                            break
                        tot += p * v / n
                    if tot is None:
                        break
                if tot is not None:
                    known[m] = tot
            vals, how = None, None
            if known and all(v is not None for v in pv):
                target = sg * sum(pv) / n
                if max(sg * v for v in known.values()) >= target - 1e-5:
                    vals, how = known, "table"
            if vals is None:
                # phase 2: exact values of every move, searching what the table lacks, within the caps
                d2 = deadline if search_s is None else min(deadline, time.perf_counter() + search_s)
                s.nodes, s.node_cap, s.deadline = 0, self.node_cap, d2
                try:
                    vals = {m: sum(p * self._val(s, c) for outs in kids[m] for p, c in outs) / n for m in moves}
                    how = "search"
                except (Budget, RecursionError):
                    info["exact"] = "budget"
                    info["nodes"] = s.nodes
                    return None, info
                info["nodes"] = s.nodes
            best = max(sg * v for v in vals.values())
            opt = [m for m in moves if m in vals and sg * vals[m] >= best - TIE]
            info["value"] = round(best, 6)          # from the mover's view
            info["n_opt"] = len(opt)
            info["n_known"] = len(vals)
            info["n_moves"] = len(moves)
            pick = opt[0]
            rt = self.rand.get((rs0.q, rs0.dec))
            if len(opt) > 1 and rt is not None:
                rsc = self._rand_scores(rt, opt, kids, rs0.turn, n)
                if rsc is not None:
                    pick = max(opt, key=lambda m: (rsc[m], -opt.index(m)))
                    info["tiebreak"] = "rand"
                    info["rand"] = round(rsc[pick], 5)
                    info["rand_worst"] = round(min(rsc.values()), 5)
            at = self.arand.get((rs0.q, rs0.dec)) if anti else None
            if at is not None:
                asc = self._rand_scores(at, moves, kids, rs0.turn, n)
                if asc is not None:
                    day2 = pick if info.get("tiebreak") == "rand" or len(opt) == 1 else None
                    pick = max(moves, key=lambda m: (asc[m], -moves.index(m)))
                    info["tiebreak"] = "arand"
                    info["arand"] = round(asc[pick], 5)
                    info["arand_optimal"] = pick in opt
                    if day2 is not None:
                        info["arand_gain"] = round(asc[pick] - asc[day2], 5)
            if len(opt) > 1 and info.get("tiebreak") not in ("rand", "arand"):
                spreads = {}
                try:
                    for m in opt:
                        if time.perf_counter() > deadline:
                            break
                        tot = 0.0
                        for outs in kids[m]:
                            for p, c in outs:
                                if c.winner == UNFINISHED:
                                    tot += p * self._spread(s, c, deadline) / n
                        spreads[m] = tot
                except TimeoutError:
                    pass
                if spreads:
                    pick = max(spreads, key=lambda m: (spreads[m], -opt.index(m)))
                    info["tiebreak"] = len(spreads)
            info["exact"] = how
            return pick, info
        finally:
            s.node_cap = s.deadline = None
            lock.release()

    @staticmethod
    def _rand_scores(rt, opt: list, kids: dict, me: str, n: int):
        """{move: expected round score against a uniformly random opponent}, or None if a child is not stored."""
        out = {}
        for m in opt:
            tot = 0.0
            for outs in kids[m]:
                for p, c in outs:
                    v = rt.after_our_move(c, me)
                    if v is None:
                        return None
                    tot += p * v / n
            out[m] = tot
        return out

    def deviation(self, prev_states: list, target):
        """Did the opponent play a non-optimal move?  ``prev_states``: the states the opponent could have moved from
        (every outcome of our last move, opponent to move); ``target``: the state we are now asked to move in.
        Returns ``(how, gap)``: how = "deviation" (EVERY move that explains ``target`` loses more than
        ``DEV_MARGIN`` against the opponent's best move, by table lookups only), "optimal" (some explaining move
        does not), "nomatch" (no move explains ``target``) or "unknown" (a value is not in the table).
        ``gap`` is the smallest loss over the explaining moves (opponent's view)."""
        if not prev_states:
            return "nomatch", None
        s = self.solvers.get((target.q, target.dec))
        if s is None:
            return "unknown", None
        tk = state_key(target, False)
        gaps = []
        for c in prev_states:
            if c.winner != UNFINISHED or c.turn == target.turn:
                continue
            kids = {m: outcomes(c, m) for m in legal_moves(c)}
            hits = [m for m, outs in kids.items() if any(state_key(r, False) == tk for _, r in outs)]
            if not hits:
                continue
            sg = 1.0 if c.turn == "X" else -1.0
            vals = {}
            for m, outs in kids.items():
                tot = 0.0
                for p, r in outs:
                    v = self._look(s, r)
                    if v is None:
                        return "unknown", None
                    tot += p * v
                vals[m] = sg * tot
            best = max(vals.values())
            gaps += [best - vals[m] for m in hits]
        if not gaps:
            return "nomatch", None
        g = min(gaps)
        return ("deviation" if g > DEV_MARGIN else "optimal"), g

    def _spread(self, s: KitSolver, rs, deadline: float) -> float:
        """Mean loss of the mover's known replies against the mover's best known reply (mover's view)."""
        sg = 1.0 if rs.turn == "X" else -1.0
        vs = []
        for m in legal_moves(rs):
            if time.perf_counter() > deadline:
                raise TimeoutError()
            tot = 0.0
            for p, c in outcomes(rs, m):
                v = self._look(s, c)
                if v is None:
                    tot = None
                    break
                tot += p * v
            if tot is not None:
                vs.append(sg * tot)
        if not vs:
            return 0.0
        b = max(vs)
        return sum(b - v for v in vs) / len(vs)

    def describe(self) -> dict:
        out = {f"q{q}_d{d}": v for (q, d), v in sorted(self.loaded.items())}
        if self.rand:
            out["rand"] = [f"q{q}_d{d}" for (q, d) in sorted(self.rand)]
        if self.arand:
            out["arand"] = [f"q{q}_d{d}" for (q, d) in sorted(self.arand)]
        return out


__all__ = ["ExactPlayer"]
