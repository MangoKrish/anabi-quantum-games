"""Anabi as a bot for the Quantum TiqTaqToe AI tournament (Evert van Nieuwenburg's kit, 7 November 2026).

The tournament server POSTs the game state to the bot whenever it is the bot's turn and wants ``{"move": [...]}``
back within the timeout (default 5000 ms); a late, illegal or non-200 answer forfeits.  This bot:

* serves ``POST /move`` (any path is accepted) on 127.0.0.1, port configurable, and answers CORS ``OPTIONS``
  preflights (``Access-Control-Allow-Origin: *``); ``GET /health`` says it is up;
* rebuilds the referee's round state from the payload alone (``tqt_kit.state_from_payload``: quantumState,
  measuredSquares, emptySquares, probabilities, pairs, settings, currentTurn) - the rules are ``tqt_kit``, the
  kit referee's game logic ported line by line and validated against the referee's own JS;
* checks that its own legal-move list equals the payload's ``legalMoves`` (same moves, same order) and logs any
  mismatch;
* picks the move with Anabi (``MCTSEngine`` + the ``ANABI`` config through ``anabi_foreign.ForeignMCTS``, the
  TiqTaqToe evaluator of ``opponents.make_evaluator``; the search is not tuned) under a think budget, default
  1.5 s from the request's arrival, well inside the 5 s timeout;
* shrinks both limits to 30 % / 60 % of ``settings.moveTime`` when the server's move time is shorter (at the
  default 5 s they stay 1.5 s / 3.0 s);
* never forfeits by its own doing: the search runs in a worker thread; if it throws, if its move is not in
  ``legalMoves``, or if it is still running at the hard deadline (default 3.0 s), the bot answers with the best
  move found so far (most visited root move of the running search), else ``legalMoves[0]``.  The answer is
  always an element of the payload's own ``legalMoves`` array, returned as is (e.g. ``[1, "M"]``);
* with decoherence on (the payload has no decoherence counters) it infers the counters by matching the new
  payload against its own previous state plus every opponent reply; if that fails it assumes 0.

Arena version (11-12 November 2026, reports/engines/arena_20261112/tiqtaqtoe_FINAL.md): before Anabi, the bot asks
``tqt_perfect.ExactPlayer`` for an exactly optimal move - from the precomputed table of the payload's (quantumness,
decoherence) in ``--tables`` (default reports/engines/arena_20261112/tqt_tables), else from a capped endgame search -
within 1.2 s (0.5 s without a table); if that fails, Anabi gets the rest of the think time.  ``--no-exact`` gives
the 7 November bot.  The hard deadline is 2.4 s (was 3.0 s).

Day 3 (13 November 2026, reports/engines/arena_20261116/tqt/), opt-in ``--anti-random`` (default OFF; without it
nothing new runs and every answer is the day-2 answer): with decoherence off and ``<tables>/arand_qQ_d0.npz``
loaded, ``DeviationTracker`` rebuilds the opponent's move from consecutive payloads (our last move's outcomes ->
the new payload) and asks the exact table whether that move was optimal.  After ``--anti-random-k`` (default 3)
proven deviations in the same game (key gameId + roundId + player; a new round is detected from the board, as the
decoherence tracker does), the bot plays the full expectimax move against a uniformly random opponent over ALL
legal moves.  A perfect opponent never triggers it.

Start for the tournament (see reports/engines/evert_kit_20261107/tiqtaqtoe/report.md):
    cd ~/Documents/qchess && .venv/bin/python -m qchess.games.tqt_tournament_bot --port 8791
and give the server the URL ``http://127.0.0.1:8791/move``.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..engines.anabi import ANABI
from ..engines.anabi_foreign import ForeignMCTS
from ..engines.base import Budget
from .tqt_kit import KitTQTBoard, legal_moves, state_from_payload

DEFAULT_PORT = 8791
DEFAULT_THINK = 1.5      # seconds from the request's arrival to the answer (target)
DEFAULT_HARD = 2.4       # seconds: answer with the fallback if the search is still running (was 3.0 on 7 Nov)
EXACT_TABLE_S = 1.2      # seconds the exact player may use when a table for the setting is loaded
EXACT_SEARCH_S = 0.5     # ... when there is no table (capped endgame search only); with a table, the most the
                         # search may take when a state is missing from it
import os as _os
DEFAULT_TABLES = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "reports", "engines",
                               "arena_20261112", "tqt_tables")
MARGIN = 0.05            # seconds kept for building the answer after the search


class LiveMCTS(ForeignMCTS):
    """``ForeignMCTS`` unchanged, except that it exposes the root of the running search (``live_root``) so a
    hard deadline can still answer with the best move found so far."""

    live_root = None

    def _iterate(self, root):
        self.live_root = root
        return super()._iterate(root)


def make_engine(seed: int = 0) -> LiveMCTS:
    """Anabi exactly as ``core.make_anabi_for(make_evaluator('tqt:...'))`` builds it, on ``LiveMCTS``."""
    from .tqt_kit import make_kit_evaluator
    cfg = {**ANABI, "name": "Anabi"}
    eng = LiveMCTS(cfg, evaluator=make_kit_evaluator())
    eng.name = "Anabi"
    eng.new_game("w", seed)
    return eng


def best_so_far(eng, allowed=None) -> tuple | None:
    """Most visited root move of the (possibly still running) search, ties by mean; else the best 1-ply value.
    Only moves in ``allowed`` (tuples; the payload's legalMoves) are considered when it is given."""
    root = getattr(eng, "live_root", None)
    if root is None:
        return None
    for _ in range(3):                       # the search thread may be resizing the dicts
        try:
            edges = [(m, e.N, e.W) for m, e in list(root.edges.items())]
            ply1 = dict(root.ply1) if root.ply1 else {}
            break
        except RuntimeError:
            continue
    else:
        return None
    if allowed is not None:
        edges = [x for x in edges if tuple(x[0]) in allowed]
        ply1 = {m: v for m, v in ply1.items() if tuple(m) in allowed}
    visited = [(n, w / n, m) for m, n, w in edges if n > 0]
    if visited:
        return max(visited, key=lambda x: (x[0], x[1]))[2]
    if ply1:
        return max(ply1.items(), key=lambda kv: kv[1])[0]
    return None


def _same_state(a, b, tol: float = 1e-9) -> bool:
    if a.turn != b.turn or a.measured != b.measured or a.empty != b.empty or list(a.amps) != list(b.amps):
        return False
    return all(abs(x[0] - y[0]) <= tol and abs(x[1] - y[1]) <= tol for x, y in zip(a.amps.values(), b.amps.values()))


class DecoherenceTracker:
    """The payload has no decoherence counters; infer them from the bot's own previous state in the round."""

    def __init__(self):
        self.pending: dict = {}      # key -> states after my last move (every outcome x every hypothesis)
        self.pending_fp: dict = {}   # key -> fingerprint of the request whose answer made that entry
        self.hyp: dict = {}          # key -> every counter vector consistent with the current payload
        self.hyp_fp: dict = {}       # key -> fingerprint of the state those hypotheses are for
        self.lock = threading.Lock()

    @staticmethod
    def key(p: dict) -> tuple:
        return (p.get("gameId"), p.get("roundId"), p.get("player") or p.get("currentTurn"))

    @staticmethod
    def _state_fp(rs) -> tuple:
        return (rs.turn, tuple(rs.measured), tuple(rs.empty),
                tuple((b, round(a[0], 12), round(a[1], 12)) for b, a in rs.amps.items()))

    @staticmethod
    def _round_start(target) -> bool:
        """The payload is the first or second state of a round (empty board, or one opponent move from it):
        then every counter is 0.  (The referee reuses gameId / roundId across matches, so a stored state may
        be stale; this check does not depend on it.)"""
        if all(target.empty) and not any(target.measured):
            return True
        if any(target.measured) and target.q >= 2:
            return False
        from .tqt_kit import RoundState
        fresh = KitTQTBoard(RoundState(target.q, target.dec, "O" if target.turn == "X" else "X"))
        for m in fresh.legal_moves():
            for _, _, r in fresh._outcomes(m):
                if _same_state(r, target):
                    return True
        return False

    def counters_for(self, p: dict):
        """(counters, how): how = "start" | "inferred" | "ambiguous" (several counter vectors fit: the first
        in sorted order) | "unknown" (0 assumed)."""
        with self.lock:
            cands = self.pending.pop(self.key(p), None)
            self.pending_fp.pop(self.key(p), None)
        target = state_from_payload(p)
        k = self.key(p)
        if self._round_start(target):
            with self.lock:
                self.hyp[k] = [(0,) * 9]
                self.hyp_fp[k] = self._state_fp(target)
            return [0] * 9, "start"
        found = set()
        for rs in cands or []:
            if rs.winner:
                continue
            b = KitTQTBoard(rs)
            for m in legal_moves(rs):
                for _, _, r in b._outcomes(m):
                    if _same_state(r, target):
                        found.add(tuple(r.counters))
        found = sorted(found)
        fp = self._state_fp(target)
        if not found:
            with self.lock:
                prev = self.hyp.get(k)
                same = self.hyp_fp.get(k) == fp
            if prev and same:                     # the same state again (a retry): keep what we knew
                return list(prev[0]), "kept"
        with self.lock:
            self.hyp[k] = found or [(0,) * 9]
            self.hyp_fp[k] = fp
        if len(found) == 1:
            return list(found[0]), "inferred"
        if found:
            return list(found[0]), "ambiguous"
        return [0] * 9, "unknown"

    def after_my_move(self, p: dict, board: KitTQTBoard, move) -> None:
        """Remember every state my move can lead to, under every counter hypothesis still open."""
        k = self.key(p)
        with self.lock:
            hyps = self.hyp.pop(k, None) or [tuple(board.rs.counters)]
        outs = []
        try:
            for h in hyps:
                rs = board.rs.clone()
                rs.counters = list(h)
                outs += [r for _, _, r in KitTQTBoard(rs)._outcomes(move)]
        except Exception:                   # noqa: BLE001
            return
        with self.lock:
            self.pending[k] = outs


class DeviationTracker:
    """Counts, per game, the opponent's moves that the exact table proves non-optimal (decoherence off only)."""

    MAX_GAMES = 5000

    def __init__(self):
        self.games: collections.OrderedDict = collections.OrderedDict()   # key -> {"n", "fp", "pending"}
        self.lock = threading.Lock()

    key = staticmethod(DecoherenceTracker.key)

    def last_fp(self, p: dict):
        with self.lock:
            e = self.games.get(self.key(p))
            return None if e is None else e.get("req")

    def observe(self, p: dict, target, exact, req_fp=None) -> tuple[int, str, float | None]:
        """(deviations so far in this game, how, gap) for the payload ``p`` (state ``target``, our turn)."""
        k = self.key(p)
        fp = DecoherenceTracker._state_fp(target)
        with self.lock:
            e = self.games.get(k)
        prev, n, how, gap = None, 0, None, None
        if DecoherenceTracker._round_start(target):
            how = "start"
            if not all(target.empty):          # the opponent moved first: check that move from the empty board
                from .tqt_kit import RoundState
                prev = [RoundState(target.q, target.dec, "O" if target.turn == "X" else "X")]
        elif e is not None and e.get("fp") == fp:  # the same state again (a retry): nothing new
            return e["n"], "kept", None
        elif e is not None and e.get("pending"):
            prev, n = e["pending"], e["n"]
        else:
            how = "lost"                       # no previous state of ours: assume no deviation (day-2 play)
        if prev is not None:
            how, gap = exact.deviation(prev, target)
            if how == "deviation":
                n += 1
            elif how == "nomatch":             # not a continuation of our last state: a new game reusing the ids
                n = 0
        with self.lock:
            self.games[k] = {"n": n, "fp": fp, "pending": None, "req": req_fp}
            self.games.move_to_end(k)
            while len(self.games) > self.MAX_GAMES:
                self.games.popitem(last=False)
        return n, how, gap

    def after_my_move(self, p: dict, rs, move) -> None:
        from .tqt_kit_solver import outcomes
        outs = [c for _, c in outcomes(rs, tuple(move))]
        with self.lock:
            e = self.games.get(self.key(p))
            if e is not None:
                e["pending"] = outs


class TournamentBot:
    def __init__(self, think: float = DEFAULT_THINK, hard: float = DEFAULT_HARD, seed: int = 0,
                 max_engines: int = 4, log_path: str | None = None, warm: bool = True, engine_factory=None,
                 tables: str | None = None, exact: bool = False, rand: bool = True, anti_random: bool = False,
                 anti_k: int = 3):
        self.think, self.hard, self.seed = float(think), float(hard), int(seed)
        self.max_engines = int(max_engines)
        self.factory = engine_factory or make_engine
        self.free: list = []
        self.n_engines = 0
        self.pool_lock = threading.Lock()
        self.tracker = DecoherenceTracker()
        self.retry: dict = {}                # request fingerprint -> (answer, info): connector retries
        self.retry_lock = threading.Lock()
        self.inflight: dict = {}             # fingerprint -> Event set when the first answer is ready
        self.records: collections.deque = collections.deque(maxlen=20000)
        self.counts = collections.Counter()
        self.log_path = log_path
        self.log_lock = threading.Lock()
        self.exact = None
        self.anti = None                     # DeviationTracker when --anti-random is on
        self.anti_k = max(1, int(anti_k))
        if exact:
            from .tqt_perfect import ExactPlayer
            kw = {"anti_random": True} if anti_random else {}
            self.exact = ExactPlayer(tables, log=lambda m: print(m, file=sys.stderr, flush=True), rand=rand, **kw)
            if anti_random:
                self.anti = DeviationTracker()
        if warm:
            self._warm()

    # ---- engines
    def _get_engine(self):
        with self.pool_lock:
            if self.free:
                return self.free.pop()
            if self.n_engines >= self.max_engines:
                return None
            self.n_engines += 1
            k = self.n_engines
        return self.factory(self.seed + k)

    def _put_engine(self, eng) -> None:
        with self.pool_lock:
            self.free.append(eng)

    def _warm(self) -> None:
        eng = self._get_engine()
        try:
            b = KitTQTBoard.new(2)
            eng.choose(b, Budget(seconds=0.3))
        finally:
            self._put_engine(eng)

    def limits(self, payload: dict) -> tuple[float, float]:
        """(think, hard) seconds: the configured ones, shrunk to 30 % / 60 % of the payload's
        ``settings.moveTime`` (the referee's timeout in seconds) when that is shorter than 5 s."""
        think, hard = self.think, self.hard
        try:
            mt = float((payload.get("settings") or {}).get("moveTime") or 0)
        except (TypeError, ValueError):
            mt = 0.0
        if mt > 0:
            think, hard = min(think, 0.3 * mt), min(hard, 0.6 * mt)
        return round(think, 4), round(hard, 4)

    # ---- one request
    @staticmethod
    def _fingerprint(payload: dict):
        try:
            return json.dumps([payload.get("gameId"), payload.get("roundId"), payload.get("currentTurn"),
                               payload.get("player"), payload.get("quantumState"), payload.get("measuredSquares"),
                               payload.get("emptySquares"), payload.get("legalMoves"), payload.get("settings")],
                              sort_keys=True, default=str)
        except Exception:                        # noqa: BLE001
            return None

    def decide(self, payload: dict, t0: float | None = None) -> tuple[list | None, dict]:
        """Retried requests (the same state POSTed again, e.g. by a connector retry) get the same answer without
        touching the decoherence tracker."""
        t0 = time.perf_counter() if t0 is None else t0
        fp = self._fingerprint(payload)
        ev = None
        if fp is not None:
            with self.retry_lock:
                ev = self.inflight.get(fp)
                if ev is None:
                    self.inflight[fp] = threading.Event()
            if ev is not None:
                # the same POST while the first is still being answered: wait for that answer, never decide twice
                _, hard = self.limits(payload)
                ev.wait(max(0.0, hard - (time.perf_counter() - t0)))
                with self.retry_lock:
                    got = self.retry.get(fp)
                if got is not None:
                    return got[0], {"source": "retry_inflight", "first_source": got[1].get("source")}
                legal = payload.get("legalMoves") or []
                return (legal[0] if legal else None), {"source": "first_legal", "inflight_timeout": True}
        try:
            return self._decide_cached(payload, t0, fp)
        finally:
            if fp is not None:
                with self.retry_lock:
                    e = self.inflight.pop(fp, None)
                if e is not None:
                    e.set()

    def _decide_cached(self, payload: dict, t0: float, fp):
        if fp is not None:
            with self.retry_lock:
                got = self.retry.get(fp)
            dec_on = False
            try:
                dec_on = int((payload.get("settings") or {}).get("decoherence", 0) or 0) > 0
            except (TypeError, ValueError):
                pass
            if got is not None and dec_on:
                # a true retry: my first answer's tracker entry is still waiting for the opponent's reply.  If it
                # is gone, this is a new round that reuses the ids (the local referee does): decide afresh.
                with self.tracker.lock:
                    k = self.tracker.key(payload)
                    if k not in self.tracker.pending or self.tracker.pending_fp.get(k) != fp:
                        got = None
            if got is not None and self.anti is not None and self.anti.last_fp(payload) != fp:
                got = None                   # anti-random: reuse an answer only for a true retry of the last request
            if got is not None:
                return got[0], {"source": "retry_cache", "first_source": got[1].get("source")}
        mv, info = self._decide(payload, t0)
        if fp is not None:
            with self.tracker.lock:
                k = self.tracker.key(payload)
                if k in self.tracker.pending:
                    self.tracker.pending_fp[k] = fp       # this request's answer created that entry
        if fp is not None and mv is not None:
            with self.retry_lock:
                self.retry[fp] = (mv, dict(info))
                while len(self.retry) > 2000:
                    self.retry.pop(next(iter(self.retry)))
        return mv, info

    def _decide(self, payload: dict, t0: float | None = None) -> tuple[list | None, dict]:
        t0 = time.perf_counter() if t0 is None else t0
        info: dict = {"source": None}
        legal = payload.get("legalMoves") or []
        if not legal:
            # not the referee's format (it always sends legalMoves): answer from our own rules if we can
            try:
                legal = [list(m) for m in legal_moves(state_from_payload(payload))]
                payload = {**payload, "legalMoves": legal}
                info["legal_from_rules"] = True
            except Exception:                    # noqa: BLE001
                legal = []
        if not legal:
            info["source"] = "no_legal_moves"
            return None, info

        def answer(m, source):
            if m is not None:
                for x in legal:
                    if list(x) == list(m):
                        info["source"] = source
                        return x, info
            return None, info

        if len(legal) == 1:
            info["source"] = "only_move"
            if self.anti is not None:
                try:
                    rs1 = state_from_payload(payload)
                    if rs1.dec == 0 and (rs1.q, 0) in self.exact.arand:
                        info["dev"] = list(self.anti.observe(payload, rs1, self.exact, self._fingerprint(payload)))
                        self.anti.after_my_move(payload, rs1, tuple(legal[0]))
                except Exception as e:           # noqa: BLE001
                    info["dev_error"] = repr(e)
            try:
                if int((payload.get("settings") or {}).get("decoherence", 0) or 0) > 0:
                    counters, info["counters"] = self.tracker.counters_for(payload)
                    self.tracker.after_my_move(payload, KitTQTBoard.from_payload(payload, counters),
                                               tuple(legal[0]))
            except Exception:                    # noqa: BLE001
                pass
            return legal[0], info
        board = None
        try:
            st = payload.get("settings") or {}
            counters, how = None, None
            if int(st.get("decoherence", 0) or 0) > 0:
                counters, how = self.tracker.counters_for(payload)
                info["counters"] = how
            board = KitTQTBoard.from_payload(payload, counters, seed=self.seed)
            ours = [list(m) for m in board.legal_moves()]
            info["legal_match"] = ours == [list(x) for x in legal]
            if not info["legal_match"]:
                sj = {json.dumps(x) for x in legal}
                so = {json.dumps(x) for x in ours}
                info["legal_only_payload"] = sorted(sj - so)[:20]
                info["legal_only_ours"] = sorted(so - sj)[:20]
                info["legal_same_set"] = sj == so
                print(f"[tqt-bot] legal-move mismatch: payload-only {info['legal_only_payload']} "
                      f"ours-only {info['legal_only_ours']}", file=sys.stderr, flush=True)
        except Exception as e:              # noqa: BLE001
            info["build_error"] = repr(e)
            print(f"[tqt-bot] could not rebuild the state: {e!r}", file=sys.stderr, flush=True)
            board = None

        chosen, source = None, None
        if board is not None:
            eng = self._get_engine()
            if eng is None:
                info["engine"] = "none_free"
            else:
                eng.live_root = None
                box: dict = {}
                done = threading.Event()
                think, hard = self.limits(payload)
                info["limits_s"] = [think, hard]
                budget_s = max(0.05, think - (time.perf_counter() - t0) - MARGIN)
                info["budget_s"] = round(budget_s, 4)
                it0 = eng._stats.get("iterations", 0) if isinstance(getattr(eng, "_stats", None), dict) else 0

                hyps = None
                use_exact = (self.exact is not None and (board.rs.q, board.rs.dec) in self.exact.loaded
                             and info.get("legal_match") is True)
                # no table for this setting (level 4): Anabi only, exactly the 7 November bot.  The capped search
                # alone scored 0.442 [0.356, 0.531] against Anabi at q4 (h2h, 11 Nov), so it is not used there.
                if use_exact:
                    hyps = [counters] if counters is not None else [None]
                    if counters is not None:
                        with self.tracker.lock:
                            hyps = list(self.tracker.hyp.get(self.tracker.key(payload)) or [counters])[:8]
                    tabled = (board.rs.q, board.rs.dec) in self.exact.loaded
                    exact_s = min(EXACT_TABLE_S if tabled else EXACT_SEARCH_S, 0.8 * think)
                anti_on = False
                if (use_exact and self.anti is not None and board.rs.dec == 0
                        and (board.rs.q, 0) in self.exact.arand):
                    try:
                        nd, dhow, gap = self.anti.observe(payload, state_from_payload(payload), self.exact,
                                                          self._fingerprint(payload))
                        info["dev"] = [nd, dhow, None if gap is None else round(gap, 6)]
                        anti_on = nd >= self.anti_k
                        info["anti"] = anti_on
                    except Exception as e:       # noqa: BLE001
                        info["dev_error"] = repr(e)

                def run():
                    try:
                        if use_exact:
                            try:
                                states = [state_from_payload(payload, h) for h in hyps]
                                partial = str(self.exact.loaded.get((board.rs.q, board.rs.dec), "")).endswith("partial")
                                # partial table: only a quick search (children known, parent not stored), so a
                                # missing state costs Anabi at most 0.05 s of its think time
                                kw = {"anti": True} if anti_on else {}
                                xm, xi = self.exact.choose(states, t0 + exact_s,
                                                           search_s=(0.05 if partial else EXACT_SEARCH_S), **kw)
                                box["exact"] = xi
                                if xm is not None and list(xm) in [list(x) for x in legal]:
                                    box["move"], box["src"] = xm, "exact"
                                    return
                                if xm is not None:
                                    box["exact_not_legal"] = list(xm)
                            except Exception as e:   # noqa: BLE001
                                box["exact"] = {"error": repr(e)}
                        rem = max(0.05, think - (time.perf_counter() - t0) - MARGIN)
                        box["move"] = eng.choose(board.copy(), Budget(seconds=rem))
                    except BaseException as e:   # noqa: BLE001
                        box["error"] = repr(e)
                    finally:
                        box["t"] = time.perf_counter()
                        done.set()
                        self._put_engine(eng)

                th = threading.Thread(target=run, name="anabi-search", daemon=True)
                th.start()
                done.wait(max(0.0, hard - (time.perf_counter() - t0)))
                if done.is_set():
                    if "error" in box:
                        info["search_error"] = box["error"]
                        print(f"[tqt-bot] search error: {box['error']}", file=sys.stderr, flush=True)
                    else:
                        chosen, source = box.get("move"), box.get("src", "anabi")
                        try:
                            info["iterations"] = eng._stats.get("iterations", 0) - it0
                        except Exception:        # noqa: BLE001
                            pass
                else:
                    info["deadline"] = True
                    print("[tqt-bot] hard deadline: answering with the best move so far", file=sys.stderr,
                          flush=True)
                if "exact" in box:
                    info["exact"] = box["exact"]
                if "exact_not_legal" in box:
                    info["exact_not_legal"] = box["exact_not_legal"]
                if chosen is None or answer(chosen, "x")[0] is None:
                    if chosen is not None:
                        info["anabi_move_not_legal"] = list(chosen)
                    chosen, source = best_so_far(eng, {tuple(x) for x in legal}), "best_so_far"
        mv, _ = answer(chosen, source) if chosen is not None else (None, info)
        if mv is None:
            mv = legal[0]
            info["source"] = "first_legal"
        if board is not None and board.rs.dec > 0:
            try:
                self.tracker.after_my_move(payload, board, tuple(mv))
            except Exception:                    # noqa: BLE001
                pass
        if "dev" in info and board is not None:
            try:
                self.anti.after_my_move(payload, board.rs, tuple(mv))
            except Exception as e:               # noqa: BLE001
                info["dev_error"] = repr(e)
        return mv, info

    def log(self, rec: dict) -> None:
        self.records.append(rec)
        self.counts["requests"] += 1
        self.counts[f"source_{rec.get('source')}"] += 1
        if rec.get("legal_match") is False:
            self.counts["legal_mismatch"] += 1
        if self.log_path:
            with self.log_lock, open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


def make_handler(bot: TournamentBot):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AnabiTQT/1.0"

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/drain":                 # match driver: request records since the last drain
                recs = []
                while bot.records:
                    recs.append(bot.records.popleft())
                self._send(200, {"records": recs})
            else:
                self._send(200, {"ok": True, "bot": "Anabi (Quantum TiqTaqToe, kit rules)", "think_s": bot.think,
                                 "hard_s": bot.hard, "counts": dict(bot.counts),
                                 "tables": bot.exact.describe() if bot.exact is not None else "off"})

        def do_POST(self):
            t0 = time.perf_counter()
            wall = time.time()
            try:
                if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                    parts = []
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            while self.rfile.readline().strip():      # trailers up to the blank line
                                pass
                            break
                        parts.append(self.rfile.read(size))
                        self.rfile.readline()                        # CRLF after each chunk
                    body = b"".join(parts)
                else:
                    n = int(self.headers.get("Content-Length", 0) or 0)
                    body = self.rfile.read(n)
                payload = json.loads(body.decode() or "{}")
            except Exception as e:           # noqa: BLE001
                self._send(400, {"error": f"bad request: {e}"})
                return
            try:
                mv, info = bot.decide(payload, t0)
            except Exception as e:           # noqa: BLE001  (decide already guards; this is the last net)
                legal = payload.get("legalMoves") or []
                mv, info = (legal[0] if legal else None), {"source": "first_legal", "error": repr(e)}
            self._send(200, {"move": mv})
            dt = time.perf_counter() - t0
            st = payload.get("settings") or {}
            rec = {"t": round(wall, 3), "gameId": payload.get("gameId"), "roundId": payload.get("roundId"),
                   "player": payload.get("player"), "q": st.get("quantumness"), "dec": st.get("decoherence"),
                   "n_legal": len(payload.get("legalMoves") or []), "move": mv, "think_s": round(dt, 4), **info}
            bot.log(rec)

        def log_message(self, *a):
            pass

    return Handler


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1", **kw) -> ThreadingHTTPServer:
    bot = TournamentBot(**kw)
    srv = ThreadingHTTPServer((host, port), make_handler(bot))
    srv.daemon_threads = True
    srv.bot = bot
    return srv


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--think", type=float, default=DEFAULT_THINK, help="target seconds per move (default 1.5)")
    ap.add_argument("--hard", type=float, default=DEFAULT_HARD,
                    help="answer with the fallback after this many seconds (default 3.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="append one JSON line per request to this file")
    ap.add_argument("--tables", default=DEFAULT_TABLES, help="folder of exact tables kit_q*_d*.npz")
    ap.add_argument("--no-exact", action="store_true", help="Anabi only (the 7 November bot)")
    ap.add_argument("--no-rand", action="store_true",
                    help="break ties between optimal moves by 'spread' only (the 12 November bot), not by rand_*.npz")
    ap.add_argument("--anti-random", action="store_true",
                    help="opt-in (default off): after the opponent's first proven non-optimal move(s) in a game, play "
                         "the best move against a uniformly random opponent (needs arand_qQ_d0.npz in --tables)")
    ap.add_argument("--anti-random-k", type=int, default=3,
                    help="proven deviations in a round before --anti-random switches (default 3, PREREG 13 Nov)")
    a = ap.parse_args(argv)
    if not (0 < a.think < a.hard):
        ap.error("need 0 < --think < --hard")
    srv = serve(a.port, a.host, think=a.think, hard=a.hard, seed=a.seed, log_path=a.log,
                tables=a.tables, exact=not a.no_exact, rand=not a.no_rand,
                **({"anti_random": True, "anti_k": a.anti_random_k} if a.anti_random else {}))
    ex = srv.bot.exact.describe() if srv.bot.exact is not None else "off"
    print(f"[tqt-bot] Anabi ready on http://{a.host}:{a.port}/move (think {a.think} s, hard {a.hard} s, "
          f"exact tables {ex})", flush=True)
    try:
        srv.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
