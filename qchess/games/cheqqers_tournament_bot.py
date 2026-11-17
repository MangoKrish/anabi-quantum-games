"""Anabi as a Quantum Cheqqers AI-tournament player (Evert van Nieuwenburg's kit, 7 November 2026).

``POST /move`` takes the tournament payload (``ai-tournament-guide-cheqqers.md``) and answers ``{"move": [...]}``,
always one of the payload's ``legalMoves``; ``OPTIONS`` answers the CORS preflight.  The game model is the arena
referee's own code (``tournament_game_v3.mjs`` = the 9 November 2026 ``local-cheqqers-server.js`` lines 1-914, engine
1.0.2) through the cqweb bridge, so Anabi searches exactly the rules the server plays.

State.  The payload has the amplitudes but not the referee's hidden bookkeeping (piece ids, which decide whether a
multi-jump may continue over a split piece; the multi-jump square; repetition history; draw counters).  So the bot
*tracks* each game: it keeps the referee state it last answered from and, on the next request, finds the
descendant (an outcome of its own move, then the opponent's moves until it is its turn again) whose payload equals
the request.  The first request is matched against the start position (and one opponent turn from it).  If nothing
matches, the board is rebuilt from the payload alone (``CQWebBoard.from_payload``: exact amplitudes, approximate
bookkeeping) and the game is marked ``approx``.

Safety.  Tracking, rebuild and search run in a worker thread; the answer is sent by ``deadline_s`` (2.1 s at the
3000 ms timeout, scaled with ``settings.timeout``) whatever the worker does; if it overruns, or anything throws, the
bot answers a legal move from the payload, never an error.  A Node bridge stuck for 4 s is restarted; automatic GC
is off (a full collection runs after each answer); old games are evicted.  Arena hardening: 11 November 2026,
report ``reports/engines/arena_20261112/cheqqers_FINAL.md``.  Day 2 (12 Nov 2026): rules copy
``tournament_game_v4.mjs`` (the 12 Nov 06:00 referee) by default, and a seeded near-best move choice per game
(``DIVERSIFY``) so that an opponent cannot replay a won game.

    .venv/bin/python -m qchess.games.cheqqers_tournament_bot --port 8080 --seconds 1.5
"""
from __future__ import annotations

import argparse
import gc
import heapq
import itertools
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("AOG_CQWEB_BRIDGE", os.path.join(ROOT, "reports", "engines", "evert_kit_20261107", "cheqqers",
                                                       "cqweb_bridge_kit.mjs"))
os.environ.setdefault("AOG_CQWEB_GAME", os.path.expanduser(
    "~/Documents/qchess-external/games/etqg-cheqqers-arena/tournament_game_v4.mjs"))

from ..engines.base import Budget  # noqa: E402
from .cheqqers_web import CQWebBoard  # noqa: E402
from . import cheqqers_web as _cqw  # noqa: E402
_cqw.CQWebBoard.dedupe_arrays = True      # only the first move per serialised array (the referee's findIndex plays it)
from .core import make_anabi_for  # noqa: E402
from .opponents import make_evaluator  # noqa: E402

LOG = sys.stderr


def book_key(pl: dict) -> str:
    """Opening-book key of a request: board size, quantumness and the canonical position (side to move, squares,
    amplitudes, legal moves).  Book builders must use this function."""
    import hashlib
    s = pl.get("settings") or {}
    return hashlib.sha1(repr((int(s.get("boardSize", 0)), int(s.get("quantumness", 0)), canon(pl))).encode()).hexdigest()


def load_book(path: str) -> dict:
    """``{"format": "cqbook-1", "entries": {book_key: move_array}}`` or (13 Nov 2026) ``{"format": "cqbook-2",
    "entries": {book_key: [[move_array, weight], ...]}}``: the move is drawn by weight with the game's RNG, so a
    book position does not replay one fixed move.  Both load as {key: [[move_array, weight], ...]}."""
    with open(path) as f:
        d = json.load(f)
    if d.get("format") == "cqbook-1":
        return {k: [[[int(x) for x in v], 1.0]] for k, v in d["entries"].items()}
    if d.get("format") != "cqbook-2":
        raise ValueError(f"{path}: not a cqbook-1/2 file")
    return {k: [[[int(x) for x in mv], float(w)] for mv, w in v] for k, v in d["entries"].items()}


def move_array(m) -> list:
    return [m[2], m[3]] if m[0] == "m" else [m[1], m[2], m[3]]


def canon(pl: dict):
    """Payload fields that define the position, in a canonical, rounding-tolerant form."""
    board = tuple((c.get("piece"), bool(c.get("crowned")), round(float(c.get("probability") or 0), 6))
                  for c in pl["board"])
    qs = []
    for s in pl.get("quantumStates") or []:
        sq = list(s["squares"])
        order = sorted(range(len(sq)), key=lambda j: sq[j])
        amps = []
        for a in s["amplitudes"]:
            parts = a["state"].split("⊗")
            re_, im_ = round(float(a["re"]), 6), round(float(a["im"]), 6)
            if re_ == 0 and im_ == 0:
                continue
            amps.append(("⊗".join(parts[j] for j in order), re_ + 0.0, im_ + 0.0))
        qs.append((s["type"], tuple(sq[j] for j in order), tuple(sorted(amps))))
    return (pl.get("currentTurn"), board, tuple(sorted(qs)),
            tuple(tuple(m) for m in pl.get("legalMoves") or []))


class _Game:
    def __init__(self):
        self.board = None          # CQWebBoard the bot last answered from (exact referee state) or None
        self.sent = None           # our move at that board
        self.used = time.monotonic()
        self.last = None           # (request key, answer, arrival) of the last computed request (retry detection)
        self.last_rep = 0          # how many times in a row that same position has arrived again
        self.tried = set()         # every answer sent for the current position (replays included)
        self.tried_key = None      # the position (request key) those answers belong to
        self.tried_reset = False
        self.inflight = None       # (request key, Event, box) of the request being computed
        self.rng = None            # per-game random.Random for the near-best choice (seeded by game key + nonce)


#: arena answer budget.  The arena times a move from its side (network through a tunnel to Europe included) and
#: allows timeout + 500 ms.  The bot answers by ``min(seconds + 0.6, 0.8 * timeout - 0.3)`` after the request
#: arrived (2.1 s at 3000 ms), leaving >= 1.4 s for the network at the default timeout.
NET_RESERVE_S = 0.3
_ACTIVE = [0]              # requests being handled in this process (all bots)
_ACTIVE_LOCK = threading.Lock()
MAX_GAMES = 24             # games kept in memory (one pruned board each); the least recently used are dropped
MAX_ENGINES = 2            # games whose engine (with its classical-evaluator cache) is kept
RETRY_EXTRA_S = 1.5        # a retry is the same position again within settings.timeout + this, back to back
STUCK_BRIDGE_S = 4.0
#: near-best choice (12 Nov 2026, day-1 review: two deterministic bots replayed the same 26 plies and the same loss).
#: After the search, the move is drawn (weight = visits) among the root moves with at least DIVERSIFY_MINFRAC of the
#: best move's visits and a mean value within ``diversify`` of it.  The draw uses a per-game RNG seeded with
#: (process nonce, gameId, roundId, player), so a replayed opening does not replay our moves.
DIVERSIFY = 0.02
DIVERSIFY_MINFRAC = 0.25
#: pondering (13 Nov 2026, day-2 review: Benjamin thinks ~2.5 s per move while Anabi idles).  After answering, the
#: search keeps growing the subtree of the move it played (chance outcomes, then the opponent's replies) until the
#: next request arrives; that request's search then starts from the matching node (same referee hash and turn).
PONDER_MAX_S = 8.0         # a ponder stops by itself after this long (opponent gone, game over)
PONDER_MAX_ITERS = 200000
PONDER_JOIN_S = 0.3        # how long a new request waits for a ponder to stop; a ponder still running is abandoned
STALE_CALL_S = 0.3         # a Node call this long, owned by a worker past its deadline, is killed       # a Node call running this long is killed and the bridge restarted


class CheqqersBot:
    def __init__(self, seconds: float = 1.5, deadline_frac: float = 0.8, track_depth: int = 8, seed: int = 0,
                 eval_depth: int = 2, anabi: dict | None = None, book: dict | None = None,
                 soft_seconds: float | None = None, diversify: float = DIVERSIFY, nonce: str | None = None,
                 ponder: bool = False):
        self.seconds = float(seconds)
        self.ponder = bool(ponder)                # think on during the opponent's turn and reuse that subtree
        self._ponds: dict = {}                    # game key -> (thread, stop Event, engine)
        self.diversify = float(diversify)         # 0 = off: the search's own move (deterministic, seed ``seed``)
        self.nonce = nonce if nonce is not None else os.urandom(8).hex()
        self.eval_depth = int(eval_depth)
        self.anabi = dict(anabi or {})            # overrides of Anabi's config (experiments only; default: none)
        self.book = dict(book or {})              # opening book (book_key -> move array); empty by default
        self.soft_seconds = soft_seconds          # with Anabi final "robust": may stop from here (experiments only)
        self.deadline_frac = float(deadline_frac)
        self.track_depth = int(track_depth)
        self.seed = int(seed)
        self.games: dict = {}
        self.engines: dict = {}
        self.busy: set = set()
        self.glock = threading.Lock()
        self.live: dict = {}                      # worker id -> answer deadline (perf_counter) of its request
        self.stats = {"requests": 0, "tracked": 0, "rebuilt": 0, "rebuilt_mismatch": 0, "fallback": 0,
                      "overrun": 0, "errors": 0, "max_ms": 0.0, "rebuild_why": {}, "bridge_restarts": 0,
                      "evicted": 0, "noop_avoided": 0}

    def _why(self, k: str) -> None:
        w = self.stats["rebuild_why"]
        w[k] = w.get(k, 0) + 1

    def deadline_s(self, timeout_ms) -> float:
        """Seconds after the request arrived by which the answer is sent."""
        try:
            t = float(timeout_ms) / 1000.0
        except (TypeError, ValueError):
            t = 3.0
        if not t > 0:
            t = 3.0
        return max(0.25, min(self.seconds + 0.6, self.deadline_frac * t - NET_RESERVE_S))

    def _evict(self) -> None:
        """Drop the least recently used games and their engines (an arena day has many games; caller holds glock)."""
        order = sorted(self.games, key=lambda k: self.games[k].used)
        for k in order[:max(0, len(order) - MAX_GAMES)]:
            del self.games[k]
            self.stats["evicted"] += 1
        recent = set(order[-MAX_ENGINES:])
        for ek in [ek for ek in self.engines if ek[0] not in recent]:
            del self.engines[ek]
        for k in order[:-MAX_ENGINES]:            # older games keep no search subtree
            g = self.games.get(k)
            if g is not None and g.board is not None and not getattr(g, "pruned", False):
                try:
                    g.board, g.pruned = self._pruned(g.board, g.sent, deep=True), True
                except Exception:  # noqa: BLE001
                    g.board = None

    @staticmethod
    def _pruned(board, sent, deep: bool = False):
        """The board to keep for tracking.  A board's children cache pins the whole last search (in Python and in
        the Node store: ~5,000 states per 8x8 game, 11 Nov 2026).  Keep only our move's outcomes; with
        ``deep=False`` their subtrees stay (cached referee children speed up the next search of this game)."""
        keep = board.copy()
        keep._kids = {}
        kids = board._kids.get(sent)
        if kids is not None:
            out = []
            for o, p, c in kids:
                if deep:
                    c = c.copy()
                    c._kids = {}
                out.append((o, p, c))
            keep._kids = {sent: out}
        return keep

    # ---- engines -----------------------------------------------------------------------------------------
    def _engine(self, size: int, rows: int, level: int, color: str, key):
        spec = f"cqw:{size}x{size}r{max(rows, 1)}:L{level}"
        with self.glock:
            k = (key, spec)
            e = self.engines.get(k)
            if e is None or id(e) in self.busy:
                e = make_anabi_for(make_evaluator(spec, depth=self.eval_depth), **self.anabi)
                e.new_game(color, self._game_seed(key) if self.diversify > 0 else self.seed)
                self._watch_root(e)
                self.engines[k] = e
            return e

    def _game_seed(self, key) -> int:
        import hashlib
        return int.from_bytes(hashlib.sha256(repr((self.nonce, key)).encode()).digest()[:8], "big")

    def _near_best(self, eng, m, g):
        """The move to play instead of the search's ``m``: drawn among the near-best root moves (``DIVERSIFY``)."""
        root = getattr(eng, "_live_root", None)
        if root is None or self.diversify <= 0:
            return m
        es = [(mv, e.N, e.W / e.N) for mv, e in list(root.edges.items()) if e.N > 0]
        best = next((x for x in es if move_array(x[0]) == move_array(m)), None)
        if best is None:
            return m
        cands = [x for x in es if x[1] >= DIVERSIFY_MINFRAC * best[1] and x[2] >= best[2] - self.diversify]
        if len(cands) < 2:
            return m
        if g.rng is None:
            import random
            g.rng = random.Random(self._game_seed(("near-best",) + tuple(map(str, getattr(g, "key", ())))))
        x = g.rng.random() * sum(n for _mv, n, _v in cands)
        for mv, n, _v in cands:
            x -= n
            if x < 0:
                break
        if move_array(mv) != move_array(m):
            self.stats["diversified"] = self.stats.get("diversified", 0) + 1
        return mv

    @staticmethod
    def _watch_root(eng) -> None:
        """Let the bot read the search tree while it grows: every iteration gets the root (Anabi unchanged)."""
        orig = eng._iterate

        def it(root):
            eng._live_root = root
            return orig(root)
        eng._iterate = it

    @staticmethod
    def _ranked(eng) -> list:
        """Root moves of the engine's current (or last) search, best first: most visits, then mean value, then the
        exact one-ply value.  Readable while the search runs (11 Nov 2026 review: an overrun played legal[0])."""
        root = getattr(eng, "_live_root", None) if eng is not None else None
        if root is None:
            return []
        for _ in range(3):
            try:
                edges = list(root.edges.items())
                ply1 = dict(root.ply1 or {})
                break
            except RuntimeError:                   # changed size during iteration: try again
                continue
        else:
            return []

        def partial(e):
            """Exact one-ply value from the outcome children evaluated so far (as ForeignMCTS._candidates)."""
            tot = 0.0
            try:
                for o, p in e.outcomes:
                    c = e.children.get(o)
                    v = None if c is None else (c.tv if c.tv is not None else c.ev)
                    if v is None:
                        return -9.0
                    tot += p * v
            except Exception:  # noqa: BLE001
                return -9.0
            return tot

        def key(item):
            m, e = item
            n = e.N
            return (n, e.W / n if n else ply1.get(m, partial(e)))
        ranked = [m for m, _e in sorted(edges, key=key, reverse=True)]
        rest = sorted((m for m in ply1 if m not in root.edges), key=lambda m: -ply1[m])
        return ranked + rest

    # ---- tracking ----------------------------------------------------------------------------------------
    @staticmethod
    def _sig_from_payload(pl: dict):
        return (0 if pl.get("currentTurn") == "white" else 1,
                {i: (c["piece"][0].upper(), bool(c.get("crowned")), float(c.get("probability") or 0))
                 for i, c in enumerate(pl["board"]) if c.get("piece") and float(c.get("probability") or 0) > 0.0001})

    @staticmethod
    def _sig_ok(h: str, want) -> bool:
        """Cheap prefilter: the referee's board hash (turn, per square colour / king / occupancy to 3 decimals)."""
        parts = h.split("|")
        if parts[0] != f"T:{want[0]}" or len(parts) - 1 != len(want[1]):
            return False
        for x in parts[1:]:
            j = 0
            while x[j].isdigit():
                j += 1
            i, col, king, occ = int(x[:j]), x[j], x[j + 1] == "K", float(x[j + 2:])
            w = want[1].get(i)
            if w is None or w[0] != col or w[1] != king or abs(w[2] - occ) > 0.0011:
                return False
        return True

    @staticmethod
    def _parse_hash(h: str) -> dict:
        out = {}
        for x in h.split("|")[1:]:
            j = 0
            while x[j].isdigit():
                j += 1
            out[int(x[:j])] = (x[j], x[j + 1] == "K", float(x[j + 2:]))
        return out

    def _sig_diff(self, h: str, sig) -> int:
        """Squares where a candidate's hash and the request differ (ordering only; 99 on a parse error)."""
        try:
            have = self._parse_hash(h)
        except Exception:  # noqa: BLE001
            return 99
        return sum(1 for i in set(have) | set(sig[1])
                   if (have.get(i) is None) != (sig[1].get(i) is None)
                   or (have.get(i) and (have[i][0] != sig[1][i][0] or have[i][1] != sig[1][i][1]
                                        or abs(have[i][2] - sig[1][i][2]) > 0.0011)))

    @staticmethod
    def _touched(m) -> tuple:
        return (m[2], m[3]) if m[0] == "m" else (m[1], m[2], m[3])

    def _descendants(self, b, me: str, depth: int, stop, sig=None):
        """Boards reachable from b by the opponent's moves (any outcomes) until it is ``me`` to move, yielded
        lazily (the caller stops at the first match).  With ``sig`` (the request's squares) the opponent's moves
        are tried first when they touch squares where the board and the request differ (11 Nov 2026: enumerating
        every reply first timed out on loaded 8x8 hosts)."""
        frontier = [b]
        for d in range(depth + 1):
            nxt = []
            for x in frontier:
                if stop():
                    return
                if x.result() is not None:
                    continue
                if x.turn == me:
                    yield x
                    continue
                if d == depth:
                    continue
                ms = x.legal_moves()
                if sig is not None:
                    try:
                        have = self._parse_hash(x.hash)
                        diff = {i for i in set(have) | set(sig[1])
                                if (have.get(i) is None) != (sig[1].get(i) is None)
                                or (have.get(i) and (have[i][0] != sig[1][i][0] or have[i][1] != sig[1][i][1]
                                                     or abs(have[i][2] - sig[1][i][2]) > 0.0011))}
                        ms = sorted(ms, key=lambda m: -len(diff.intersection(self._touched(m))))
                    except Exception:  # noqa: BLE001
                        pass
                for m in ms:
                    if stop():
                        return
                    for _o, p, c in x.children(m):
                        if p <= 1e-12 or c.result() is not None:
                            continue
                        if c.turn == me:
                            yield c
                        else:
                            nxt.append(c)
            frontier = nxt
            if not frontier:
                break

    def _match(self, cands, pl, stop):
        """The candidate whose payload equals the request.  First pass: only candidates passing the hash
        prefilter.  Second pass (if none): every other candidate - the referee's hash and its payload can
        disagree in unnormalised interference states (level 3: the hash shows 1 - P(empty), the payload
        P(occupied)), so the prefilter may reject the true state (found 11 Nov 2026)."""
        want, sig = canon(pl), self._sig_from_payload(pl)
        rest = []                                 # (square difference, n, board) of prefilter rejects
        self._seen = 0

        def full(c):
            try:
                return canon(c.payload(3000)) == want
            except Exception:  # noqa: BLE001
                return False
        for c in cands:
            self._seen += 1
            if stop():
                return None
            if not self._sig_ok(c.hash, sig):
                heapq.heappush(rest, (self._sig_diff(c.hash, sig), self._seen, c))
            elif full(c):
                return c
            # interleaved: every 4th candidate also try the closest prefilter reject (in unnormalised level-3
            # states the true one fails the prefilter; waiting to the end let tracking time out, review 3)
            if rest and self._seen % 4 == 0:
                _d, _n, r = heapq.heappop(rest)
                if full(r):
                    self.stats["matched_unfiltered"] = self.stats.get("matched_unfiltered", 0) + 1
                    return r
        rest = [c for _d, _n, c in sorted(rest, key=lambda t: (t[0], t[1]))]
        for c in rest:
            if stop():
                return None
            try:
                if canon(c.payload(3000)) == want:
                    self.stats["matched_unfiltered"] = self.stats.get("matched_unfiltered", 0) + 1
                    return c
            except Exception:  # noqa: BLE001
                continue
        return None

    def _board_for(self, pl: dict, g: "_Game", prev, stop):
        me = "w" if pl.get("player", pl.get("currentTurn")) == "white" else "b"
        s = pl["settings"]
        size, level = int(s["boardSize"]), int(s["quantumness"])
        found, why = None, "new_game"
        if prev is not None and prev[0] is not None and prev[1] is not None:
            why = "no_match"
            try:
                sig = self._sig_from_payload(pl)
                kids = [c for _o, p, c in prev[0].children(prev[1]) if p > 1e-12]
                cands = (x for c in kids for x in self._descendants(c, me, self.track_depth, stop, sig))
                found = self._match(cands, pl, stop)
                if found is None and not stop() and os.environ.get("CQ_DEBUG_DIR"):
                    self._dump(pl, prev, [])
            except Exception as e:  # noqa: BLE001  (e.g. the bridge holding the old state was restarted)
                why = "track_error"
                print("tracking failed:", repr(e), file=LOG)
            if found is None and stop():
                why = "track_timeout"
        if found is None and not stop():
            # a new game (the kit reuses gameId/roundId across games): its start, or one opponent turn from it.
            spr = size // 2
            npieces = sum(1 for c in pl["board"] if c.get("piece"))
            rows_guess = [size // 2 - 1] + [r for r in range(1, size // 2)
                                            if r != size // 2 - 1 and 2 * r * spr == npieces]
            # occupied squares: the start count, + 1 if White opened with a split (a split piece shows on two
            # squares; without the + 2 every Black game that opens against a split was rebuilt, 11 Nov 2026)
            if npieces <= 2 * (size // 2 - 1) * spr + 2 or len(rows_guess) > 1:
                for rows in rows_guess:
                    start = CQWebBoard.new(size, rows, level)
                    sig = self._sig_from_payload(pl)
                    found = self._match(itertools.chain([start], self._descendants(start, me, 2, stop, sig)), pl, stop)
                    if found is not None:
                        break
        if found is not None:
            self.stats["tracked"] += 1
            return found
        b = CQWebBoard.from_payload(pl)
        self.stats["rebuilt"] += 1
        self._why(why)
        if not b.payload_match:
            self.stats["rebuilt_mismatch"] += 1
        return b

    @staticmethod
    def _dump(pl, prev, cands):
        """Debug only (env CQ_DEBUG_DIR): the request that tracking could not match, with what was searched."""
        try:
            d = os.environ["CQ_DEBUG_DIR"]
            os.makedirs(d, exist_ok=True)
            rec = {"payload": pl, "prev_hash": prev[0].hash, "prev_turn": prev[0].turn, "sent": list(prev[1]),
                   "prev_payload": prev[0].payload(3000), "cands": [[c.hash, c.turn] for c in cands[:200]],
                   "ncands": len(cands)}
            with open(os.path.join(d, f"nomatch_{os.getpid()}_{time.time():.3f}.json"), "w") as f:
                json.dump(rec, f)
        except Exception as e:  # noqa: BLE001
            print("dump failed:", repr(e), file=LOG)

    # ---- referee no-ops ----------------------------------------------------------------------------------
    def _is_noop(self, board, m) -> bool:
        """True if the referee would leave the game unchanged with the same side to move.  Happens when applying
        the move throws inside the referee (e.g. QuantumLimitExceeded: an entangled system above 2^15 on 8x8) and
        its catch-all swallows it; replaying such a move loops the game forever (7 Nov 2026, Q8-L2 game 19)."""
        kids = board.children(m)
        return bool(kids) and all(c.hash == board.hash and c.turn == board.turn for _o, _p, c in kids)

    def _noop_status(self, board, m, deadline, reserve: float):
        """_is_noop from cached referee children; a Node call only if more than ``reserve`` s are left.  None =
        unknown (runs on the request thread, so it must never block: 11 Nov 2026 review)."""
        cache = getattr(board, "_kids", None)
        if cache is None or m in cache:
            return self._is_noop(board, m)
        if deadline - time.perf_counter() > reserve:
            return self._is_noop(board, m)
        return None

    def _avoid_noop(self, board, m, ms, deadline, ranked=()):
        if not self._noop_status(board, m, deadline, 0.3):
            return m
        self.stats["noop_avoided"] += 1
        order = [x for x in ranked if x in ms] + [x for x in ms if x not in ranked]
        for reserve in (None, 0.4):               # cached alternatives first, then Node calls while time is left
            for alt in order:
                if alt == m or (reserve is not None and time.perf_counter() > deadline - 0.05):
                    continue                      # (the cache-only pass costs no Node call: always allowed)
                if reserve is None:
                    cache = getattr(board, "_kids", None)
                    if cache is not None and alt not in cache:
                        continue
                    st = self._is_noop(board, alt)
                else:
                    st = self._noop_status(board, alt, deadline, reserve)
                if st is False:
                    return alt
        return m

    # ---- one request -------------------------------------------------------------------------------------
    def _check_bridge(self) -> None:
        """A Node call that has run for STUCK_BRIDGE_S blocks every later request: kill it; the next call starts a
        fresh bridge, and games on the old one are rebuilt from their payloads."""
        b = _cqw._BRIDGE
        if b is not None and not b.broken and b.busy_for() > STUCK_BRIDGE_S:
            print(f"bridge stuck for {b.busy_for():.1f} s: restarting it", file=LOG)
            b.kill()
            self.stats["bridge_restarts"] += 1

    def _check_stale_bridge(self) -> None:
        """The Node call in progress belongs to a worker whose answer deadline has passed and has run > STALE_CALL_S:
        that request is abandoned and its call would block the current one, so restart the bridge.  Checked at each
        new request and every 0.1 s while waiting for a worker (11 Nov 2026 reviews; STUCK_BRIDGE_S stays the
        backstop)."""
        b = _cqw._BRIDGE
        if b is None or b.broken or b.busy_for() <= STALE_CALL_S:
            return
        owner = getattr(b, "call_owner", None)
        with self.glock:
            d = self.live.get(owner)
        if d is not None and d < time.perf_counter():
            print(f"bridge call of an abandoned request running {b.busy_for():.1f} s: restarting it", file=LOG)
            b.kill()
            self.stats["bridge_restarts"] += 1
            self.stats["stale_kills"] = self.stats.get("stale_kills", 0) + 1

    def choose(self, pl: dict, t0: float | None = None) -> list:
        with _ACTIVE_LOCK:
            _ACTIVE[0] += 1
        try:
            return self._choose_req(pl, t0)
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE[0] -= 1

    def _choose_req(self, pl: dict, t0: float | None = None) -> list:
        """Everything (tracking, rebuild, search) runs in a worker; the answer is sent at the deadline at the
        latest, whatever the worker is doing.  Only a worker that finished in time updates the game's state.
        ``t0``: perf_counter when the request arrived (default: now)."""
        t0 = time.perf_counter() if t0 is None else t0
        self.stats["requests"] += 1
        legal = [list(m) for m in pl.get("legalMoves") or []]
        if not legal:
            return []
        deadline = t0 + self.deadline_s((pl.get("settings") or {}).get("timeout") or 3000)
        track_until = t0 + 0.35 * (deadline - t0)
        try:
            self._check_bridge()
        except Exception as e:  # noqa: BLE001
            print("bridge check failed:", repr(e), file=LOG)
        key = (pl.get("gameId"), pl.get("roundId"), pl.get("player"))
        if self.ponder:
            try:
                self._stop_ponders()
            except Exception as e:  # noqa: BLE001
                print("ponder stop failed:", repr(e), file=LOG)
        try:
            rkey = canon(pl)
        except Exception:  # noqa: BLE001
            rkey = None
        try:
            window = float((pl.get("settings") or {}).get("timeout") or 3000) / 1000.0 + RETRY_EXTRA_S
        except (TypeError, ValueError):
            window = 3.0 + RETRY_EXTRA_S
        wait_for, avoid = None, ()
        with self.glock:
            g = self.games.get(key)
            if g is None:
                g = self.games[key] = _Game()
                g.key = key
                self._evict()
            g.used = time.monotonic()
            # A retried request (the connector re-sends after a slow answer) is the SAME position arriving again
            # right after it, with no newer position computed in between, within the retry window.  Only then is
            # the stored answer reused.  A position that genuinely repeats later in the game (a king cycle) is
            # always searched and tracked normally (11 Nov 2026, second risk review: replaying old answers walked
            # into threefold repetition).
            same = rkey is not None and g.last is not None and g.last[0] == rkey
            if same:
                g.last_rep += 1
                if t0 - g.last[2] <= window and g.last[1] in legal and g.last_rep < 3:
                    # unless the referee is re-asking because our move was a no-op (it threw inside the referee):
                    # decided from cached referee children only; unknown -> search again
                    noop = None
                    try:
                        cache = getattr(g.board, "_kids", None) if g.board is not None else None
                        kids = cache.get(g.sent) if cache is not None and g.sent is not None else None
                        if kids:
                            noop = all(c.hash == g.board.hash and c.turn == g.board.turn for _o, _p, c in kids)
                    except Exception:  # noqa: BLE001
                        noop = None
                    if noop is False:
                        self.stats["duplicate"] = self.stats.get("duplicate", 0) + 1
                        return list(g.last[1])
                if g.last_rep >= 2:               # the same position asked a third time: none of the moves so far
                    avoid = tuple(g.tried) if g.tried_key == rkey else (tuple(g.last[1]),)
            else:
                g.last_rep = 0
            if rkey is not None and g.inflight is not None and g.inflight[0] == rkey:
                wait_for = g.inflight
        if wait_for is not None:                  # the same request is being computed right now: wait for it
            self.stats["duplicate"] = self.stats.get("duplicate", 0) + 1
            wait_for[1].wait(max(0.0, deadline - time.perf_counter()))
            a = wait_for[2].get("answer")
            return list(a) if a is not None and a in legal else legal[0]
        try:
            self._check_stale_bridge()
        except Exception as e:  # noqa: BLE001
            print("stale bridge check failed:", repr(e), file=LOG)
        with self.glock:
            prev = (g.board, g.sent)
            g.board, g.sent = None, None          # consumed; set again only by a worker that finishes in time
            mine = (rkey, threading.Event(), {})
            g.inflight = mine
        answer = None
        try:
            answer = self._choose(pl, t0, legal, deadline, track_until, key, g, prev, avoid)
        finally:
            with self.glock:
                mine[2]["answer"] = answer
                if answer is not None and rkey is not None:
                    if not (g.last is not None and g.last[0] == rkey):
                        g.last_rep = 0
                    g.last = (rkey, list(answer), t0)
                    if g.tried_key != rkey or g.tried_reset:
                        g.tried_key, g.tried, g.tried_reset = rkey, set(), False
                    g.tried.add(tuple(answer))
                if g.inflight is mine:
                    g.inflight = None
            mine[1].set()
        return answer

    def _choose(self, pl, t0, legal, deadline, track_until, key, g, prev, avoid=()):
        cancel = threading.Event()
        box: dict = {}

        def work():
            me_t = threading.get_ident()
            with self.glock:
                self.live[me_t] = deadline
            try:
                tt = time.perf_counter()
                board = self._board_for(pl, g, prev, lambda: cancel.is_set() or time.perf_counter() > track_until)
                self.stats["track_ms_max"] = max(self.stats.get("track_ms_max", 0.0), (time.perf_counter() - tt) * 1000)
                box["board"] = board
                ms = board.legal_moves()
                box["ms"] = ms
                if len(legal) == 1:
                    box["m"] = None
                    return
                if self.book:                     # opening book: only an exact position whose move is legal
                    try:
                        ent = self.book.get(book_key(pl))
                        arrays = [move_array(x) for x in ms]
                        ent = [(mv, w) for mv, w in (ent or []) if mv in legal and mv in arrays and w > 0]
                        bm = None
                        if len(ent) == 1:
                            bm = ent[0][0]
                        elif ent:
                            if g.rng is None:
                                import random
                                g.rng = random.Random(self._game_seed(("near-best",) + tuple(map(str, getattr(g, "key", ())))))
                            x = g.rng.random() * sum(w for _mv, w in ent)
                            for bm, w in ent:
                                x -= w
                                if x < 0:
                                    break
                        if bm is not None:
                            box["m"] = ms[arrays.index(bm)]
                            self.stats["book"] = self.stats.get("book", 0) + 1
                            return
                    except Exception as e:  # noqa: BLE001
                        print("book lookup failed:", repr(e), file=LOG)
                s = pl["settings"]
                me = "w" if pl.get("player") == "white" else "b"
                eng = self._engine(int(s["boardSize"]), getattr(board, "rows", 0), int(s["quantumness"]), me, key)
                eng._live_root = None
                box["eng"] = eng
                left = deadline - time.perf_counter() - 0.25
                if left < 0.05 or cancel.is_set():
                    return
                self.busy.add(id(eng))
                node = None
                if self.ponder:
                    try:
                        node = self._reuse_node(eng, prev[1], board)
                    except Exception as e:  # noqa: BLE001
                        print("reuse lookup failed:", repr(e), file=LOG)
                        node = None
                    if node is not None:
                        eng.reuse = True
                        eng._reused_root = lambda _b, _n=node: _n
                        self.stats["reused"] = self.stats.get("reused", 0) + 1
                    else:
                        self.stats["reuse_miss"] = self.stats.get("reuse_miss", 0) + 1
                try:
                    sec = min(self.seconds, left)
                    bud = Budget(seconds=sec) if not self.soft_seconds else \
                        Budget(seconds=sec, soft=min(1.0, self.soft_seconds / max(sec, 1e-3)))
                    box["m"] = eng.choose(board.copy(), bud)
                    if self.diversify > 0 and box["m"] is not None:
                        try:
                            box["m"] = self._near_best(eng, box["m"], g)
                        except Exception as e:  # noqa: BLE001
                            print("near-best failed:", repr(e), file=LOG)
                finally:
                    if node is not None:
                        eng.reuse = False
                        eng.__dict__.pop("_reused_root", None)
                        eng._prev = None
                    self.busy.discard(id(eng))
                    box["searched"] = True
            except Exception as e:  # noqa: BLE001
                box["err"] = repr(e)
            finally:
                with self.glock:
                    self.live.pop(me_t, None)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():                      # join in slices: an abandoned call can be killed meanwhile
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            th.join(min(0.1, left))
            if th.is_alive():
                try:
                    self._check_stale_bridge()
                except Exception:  # noqa: BLE001
                    pass
        answer = legal[0]
        if th.is_alive():
            cancel.set()
            self.stats["overrun"] += 1
        if "err" in box:
            self.stats["errors"] += 1
            print("bot error:", box["err"], file=LOG)
        m = box.get("m")
        ranked = []
        try:
            ranked = self._ranked(box.get("eng"))
        except Exception as e:  # noqa: BLE001
            print("ranking failed:", repr(e), file=LOG)
        if m is None and ranked and len(legal) > 1:  # search overran or raised: its provisional best, not legal[0]
            m = ranked[0]
            self.stats["provisional"] = self.stats.get("provisional", 0) + 1
        if m is not None and "board" in box:
            try:
                m = self._avoid_noop(box["board"], m, box["ms"], deadline, ranked)
            except Exception as e:  # noqa: BLE001
                print("noop check failed:", repr(e), file=LOG)
        if m is not None and move_array(m) not in legal:
            # the board's moves differ from legalMoves (an inexact rebuild): best searched move that IS legal
            alt = next((x for x in ranked if move_array(x) in legal), None)
            if alt is not None:
                m = alt
                self.stats["ranked_legal"] = self.stats.get("ranked_legal", 0) + 1
        if m is not None and move_array(m) in legal:
            answer = move_array(m)
        elif len(legal) > 1:
            self.stats["fallback"] += 1
        if avoid and tuple(answer) in avoid and len(legal) > 1:
            # the same position keeps coming back unchanged (the referee dropped our moves): a move not tried yet
            # for it, skipping known no-ops; when every legal move has been tried, start again from legal[0]
            # (11 Nov 2026 review 3: avoiding only the last answer alternated two dropped moves forever)
            board = box.get("board")

            def known_noop(x):
                try:
                    cache = getattr(board, "_kids", None) if board is not None else None
                    kids = cache.get(x) if cache is not None else None
                    return bool(kids) and all(c.hash == board.hash and c.turn == board.turn for _o, _p, c in kids)
                except Exception:  # noqa: BLE001
                    return False
            bymove = {tuple(move_array(x)): x for x in (box.get("ms") or [])}
            alt = next((move_array(x) for x in ranked if move_array(x) in legal
                        and tuple(move_array(x)) not in avoid and not known_noop(x)), None)
            if alt is None:
                alt = next((x for x in legal if tuple(x) not in avoid
                            and not (tuple(x) in bymove and known_noop(bymove[tuple(x)]))), None)
            if alt is None:
                alt = legal[0]
                g.tried_reset = True
            answer = list(alt)
            self.stats["forced_change"] = self.stats.get("forced_change", 0) + 1
        if "board" in box and "ms" in box:       # tracking finished (even if the search overran): keep the state
            arrays = [move_array(x) for x in box["ms"]]
            if answer in arrays:
                sent = box["ms"][arrays.index(answer)]
                try:
                    kept = self._pruned(box["board"], sent)
                except Exception as e:  # noqa: BLE001
                    print("prune failed:", repr(e), file=LOG)
                    kept = box["board"]
                with self.glock:
                    g.board, g.sent, g.pruned = kept, sent, False
        self.stats["max_ms"] = max(self.stats["max_ms"], (time.perf_counter() - t0) * 1000)
        if self.ponder and box.get("searched") and not th.is_alive() and box.get("eng") is not None:
            try:
                self._start_ponder(key, box["eng"], answer)
            except Exception as e:  # noqa: BLE001
                print("ponder start failed:", repr(e), file=LOG)
        return answer

    # ---- pondering (13 Nov 2026) -------------------------------------------------------------------------------
    def _start_ponder(self, key, eng, answer) -> None:
        root = getattr(eng, "_spent", None)
        if root is None or not root.edges:
            return
        edge = next((e for mv, e in root.edges.items() if move_array(mv) == list(answer)), None)
        if edge is None:
            return
        stop = threading.Event()

        def run():
            it = 0
            try:
                eng._budget = Budget(seconds=PONDER_MAX_S)
                end = time.perf_counter() + PONDER_MAX_S
                while not stop.is_set() and it < PONDER_MAX_ITERS and time.perf_counter() < end:
                    eng._budget.tick()
                    v = eng._visit_edge(root, edge)
                    root.N += 1
                    root.W += v
                    it += 1
            except Exception as e:  # noqa: BLE001
                self.stats["ponder_err"] = self.stats.get("ponder_err", 0) + 1
                if self.stats["ponder_err"] <= 3:
                    print("ponder error:", repr(e), file=LOG)
            finally:
                self.stats["ponder_iters"] = self.stats.get("ponder_iters", 0) + it
                with self.glock:
                    self.busy.discard(id(eng))

        th = threading.Thread(target=run, daemon=True)
        with self.glock:
            self.busy.add(id(eng))                # a request must not search this engine while it ponders
            self._ponds[key] = (th, stop, eng)
        th.start()

    def _stop_ponders(self) -> None:
        """Stop every ponder (the arena plays one game at a time; a new request means the turn is ours)."""
        with self.glock:
            ps = list(self._ponds.values())
            self._ponds.clear()
        for _th, stop, _e in ps:
            stop.set()
        for th, _stop, _e in ps:
            th.join(PONDER_JOIN_S)
            if th.is_alive():
                self.stats["ponder_stuck"] = self.stats.get("ponder_stuck", 0) + 1

    @staticmethod
    def _reuse_node(eng, sent, board):
        """The node of the previous search tree for ``board``: under the edge of the move we sent, an outcome,
        then (unless it is our turn again) an opponent move and its outcome.  Same referee hash and turn."""
        root = getattr(eng, "_spent", None)
        if root is None or sent is None or not root.edges:
            return None
        want = move_array(sent)
        edge = next((e for mv, e in root.edges.items() if move_array(mv) == want), None)
        if edge is None:
            return None
        h, turn = board.hash, board.turn

        def ok(n):
            return n is not None and n.N > 0 and n.board is not None and n.turn == turn and n.board.hash == h
        for n1 in list(edge.children.values()):
            if ok(n1):
                return n1
            for e2 in list(n1.edges.values()):
                for n2 in list(e2.children.values()):
                    if ok(n2):
                        return n2
        return None


def make_server(bot: CheqqersBot, port: int, host: str = "127.0.0.1", log_path: str | None = None):
    """The HTTP server.  Python's garbage collector stays automatic.  (11 Nov 2026: switching gen-2 off and
    collecting after each answer instead was tried and withdrawn - on a loaded host that collection took up to
    1.7 s and delayed the accept of the NEXT request, which our own timer cannot see: 4 referee timeouts in 36
    self-play games.  Automatic GC: 0 forfeits in 1,000 self-play games at load 16, worst answer 2.38 s.)
    ``main`` freezes the start-up objects so that every collection is shorter."""
    gc.enable()
    gc.set_threshold(700, 10, 10)
    logf = open(log_path, "a") if log_path else None

    class H(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            body = json.dumps({"ok": True, "stats": bot.stats}).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            """The request body: Content-Length, or Transfer-Encoding: chunked (11 Nov 2026 review: a chunked
            POST was read as empty and answered with an empty move, a forfeit)."""
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                parts = []
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass                  # trailers
                        return b"".join(parts)
                    parts.append(self.rfile.read(size))
                    self.rfile.readline()         # CRLF after the chunk
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def do_POST(self):
            t0 = time.perf_counter()
            try:
                pl = json.loads(self._body().decode())
                legal = pl.get("legalMoves") or []
                try:
                    mv = bot.choose(pl, t0)
                except Exception as e:  # noqa: BLE001
                    print("choose failed:", repr(e), file=LOG)
                    mv = list(legal[0]) if legal else []
            except Exception as e:  # noqa: BLE001
                print("bad request:", repr(e), file=LOG)
                pl, mv = {}, []
            body = json.dumps({"move": mv}).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            if logf:
                logf.write(json.dumps({"t": time.time(), "ms": ms,
                                       "game": pl.get("gameId"), "round": pl.get("roundId"),
                                       "player": pl.get("player"), "n_legal": len(pl.get("legalMoves") or []),
                                       "move": mv}) + "\n")
                logf.flush()

        def log_message(self, *a):
            pass

    class S(ThreadingHTTPServer):
        daemon_threads = True

    return S((host, port), H)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 by default; use 0.0.0.0 only when the tournament server must reach this machine")
    ap.add_argument("--seconds", type=float, default=1.5, help="search time per move")
    ap.add_argument("--log", default=None, help="append one JSON line per request")
    ap.add_argument("--eval-depth", type=int, default=2, help="classical search depth of the leaf evaluator")
    ap.add_argument("--anabi", default="{}", help="JSON overrides of Anabi's search config (experiments only)")
    ap.add_argument("--book", default=None, help="opening book (cqbook-1 JSON); default none")
    ap.add_argument("--diversify", type=float, default=DIVERSIFY,
                    help="near-best choice margin in mean value (0 = off: deterministic, as on 12 Nov)")
    ap.add_argument("--ponder", action="store_true", help="think on during the opponent's turn and reuse that subtree")
    ap.add_argument("--soft-seconds", type=float, default=None, help="soft budget with --anabi '{\"final\": \"robust\"}' (experiments only)")
    a = ap.parse_args(argv)
    bot = CheqqersBot(seconds=a.seconds, eval_depth=a.eval_depth, anabi=json.loads(a.anabi),
                      book=load_book(a.book) if a.book else None, soft_seconds=a.soft_seconds, diversify=a.diversify,
                      ponder=a.ponder)
    srv = make_server(bot, a.port, a.host, a.log)
    try:                                          # warm-up: start the Node bridge before the first request
        w = CQWebBoard.new(6, 2, 2)
        w.children(w.legal_moves()[0])
        CQWebBoard.from_payload(w.payload(3000))
    except Exception as e:  # noqa: BLE001
        print("warm-up failed (the first request starts the bridge):", repr(e), file=LOG)
    gc.collect()
    gc.freeze()                                   # imported modules and the warm-up stay out of later collections
    print(f"Anabi Cheqqers bot on http://{a.host}:{a.port}/move ({a.seconds} s per move)", file=LOG, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
