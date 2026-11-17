"""Cheqqers on its *tournament* rules (cheqqers-ai's home rules), through the referee's own code.

Rules = the 1st European Tournament in Quantum Games' Cheqqers referee (Maastricht, 20-21 June 2026;
github.com/VincenzoLipardi/ETQG_2026 ``Cheqqers/local-cheqqers-server.js``, game logic lines 1-906 unmodified) on
Evert van Nieuwenburg's ``@quantumplayed/quantum-game-engine`` 1.0.3, run offline in Node through
``reports/engines/anabi_other_games/cqweb/cqweb_bridge.mjs`` (Krishna allowed both, 26 October 2026).  This is
the same idea as ``qchess.engines.bfrules`` for Bee-Flow: no port, so Anabi's model and the referee are the same
code.  The bridge stores game states, deep-clones them and enumerates measurement outcomes by forcing them through
the game's own ``measureSquare(index, forced)``.

Differences from the paper rules (``cheqqers.py``): men capture backwards, kings fly (move and capture at any
distance), a multi-jump continues with the same piece only, the longest capture is compulsory, a piece is measured
only when it is a move's landing square (level 2+), entangled pieces that cannot coexist cannot be captured
together, wins by probability mass < 0.0001, draws by threefold repetition / 25 king moves / 50 moves without a
capture.  Square 0 is the top-left dark square (Black's side), White starts on the high indices and moves up.

    CQWebBoard.new(6, 2, 2)     # the tournament setting: 6x6, two rows, quantumness 2
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import weakref
import zlib
from collections import Counter
from itertools import product as _product

import numpy as np

from .cheqqers import ClassicalCheckers
from .core import BLACK, WHITE, GameBoard

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BRIDGE = os.path.join(_HERE, "..", "..", "reports", "engines", "anabi_other_games", "cqweb", "cqweb_bridge.mjs")
DEFAULT_GAME = os.path.expanduser("~/Documents/qchess-external/games/etqg-cheqqers/tournament_game.mjs")


class _Bridge:
    def __init__(self):
        bridge = os.environ.get("AOG_CQWEB_BRIDGE", DEFAULT_BRIDGE)
        game = os.environ.get("AOG_CQWEB_GAME", DEFAULT_GAME)
        node = os.environ.get("AOG_NODE", "node")
        self.p = subprocess.Popen([node, bridge, game], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.refs: Counter = Counter()
        self.dead: list = []
        self.pid = os.getpid()
        self.broken = False
        self.call_started = None                  # monotonic start of the call in progress (None: idle)
        self.call_owner = None                    # thread ident of the caller of that call

    def busy_for(self) -> float:
        t = self.call_started
        return 0.0 if t is None else time.monotonic() - t

    def kill(self) -> None:
        """Give up on this Node process (stuck or dead): later calls on it raise; ``bridge()`` starts a new one."""
        self.broken = True
        try:
            self.p.kill()
        except Exception:  # noqa: BLE001
            pass

    def call(self, obj: dict) -> dict:
        if self.broken:
            raise RuntimeError("cqweb bridge died")
        with self.lock:
            self.call_owner = threading.get_ident()
            self.call_started = time.monotonic()
            try:
                if self.dead:
                    rel, self.dead = self.dead, []
                    self.p.stdin.write(json.dumps({"cmd": "release", "ids": rel}) + "\n")
                    self.p.stdin.flush()
                    self.p.stdout.readline()
                self.p.stdin.write(json.dumps(obj) + "\n")
                self.p.stdin.flush()
                line = self.p.stdout.readline()
            except (OSError, ValueError):
                line = ""
            finally:
                self.call_started = None
                self.call_owner = None
        if not line:
            self.broken = True
            raise RuntimeError("cqweb bridge died")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def incref(self, i):
        self.refs[i] += 1

    def decref(self, i):
        self.refs[i] -= 1
        if self.refs[i] <= 0:
            del self.refs[i]
            self.dead.append(i)


_BRIDGE: _Bridge | None = None


def bridge() -> _Bridge:
    global _BRIDGE
    if _BRIDGE is None or _BRIDGE.pid != os.getpid() or _BRIDGE.broken or _BRIDGE.p.poll() is not None:
        _BRIDGE = _Bridge()
    return _BRIDGE


def _mv(a) -> tuple:
    return (a[0], bool(a[1]), a[2], a[3]) if a[0] == "m" else tuple(a)


class CQWebBoard(GameBoard):
    rules_variant = "cheqqers-web"

    def __init__(self, info: dict, size: int, rows: int, level: int, history=None, br: "_Bridge | None" = None):
        self.id = info["id"]
        self.size, self.rows, self.level = size, rows, level
        self.nsq = size * size // 2
        self.turn = info["turn"]
        self._result = info["result"]
        self._moves = [_mv(a) for a in info["moves"]] if "moves" in info else None   # None: fetched when needed
        self._groups = info.get("groups")         # measurement groups, if the bridge sent them with the state
        self.multi = info["multi"]
        self.since_take = info["since_take"]
        self.hash = info["hash"]
        self.history = list(history or [])
        self.rng = np.random.default_rng(0)
        self._kids: dict = {}
        b = self._br = br if br is not None else bridge()   # the Node process holding this state
        b.incref(self.id)
        weakref.finalize(self, b.decref, self.id)

    @classmethod
    def new(cls, size: int = 6, rows: int = 2, level: int = 2) -> "CQWebBoard":
        br = bridge()
        return cls(br.call({"cmd": "new", "size": size, "rows": rows, "type": level}), size, rows, level, br=br)

    @classmethod
    def from_payload(cls, payload: dict) -> "CQWebBoard":
        """Rebuild a board from an AI-tournament request (needs the kit bridge, ``AOG_CQWEB_BRIDGE`` =
        reports/engines/evert_kit_20261107/cheqqers/cqweb_bridge_kit.mjs).  ``.payload_match`` says whether the
        rebuilt move list equals the request's ``legalMoves`` (same order)."""
        br = bridge()
        r = br.call({"cmd": "from_payload", "payload": payload})
        s = payload["settings"]
        b = cls(r, int(s["boardSize"]), 0, int(s["quantumness"]), br=br)
        b.payload_match = bool(r["match"])
        return b

    def copy(self) -> "CQWebBoard":
        c = CQWebBoard.__new__(CQWebBoard)
        c.__dict__.update(self.__dict__)
        c.history = list(self.history)
        c._kids = self._kids                      # children are immutable states: safe to share
        self._br.incref(c.id)
        weakref.finalize(c, self._br.decref, c.id)
        return c

    def result(self):
        return self._result

    #: Expose only the first move per serialised array.  At quantumness 3 a split [a, b, c] and a merge [a, b, c]
    #: can both be legal and serialise alike; the AI-tournament referee maps an answer back with findIndex, so only
    #: the first can ever be played (11 Nov 2026 review).  The full list stays in ``_moves`` (bridge indices).
    dedupe_arrays = False

    def _all_moves(self):
        if self._moves is None:                   # fast bridge: a child's moves are fetched on first use
            self._moves = [_mv(a) for a in self._br.call({"cmd": "moves", "id": self.id})["moves"]]
        return self._moves

    def _playable(self):
        self._all_moves()
        if not self.dedupe_arrays:
            return list(self._moves)
        seen, out = set(), []
        for m in self._moves:
            a = (m[2], m[3]) if m[0] == "m" else (m[1], m[2], m[3])
            if a not in seen:
                seen.add(a)
                out.append(m)
        return out

    def legal_moves(self, allow_quantum: bool = True):
        ms = self._playable()
        if not allow_quantum:
            cl = [m for m in ms if m[0] == "m"]
            return cl or ms
        return ms

    def _children(self, m):
        k = self._kids.get(m)
        if k is None:
            mi = self._all_moves().index(m)
            r = self._br.call({"cmd": "children", "id": self.id, "mi": mi})["children"]
            k = []
            for c in r:
                o = tuple(c["o"]) if c["o"] else None
                k.append((o, c["p"], CQWebBoard(c, self.size, self.rows, self.level, self.history + [m], br=self._br)))
            self._kids[m] = k
        return k

    def children(self, m):
        return self._children(m)

    def outcome_probs(self, m):
        return [(o, p) for o, p, _ in self._children(m)]

    def apply(self, m, outcome=None):
        for o, p, b in self._children(m):
            if o == (tuple(outcome) if outcome is not None else None):
                return b
        raise ValueError(f"outcome {outcome} not possible for {m}")

    def groups_and_pieces(self):
        if self._groups is not None:
            return self._groups
        self._groups = self._br.call({"cmd": "groups", "id": self.id})
        return self._groups

    def branches(self):
        """Classical boards in the superposition: [(p, key)], key = per square None or (colour, crowned)."""
        r = self.groups_and_pieces()
        pieces = [None if x is None else (x[0], bool(x[1])) for x in r["pieces"]]
        gs = [[(d[0], set(d[1])) for d in g["dist"]] for g in r["groups"]]
        sqs = [set(g["squares"]) for g in r["groups"]]
        if not gs:
            return [(1.0, tuple(pieces))]
        size = 1
        for g in gs:
            size *= len(g)
        if size > 4096:
            rng = np.random.default_rng(zlib.crc32(self.hash.encode()))
            cnt = Counter(tuple(int(rng.choice(len(g), p=[x[0] for x in g])) for g in gs) for _ in range(64))
            items = [(c / 64.0, idx) for idx, c in cnt.items()]
        else:
            items = []
            for idx in _product(*[range(len(g)) for g in gs]):
                p = 1.0
                for g, i in zip(gs, idx):
                    p *= g[i][0]
                if p > 1e-12:
                    items.append((p, idx))
        out = []
        for p, idx in items:
            b = list(pieces)
            for g, s, i in zip(gs, sqs, idx):
                occ = g[i][1]
                for x in s:
                    if x not in occ:
                        b[x] = None
            out.append((p, tuple(b)))
        return out

    def payload(self, timeout_ms: int) -> dict:
        return self._br.call({"cmd": "payload", "id": self.id, "timeout": timeout_ms})["payload"]

    def summary(self) -> dict:
        return {"hash": self.hash, "turn": self.turn, "result": self._result}


class ClassicalCheckersWeb(ClassicalCheckers):
    """The classical engine per branch, on the tournament's classical rules: men move forward one step and
    capture in all four directions, kings fly; a capture chain continues with the same piece (not after crowning);
    the longest-capture rule is not applied (an evaluator, not a referee)."""

    def __init__(self, size: int, depth: int = 2, qdepth: int = 6, scale: float = 3.0):
        self.size, self.depth, self.qdepth, self.scale = size, depth, qdepth, scale
        spr = size // 2
        self.nsq = size * spr
        self.ixy, self.xyi = {}, {}
        for i in range(self.nsq):
            row = size - 1 - i // spr
            col = (i % spr) * 2 + (row % 2)
            self.ixy[i] = (col, row)
            self.xyi[(col, row)] = i
        self.cache: dict = {}
        self.nodes = 0

    def moves(self, sq, color, only_from=None):
        takes, plain = [], []
        fwd = 1 if color == WHITE else -1
        for i in range(self.nsq):
            p = sq[i]
            if p is None or p[0] != color or (only_from is not None and i != only_from):
                continue
            x, y = self.ixy[i]
            king = p[1]
            for dx, dy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
                cx, cy, enemy = x, y, None
                while True:
                    cx += dx
                    cy += dy
                    j = self.xyi.get((cx, cy))
                    if j is None:
                        break
                    q = sq[j]
                    if enemy is None:
                        if q is None:
                            if dy == fwd or king:
                                if only_from is None:
                                    plain.append((i, j, None))
                            if not king:
                                break
                        elif q[0] == color:
                            break
                        else:
                            enemy = j
                    else:
                        if q is None:
                            takes.append((i, j, enemy))
                            if not king:
                                break
                        else:
                            break
        return takes if takes else ([] if only_from is not None else plain)

    def play(self, sq, color, mv):
        f, t, k = mv
        b = list(sq)
        p = b[f]
        b[f] = None
        if k is not None:
            b[k] = None
        y = self.ixy[t][1]
        last = self.size - 1 if color == WHITE else 0
        crowned = not p[1] and y == last
        if crowned:
            p = (p[0], True)
        b[t] = p
        b = tuple(b)
        nxt = BLACK if color == WHITE else WHITE
        if k is not None and not crowned and self.moves(b, color, only_from=t):
            nxt = color
        return b, nxt


__all__ = ["CQWebBoard", "ClassicalCheckersWeb", "bridge"]
