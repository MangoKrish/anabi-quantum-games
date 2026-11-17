"""Shared layer: one Anabi for every quantum game.

Anabi's search (``qchess.engines.mcts.MCTSEngine`` with the config ``qchess.engines.anabi.ANABI``: PUCT, exact
Born averaging at measurement nodes, a one-ply minimax step at the leaves, most-visited move, 0.5 s per move)
already searches boards of foreign rule sets: ``qchess.engines.base`` dispatches ``legal_moves`` /
``outcome_probs`` / ``apply`` to the board when ``board.rules_variant`` is set (that is how Anabi played the
Deep Dirac, Andreas-Erick, Bee-Flow, qhorgues and official-SDK rules).  So a new game needs only

1. a board class implementing ``GameBoard`` below (rules), and
2. an evaluator ``f(board, me) -> value in [-1, 1]``.

``make_anabi_for`` builds Anabi on those two pieces with ``anabi_foreign.ForeignMCTS`` (= ``MCTSEngine``
unchanged except a rules-agnostic zero-iteration fallback) and Anabi's exact config.  Nothing of the search
is copied or edited here.

The evaluator is Anabi's evaluator idea with the game swapped: ``BranchAveragedEvaluator`` takes the
classical boards in the superposition with their Born weights, values each with a *classical* engine for the
game (for chess that engine is Fairy-Stockfish depth 8), and averages - exactly over at most ``exact_max`` = 16
branches, otherwise over ``samples`` = 8 branches drawn with common random numbers keyed on the position
(the rule of ``evaluators.FSFEvaluator._select``).  Classical values use Anabi's scale: forced results
(a win or loss the classical engine can see) in the band 0.98 - 0.0025 per ply, heuristic scores as
0.85 tanh(score / scale).

Board protocol (``GameBoard``): ``turn`` ("w" moves first / "b"), ``history`` (list, one entry per ply: the
MCTS reads the ply count from it), ``rng`` (numpy Generator; the MCTS clones boards without advancing it),
``copy()``, ``legal_moves(allow_quantum=True)``, ``outcome_probs(m) -> [(outcome, p)]`` (``[(None, 1.0)]``
for a deterministic move), ``apply(m, outcome) -> new board``, ``result() -> "white" | "black" | "draw" |
None``, ``branches() -> [(weight, key)]`` and ``king_prob(colour)`` (1.0: no kings; only used for positions
with no legal move, which these games resolve in ``result``).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..engines.anabi import ANABI, ANABI_SECONDS
from ..engines.anabi_foreign import ForeignMCTS
from ..engines.base import Budget

WHITE, BLACK = "w", "b"
HEUR_MAX = 0.85          # as qchess.engines.evaluators: |heuristic| <= 0.85
FORCED_MAX = 0.98        # forced results: 0.98 - 0.0025 per ply
FORCED_STEP = 0.0025


def other(c: str) -> str:
    return BLACK if c == WHITE else WHITE


def forced_value(win: bool, plies: int) -> float:
    """Anabi's forced band: a win in ``plies`` plies -> 0.98 - 0.0025 * plies (a loss: the negative)."""
    v = FORCED_MAX - FORCED_STEP * min(max(plies, 0), 20)
    return v if win else -v


def heuristic_value(score: float, scale: float) -> float:
    return HEUR_MAX * math.tanh(score / scale)


class GameBoard:
    """Base class of the game boards (see the module docstring for the protocol)."""

    rules_variant = "game"

    def king_prob(self, color: str) -> float:
        return 1.0

    def push(self, m, u: float):
        """Referee step: play ``m``; for a measuring move pick the outcome whose cumulative probability
        interval contains ``u`` (``u`` uniform in [0, 1)).  Returns ``(new_board, outcome, p)``."""
        outs = self.outcome_probs(m)
        acc = 0.0
        pick = outs[-1]
        for o, p in outs:
            acc += p
            if u < acc:
                pick = (o, p)
                break
        return self.apply(m, pick[0]), pick[0], pick[1]


# ---- evaluator ----------------------------------------------------------------------------------------
class BranchAveragedEvaluator:
    """Anabi's evaluator with the classical engine swapped: Born-weighted mean of ``classical(board, key, me)``
    over the classical branches, exact up to ``exact_max`` branches, else ``samples`` CRN draws."""

    def __init__(self, classical, exact_max: int = 16, samples: int = 8, seed: int = 0, name: str = "branch-avg"):
        self.classical = classical
        self.exact_max = int(exact_max)
        self.samples = int(samples)
        self.seed = int(seed)
        self.name = name
        self.counters: Counter = Counter()

    def __call__(self, board, me: str) -> float:
        self.counters["calls"] += 1
        br = board.branches()
        if len(br) > self.exact_max:
            self.counters["sampled_positions"] += 1
            mass = sum(w for w, _ in br)
            p = np.array([w for w, _ in br]) / mass
            key = f"{self.seed}|{board.turn}|{sorted(repr(k) for _, k in br)}"
            rng = np.random.default_rng(zlib.crc32(key.encode()))
            idx = rng.choice(len(br), size=self.samples, p=p)
            cnt = Counter(int(i) for i in idx)
            br = [(mass * c / self.samples, br[i][1]) for i, c in sorted(cnt.items())]
        tot = sum(w for w, _ in br)
        self.counters["branches"] += len(br)
        return sum(w * self.classical(board, k, me) for w, k in br) / tot

    def close(self) -> None:
        pass


# ---- Anabi -----------------------------------------------------------------------------------------------
def make_anabi_for(evaluator, name: str = "Anabi", **overrides) -> ForeignMCTS:
    """Anabi's exact search config (``ANABI``: puct, c 1.0, exact chance nodes, minimax1 leaf, visits) on a
    game evaluator.  ``overrides`` replace top-level config keys (used only for ablations)."""
    cfg = {**ANABI, **overrides, "name": name}     # cfg["eval"] is ignored: the evaluator object is passed
    eng = ForeignMCTS(cfg, evaluator=evaluator)
    eng.name = name
    return eng


# ---- seeds -----------------------------------------------------------------------------------------------
def fnv1a32(s: str) -> int:
    """FNV-1a 32-bit hash (stable across Python versions)."""
    h = 0x811C9DC5
    for b in s.encode():
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def stream_seed(base: int, unit: str, stream: str | None = None) -> int:
    return fnv1a32(f"{base}|{unit}" if stream is None else f"{base}|{unit}|{stream}")


# ---- match loop -------------------------------------------------------------------------------------------
@dataclass
class GameRecord:
    game_id: str
    white: str
    black: str
    seed: int
    moves: list = field(default_factory=list)       # [json-able move, outcome, p, seconds]
    result: str | None = None                         # "white" | "black" | "draw"
    ending: str = ""
    plies: int = 0
    think: dict = field(default_factory=lambda: {"w": [], "b": []})
    extra: dict = field(default_factory=dict)

    def score_white(self) -> float:
        return {"white": 1.0, "black": 0.0, "draw": 0.5}[self.result]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=_jsonable)


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (tuple, frozenset, set)):
        return list(x)
    return repr(x)


def play_game(board, white, black, *, seed: int, game_id: str, seconds: dict | float = ANABI_SECONDS,
              ply_cap: int = 200, move_to_json=lambda m: m, log_states: bool = False) -> GameRecord:
    """Play one game.  ``white`` / ``black`` follow ``qchess.engines.base.Engine`` (``new_game``, ``choose``,
    ``observe``, ``close`` optional).  Measurement outcomes come from the referee stream
    ``stream_seed(seed, game_id-without-colour, "referee")`` so both colours of a start share it.  Each
    engine gets ``Budget(seconds)``; ``seconds`` may be a dict ``{"w": s, "b": s}``.  A move outside the legal
    list or an exception forfeits the game for the mover (recorded in ``ending``)."""
    secs = seconds if isinstance(seconds, dict) else {WHITE: seconds, BLACK: seconds}
    unit = game_id.rsplit("|", 1)[0]
    ref = np.random.default_rng(stream_seed(seed, unit, "referee"))
    rec = GameRecord(game_id, getattr(white, "name", "white"), getattr(black, "name", "black"), seed)
    players = {WHITE: white, BLACK: black}
    for c, e in players.items():
        e.new_game(c, stream_seed(seed, game_id, f"engine-{c}"))
    try:
        while True:
            r = board.result()
            if r is not None:
                rec.result, rec.ending = r, rec.ending or "rules"
                break
            if len(board.history) >= ply_cap:
                rec.result, rec.ending = "draw", "ply_cap"
                break
            c = board.turn
            eng = players[c]
            legal = board.legal_moves()
            b = Budget(seconds=secs[c])
            t0 = time.perf_counter()
            try:
                m = eng.choose(board.copy_for_engine() if hasattr(board, "copy_for_engine") else board.copy(), b)
            except Exception as exc:                    # crash = forfeit
                rec.result, rec.ending = ("black" if c == WHITE else "white"), f"crash:{type(exc).__name__}:{exc}"
                break
            dt = time.perf_counter() - t0
            if m not in legal:
                rec.result, rec.ending = ("black" if c == WHITE else "white"), f"illegal:{m!r}"
                break
            u = float(ref.random())
            before = board
            board, o, p = board.push(m, u)
            rec.moves.append([move_to_json(m), o if not isinstance(o, tuple) else list(o), round(p, 12), round(dt, 4)])
            rec.think[c].append(round(dt, 4))
            for e in players.values():
                f = getattr(e, "observe", None)
                if f is not None:
                    f(before, m, o, board)
    finally:
        pass
    rec.plies = len(board.history)
    rec.extra["final"] = board.summary() if hasattr(board, "summary") else None
    for c, e in players.items():
        st = getattr(e, "stats", None)
        if st is not None:
            try:
                s = st()
                rec.extra[f"stats_{c}"] = {k: s[k] for k in ("searches", "iterations", "aborted", "time",
                                                               "foreign_fallback") if k in s}
            except Exception:
                pass
    return rec


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


__all__ = ["GameBoard", "BranchAveragedEvaluator", "make_anabi_for", "play_game", "GameRecord", "stream_seed",
           "forced_value", "heuristic_value", "other", "WHITE", "BLACK", "config_hash"]
