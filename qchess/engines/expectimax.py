"""Deep expectiminimax search for Cantwell quantum chess (arXiv:1906.05836 rules).

The game tree has three kinds of node:

* **my decision node** (``board.turn == me``): value = max over my legal moves;
* **opponent decision node**: value = the opponent model of ``qchess.engines.gametheory`` applied to
  the replies (``minimax`` = min, ``eps`` = ``(1-eps) min + eps mean``, ``softmax`` = quantal
  response);
* **chance node**: a measuring move (capture, blocked move).  Its outcomes are the two results of
  the paper's M1 measurement, expanded exactly with their Born probabilities
  (``base.chance_children``), or sampled (``"chance": "sample"``).  A unitary move (jump, slide,
  split, merge) is a chance node with a single outcome of probability 1.

Leaves are scored by an evaluator (``qchess.engines.evaluators``); a board on which a king has
probability 0 is terminal (Rule 10) and scores +1 / -1 / 0.  Every leaf value (terminal or
evaluator) found ``ply`` plies below the root is multiplied by ``1 - discount * ply``, so faster
wins and slower losses are preferred and a depth-1 search equals ``GreedyEngine`` (which multiplies
every child value by ``1 - ply_discount``).  A non-terminal position with no legal move scores
``base.king_balance`` (0 on a classical board, as stalemate = draw in the E0 tablebase).

Moves and children come from ``qchess.engines.fastgen`` (``fast_legal_moves`` / ``fast_is_legal`` /
``fast_apply``, bitwise-identical to the board's own methods but without rebuilding Qiskit
controlled gates on every slide and pawn capture).  ``"fast": false`` is the reference path: legal
moves from ``board.legal_moves``, legality from ``board.is_legal``, and every child built as
``board.copy()`` + ``ForcedRNG(outcome)`` + ``board.push(m)`` (no fastgen call at all).

Pruning (``"prune"``):

* ``"star1"`` (default): fail-soft alpha-beta at my nodes and at minimax opponent nodes, plus
  Ballard's Star1 bounds at chance nodes (B. W. Ballard, "The *-minimax search procedure for trees
  containing chance nodes", Artificial Intelligence 21, 1983; see also Hauk, Buro & Schaeffer,
  "Rediscovering *-minimax search", CG 2004).  With value bounds ``L = -1, U = 1`` the window
  handed to outcome ``i`` of a chance node with probabilities ``p_j`` is
  ``alpha_i = (alpha - sum_{j<i} p_j v_j - U sum_{j>i} p_j) / p_i`` and
  ``beta_i = (beta - sum_{j<i} p_j v_j - L sum_{j>i} p_j) / p_i``; the node fails low/high as
  soon as one outcome falls outside its window.
* ``"alphabeta"``: alpha-beta at decision nodes only; windows do not cross chance nodes with more
  than one outcome.
* ``"none"``: plain expectiminimax (the reference the other two must reproduce exactly).

Opponent models other than ``minimax`` are not alpha-beta-sound at opponent nodes, so those nodes
search every reply with a full window and return an exact value; pruning still happens at my
nodes and (with ``star1``) at chance nodes.

The root always searches every legal root move; with ``"root": "exact"`` (default) each one with
a full window, so every root ``Candidate`` holds an exact backed-up value and the decision rules
(``maximin``, ``meanstd``, ``cvar``, ``softmax``) see true values.  ``"root": "pvs"`` searches root
moves after the first with a window ``(best so far, 1)`` (much cheaper), and ``"auto"`` uses
``pvs`` when the decision rule is ``argmax`` or ``visits`` and ``exact`` otherwise.  Under ``pvs``
a move that fails low *strictly* below the best value so far (by more than 1e-9) is an upper-bound
candidate: ``extra["bound"] = "upper"``, ``mean`` = the bound, ``worst`` = -1 (unknown), ``std``
= 0.  A move that ties the best value, or whose root chance node failed high on an outcome (so
its per-outcome record is incomplete), is re-searched with full windows, so every candidate
without ``"bound"`` is exact and the best move is always exact.  ``choose`` decides among the
exact candidates only; callers of ``analyse`` must skip (or treat as worse) the bound ones.

Memory: the transposition table, child, move, capture and evaluation caches live for one
``analyse`` call only (each capped at insertion), so an engine holds no search memory between
moves.  Kept across moves they saved about 0.5% of the evaluator calls and no time, and their
growth (up to ~0.6 GB per engine) made Python's full garbage collections long enough to overrun
the move budget.  ``analyse`` starts with empty caches and, on return, only moves the full ones
aside (fresh empty dicts take their place, which costs nothing); they are deallocated in
``observe``, which the arena calls after every move outside the move clock (wrappers forward it),
or else at the start of the next search, inside its budget.  Freeing them on return took 2-3 ms
at a 0.5 s clock and 13-19 ms at 2 s (about 11k TT entries and boards), all of it overrun once
``{"soft": 0.85, "id_growth": 0}`` makes every search end at the hard limit (review of 19 October
2026; ``MCTSEngine`` frees its tree in ``observe`` for the same reason).

Decision log: every ``choose`` appends its decision to ``stats()`` (one entry per ``choose`` call,
reset by ``new_game``; values rounded to 6 decimals).  ``MCTSEngine`` and ``GreedyEngine`` keep
the same log through ``decide_and_log`` / ``log_decision`` below.

* ``vstar``: the root value v*, the highest ``mean`` among the candidates the rule chose from
  (the exact ones: ``pvs`` upper bounds are left out, as in ``gametheory.adaptive_rank``);
* ``chosen_mean`` / ``chosen_worst`` / ``chosen_std``: the chosen candidate's statistics;
* under the ``adaptive`` rule, ``regimes``: the regime of each move (``"proven"``, ``"seek"``,
  ``"safe"``, ``"neutral"``, from ``gametheory.decide_adaptive``) and the counters
  ``regime_proven`` / ``regime_seek`` / ``regime_safe`` / ``regime_neutral``.  ``regimes``
  appears with the first adaptive decision and stays aligned with ``vstar``: a move decided by
  another rule (a wrapper such as ``bayes`` can switch an inner engine's rule between moves)
  has ``None`` there.

Every list holds ``None`` for a move played without a search, and such a move has no regime
(``None`` in ``regimes`` once that list exists, no counter): ``MCTSEngine`` when only one move is
legal, and the placeholder candidate (``extra["fallback"]``, ``mean`` 0.0, not a search value)
that ``ExpectimaxEngine`` returns when not one depth-1 root move finished within the budget and
``MCTSEngine`` when not one iteration did (``log_decision`` checks ``is_fallback``; review of 19
October 2026: they were logged as v* = 0 and counted as ``"neutral"``).  The log never changes
the choice: other rules still go through ``gametheory.decide`` with the same ``rng`` draws, and
the adaptive rule, which draws nothing, is taken from ``gametheory.decide_adaptive``, the
function ``decide`` itself calls for it.  So the log adds no draw to the engine ``rng``, which
the searches share (MCTS thompson selection, sampled chance, rollouts and the ``eps``
opponent; expectimax ``"chance": "sample"``): one extra draw would change every later search.

Wrapped engines.  ``bayes`` plays its inner engine's ``choose``, so the log is in its
``stats()["inner"]``, one entry per move (``bayes``'s own top-level ``regimes`` is its posterior
regime count, not this log).  ``trap`` keeps no decision log at any level: ``TrapEngine`` runs
its inner engine's ``analyse``, which logs nothing, and decides with ``gametheory.decide`` itself
without logging, so neither its ``stats()`` nor ``stats()["inner"]`` has per-move entries; the
same holds for the ``bayes(trap(...))`` and ``trap(bayes(...))`` stacks.  (The inner ``choose``
runs, and logs, only on trap's pass-through paths: an inner engine that cannot analyse, which
keeps no log anyway, or an analysis that came back empty.)

Arena hooks: ``set_game_info(info)`` (``ply_cap``, ``start_ply``, ``n``) is forwarded to the
evaluator (a ``tablebase`` evaluator needs the plies left; the ``fsf`` one starts its engine for
the game's board size before the clock runs).  ``new_game`` resets an evaluator the engine owns.
During ``analyse`` the evaluator gets ``set_budget(budget)`` (an engine-backed evaluator stops
with ``OutOfTime`` instead of starting a search after the budget expired).  ``analyse`` leaves the
caller's board untouched, including the state of ``board.rng``.

Config keys (all optional; defaults in brackets):

=================  ==============================================================================
``depth``          maximum plies [4]; iterative deepening from 1
``prune``          ``"star1"`` | ``"alphabeta"`` | ``"none"`` [``"star1"``]
``chance``         ``"exact"`` | ``"sample"`` [``"exact"``]; ``sample`` draws ``chance_samples``
                   outcomes (with replacement, Born weights) at every non-root chance node and
                   averages them; root moves are always expanded exactly
``chance_samples`` outcomes drawn per sampled chance node [1]
``tt``             transposition table on/off [True]
``tt_key``         ``"position"`` (``base.position_key``: occupancy distribution, phases ignored) |
                   ``"exact"`` (side, types and the full amplitude dict) [``"position"``]
``ordering``       ``"heuristic"`` | ``"eval"`` | ``"none"`` [``"heuristic"``]
``quiescence``     extra plies limited to capturing moves, with stand-pat [0]
``beam``           keep only the first k ordered moves at every non-root decision node [0 = all]
``eval``           evaluator config for ``make_evaluator`` [``{"kind": "material"}``]
``opponent``       opponent model [``{"model": "minimax"}``]: ``minimax`` | ``eps`` (``eps`` in [0, 1])
                   | ``softmax``; checked at construction and at the start of every search
``decide``         root decision rule [``{"rule": "argmax"}``], ``gametheory.check_decide``-ed at
                   construction and at the start of every search
``quantum``        allow my splits/merges [True]
``opp_quantum``    let the modelled opponent split/merge [True]
``root``           ``"exact"`` | ``"pvs"`` | ``"auto"`` [``"exact"``]
``discount``       per-ply discount of terminal values [0.001]
``id_growth``      do not start iteration d+1 if (time or nodes of iteration d) * id_growth exceeds
                   what is left of the budget [4.0, the median measured cost ratio of successive
                   iterations; 0 disables]
``soft``           fraction of the move budget (seconds, and nodes under a node cap) after which no
                   new iteration starts [None: the ``Budget``'s own ``soft``, 0.5 by default].
                   ``analyse`` writes it onto the budget it is handed for the length of the call
                   and restores the caller's value on return (wrappers that set a budget's
                   ``soft`` for their inner engine, see ``traps`` / ``bayes``, are overridden by
                   this key).  A number in [0, 1]; a JSON boolean is rejected (``false`` would
                   otherwise be 0.0, a depth-1 engine).  ``soft = 0.85`` with ``id_growth = 0`` uses
                   nearly the whole clock: iterations keep starting until 85 % of it is gone, and
                   the last one is then almost always cut by the hard limit at 100 % (its partial
                   root results are dropped and the previous depth is played, so the extra time
                   buys a deeper search only when an iteration happens to finish).  With the
                   default ``id_growth`` the overrun predictor usually stops earlier.
``cache_boards``   cap on cached child positions per search [20000]
``fast``           use ``qchess.engines.fastgen`` for move generation / legality / application
                   [True]; False = the board's own methods throughout (reference path)
=================  ==============================================================================
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..geometry import color_of
from ..moves import Move, MoveType
from . import gametheory
from .base import (Budget, Candidate, Engine, ForcedRNG, OutOfTime, is_quantum, king_balance,
                   outcome_probs, position_key, terminal_value)

try:
    from . import fastgen as _fastgen
except ImportError:  # pragma: no cover - fastgen is part of the package
    _fastgen = None

LO, HI = -1.0, 1.0
EXACT, LOWER, UPPER = 0, 1, 2
CAPTURES = frozenset({MoveType.CAPTURE_JUMP, MoveType.CAPTURE_SLIDE, MoveType.PAWN_CAPTURE})
PIECE_VALUE = {"K": 20.0, "Q": 9.0, "R": 5.0, "B": 3.0, "N": 3.0, "P": 1.0}
P_LOSS_VALUE = -0.95   # a root outcome valued at or below this counts towards Candidate.p_loss

_PRUNE = ("star1", "alphabeta", "none")
_CHANCE = ("exact", "sample")
_ORDERING = ("heuristic", "eval", "none")
_ROOT = ("exact", "pvs", "auto")
_TT_KEY = ("position", "exact")
_OPP_MODELS = ("minimax", "eps", "softmax")

# caps on the per-search caches (entries; checked at insertion, nothing is stored once full)
_MOVES_CAP = 30000
_EVAL_CAP = 100000
_TT_CAP = 300000
# a null-window root move within this of the best value is re-searched exactly (gametheory.decide
# compares means rounded to 9 decimals, so a bound must lie clearly below the best value)
_TIE = 1e-9


def check_opponent(opp_cfg) -> str:
    """The opponent model of ``opp_cfg``; ``ValueError`` for an unknown model, an ``eps`` outside
    [0, 1] (the node value would leave [-1, 1] and break the Star1 bounds) or a non-numeric
    ``temp``."""
    cfg = opp_cfg if opp_cfg is not None else {}
    if not isinstance(cfg, dict):
        raise ValueError(f"expectimax: opponent config must be a dict, got {type(cfg).__name__}")
    model = cfg.get("model", "minimax")
    if model not in _OPP_MODELS:
        raise ValueError(f"expectimax: opponent model={model!r}, expected one of {_OPP_MODELS}")
    try:
        eps = float(cfg.get("eps", 0.1))
        temp = float(cfg.get("temp", 0.05))
    except (TypeError, ValueError) as e:
        raise ValueError(f"expectimax: bad opponent parameter in {cfg!r}") from e
    if model == "eps" and not 0.0 <= eps <= 1.0:
        raise ValueError(f"expectimax: opponent eps={eps!r} must lie in [0, 1]")
    if model == "softmax" and temp != temp:
        raise ValueError("expectimax: opponent temp is NaN")
    return model


def check_soft(soft, who: str = "expectimax") -> float | None:
    """A cfg ``soft`` value: None (unset) or a float in [0, 1]; ``ValueError`` otherwise.  A bool
    is rejected: ``float(False)`` is 0.0, which would stop iterative deepening after depth 1."""
    if soft is None:
        return None
    if isinstance(soft, bool):
        raise ValueError(f"{who}: soft={soft!r} is a bool; give a number in [0, 1] (or leave it out)")
    try:
        v = float(soft)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{who}: soft={soft!r} must be a number in [0, 1]") from e
    if not 0.0 <= v <= 1.0:            # also rejects NaN
        raise ValueError(f"{who}: soft={soft!r} must lie in [0, 1]")
    return v


# ---- decision log (module docstring; shared with mcts.py and simple.py) ----------------------------
REGIMES = ("proven", "seek", "safe", "neutral")
DECISION_LISTS = ("vstar", "chosen_mean", "chosen_worst", "chosen_std")


def _r6(x) -> float | None:
    return None if x is None else round(float(x), 6)


def is_fallback(c: Candidate | None) -> bool:
    """Is ``c`` the placeholder an engine returns when the budget ran out before its search valued
    any root move (``extra["fallback"]``: ``mean`` 0.0 is not a search value)?"""
    return c is not None and bool((c.extra or {}).get("fallback"))


def log_decision(stats: dict, cands, chosen: Candidate | None, regime: str | None = None) -> None:
    """Append one move's decision to ``stats``: v* (the highest mean among ``cands``' exact
    candidates, or among all of them when every one is a bound), the chosen candidate's mean /
    worst / std and, for the adaptive rule, its ``regime``.  A move played without a search,
    ``chosen = None`` or a placeholder (``is_fallback``), logs ``None`` everywhere and no regime
    (whatever ``regime`` says: the adaptive rule calls a lone 0.0 placeholder "neutral")."""
    if is_fallback(chosen):
        chosen, regime = None, None
    if chosen is None:
        vals = (None, None, None, None)
    else:
        real = [c for c in cands if not is_fallback(c)]
        pool = [c for c in real if "bound" not in (c.extra or {})] or real or [chosen]
        vals = (max(c.mean for c in pool), chosen.mean, chosen.worst, chosen.std)
    for k, v in zip(DECISION_LISTS, vals):
        stats.setdefault(k, []).append(_r6(v))
    if regime is not None and "regimes" not in stats:
        stats["regimes"] = [None] * (len(stats["vstar"]) - 1)   # earlier moves: no adaptive rule
        for r in REGIMES:
            stats.setdefault(f"regime_{r}", 0)
    if "regimes" in stats:
        stats["regimes"].append(regime)
    if regime is not None:
        stats[f"regime_{regime}"] = stats.get(f"regime_{regime}", 0) + 1


def decide_and_log(cands, decide_cfg, rng, stats: dict) -> Candidate:
    """``gametheory.decide(cands, decide_cfg, rng)``, logged with ``log_decision``.  The same
    choice and the same ``rng`` draws as ``decide``: the adaptive rule is deterministic, and
    ``decide`` returns ``decide_adaptive(cands, cfg)[0]`` for it, so its regime is read from
    ``decide_adaptive`` directly."""
    if (decide_cfg or {}).get("rule", "argmax") == "adaptive":
        chosen, regime = gametheory.decide_adaptive(cands, decide_cfg)
    else:
        chosen, regime = gametheory.decide(cands, decide_cfg, rng), None
    log_decision(stats, cands, chosen, regime)
    return chosen


def stats_copy(stats: dict) -> dict:
    """``dict(stats)`` with its lists copied, so a snapshot does not grow with later moves."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in stats.items()}


def exact_key(board) -> tuple:
    """Identity of a position for caching: side to move, piece types and the amplitude dict."""
    return (board.turn, tuple(board.types), frozenset(board.state.amps.items()))


def generate_captures(board, is_legal=None) -> list[Move]:
    """All legal capturing moves (capture jump / slide, pawn capture) of the side to move.
    ``is_legal(board, m)`` defaults to ``board.is_legal(m)``."""
    geo, v, me = board.geo, board.types, board.turn
    out = []
    for s in geo.squares():
        ps = v[s]
        if ps is None or color_of(ps) != me:
            continue
        k = ps.upper()
        if k == "P":
            targets = geo.pawn_captures(s, me)
        elif k in "NK":
            targets = geo.jump_targets(s, ps)
        else:
            targets = [t for t, _ in geo.ray_targets(s, ps)]
        for t in targets:
            pt = v[t]
            if pt is None or color_of(pt) == me:
                continue
            m = Move(s, t)
            if (is_legal(board, m) if is_legal is not None else board.is_legal(m)):
                out.append(m)
    return out


def _clip(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    return HI if v > HI else LO if v < LO else v


class ExpectimaxEngine(Engine):
    """Iterative-deepening expectiminimax with alpha-beta / Star1 pruning, a transposition table,
    move ordering, quiescence and beam forward pruning.  See the module docstring for the config.

    ``evaluator``: optional evaluator object (``__call__(board, me) -> float``, ``close()``) used
    instead of ``make_evaluator(cfg["eval"])``; the engine closes only evaluators it created."""

    def __init__(self, cfg: dict | None = None, evaluator=None):
        super().__init__(cfg)
        cfg = self.cfg
        self.max_depth = max(1, int(cfg.get("depth", 4)))
        self.prune = cfg.get("prune", "star1")
        self.chance_mode = cfg.get("chance", "exact")
        self.chance_samples = max(1, int(cfg.get("chance_samples", 1)))
        self.use_tt = bool(cfg.get("tt", True))
        self.tt_key = cfg.get("tt_key", "position")
        self.ordering = cfg.get("ordering", "heuristic")
        self.qdepth = max(0, int(cfg.get("quiescence", 0)))
        self.beam = max(0, int(cfg.get("beam", 0)))
        self.decide_cfg = cfg.get("decide", {"rule": "argmax"})
        self.opp_cfg = cfg.get("opponent", {"model": "minimax"})
        self.allow_quantum = bool(cfg.get("quantum", True))
        self.opp_quantum = bool(cfg.get("opp_quantum", True))
        self.root_mode = cfg.get("root", "exact")
        self.discount = float(cfg.get("discount", 0.001))
        self.id_growth = float(cfg.get("id_growth", 4.0))
        self.soft = check_soft(cfg.get("soft"), "expectimax")
        self.cache_boards = int(cfg.get("cache_boards", 20000))
        self.fast = bool(cfg.get("fast", True)) and _fastgen is not None
        for key, val, allowed in (("prune", self.prune, _PRUNE), ("chance", self.chance_mode, _CHANCE),
                                  ("ordering", self.ordering, _ORDERING), ("root", self.root_mode, _ROOT),
                                  ("tt_key", self.tt_key, _TT_KEY)):
            if val not in allowed:
                raise ValueError(f"expectimax: {key}={val!r}, expected one of {allowed}")
        check_opponent(self.opp_cfg)
        gametheory.check_decide(self.decide_cfg, "expectimax")
        if evaluator is None:
            from .evaluators import make_evaluator
            evaluator = make_evaluator(cfg.get("eval", {"kind": "material"}))
            self._own_evaluator = True
        else:
            self._own_evaluator = False
        self.evaluator = evaluator
        self._shared_rng = np.random.default_rng(0)
        self._spent = None                # caches of the last search, freed in observe()
        self._reset_search_state()

    # ---- engine interface ------------------------------------------------------------------------
    def new_game(self, color: str, seed: int) -> None:
        super().new_game(color, seed)
        self._shared_rng = np.random.default_rng(seed + 1)
        self._free_search_memory()
        # a game's evaluations must not depend on the games this (pooled) engine played before
        reset = getattr(self.evaluator, "reset", None) if self._own_evaluator else None
        if reset is not None:
            reset()

    def set_game_info(self, info: dict) -> None:
        """Arena hook before a game (``ply_cap``, ``start_ply``, ``n``), forwarded to the
        evaluator: a ``tablebase`` evaluator values endgames with the plies left before the cap,
        an ``fsf`` evaluator starts its engine for board size ``n`` outside the move budget."""
        hook = getattr(self.evaluator, "set_game_info", None)
        if hook is not None:
            hook(dict(info or {}))
        self._free_search_memory()        # evaluator values depend on the horizon

    def choose(self, board, budget: Budget) -> Move:
        cands = self.analyse(board, budget)
        if not cands:
            raise ValueError("expectimax: no legal move")
        # root "pvs" upper bounds never compete with exact values (the best move is always exact)
        exact = [c for c in cands if "bound" not in c.extra]
        return decide_and_log(exact or cands, self.decide_cfg, self.rng, self._stats).move

    def observe(self, before, move, result, after) -> None:
        # The arena calls observe outside the move clock: deallocate the last search's caches
        # here, not on analyse's return, where it was charged as an overrun.
        self._spent = None

    def stats(self) -> dict:
        return stats_copy(self._stats)

    def close(self) -> None:
        if self._own_evaluator and self.evaluator is not None:
            self.evaluator.close()
        self.evaluator = None
        self._free_search_memory()

    # ---- caches (all per search) -----------------------------------------------------------------
    def _free_search_memory(self) -> None:
        self._spent = None
        self._fresh_caches()

    def _set_aside_search_memory(self) -> None:
        """End of a search: move the caches aside without deallocating them (``observe``, or the
        next search's start, frees them)."""
        self._spent = (self._tt, self._best, self._child_cache, self._moves_cache, self._eval_cache,
                       self._caps_cache)
        self._fresh_caches()

    def _fresh_caches(self) -> None:
        self._tt: dict = {}               # (tt key, depth, ply) -> (value, flag)
        self._best: dict = {}             # tt key -> best move found at that position
        self._child_cache: dict = {}      # (exact key, move) -> [(p, child)]
        self._moves_cache: dict = {}      # exact key -> [legal moves, heuristic order or None]
        self._eval_cache: dict = {}       # (exact key, me, len(history)) -> evaluator value
        self._caps_cache: dict = {}       # exact key -> legal captures (quiescence)

    def _reset_search_state(self) -> None:
        self._free_search_memory()
        self._n = {"nodes": 0, "tt_hits": 0, "cutoffs": 0, "chance_cutoffs": 0, "evals": 0}

    # ---- the search ------------------------------------------------------------------------------
    def analyse(self, board, budget: Budget) -> list[Candidate]:
        """Every legal root move with its backed-up value (see module docstring).  The board,
        including the state of ``board.rng``, is left exactly as it was."""
        check_opponent(self.opp_cfg)      # wrappers may have replaced it since __init__
        gametheory.check_decide(self.decide_cfg, "expectimax")
        bitgen = getattr(getattr(board, "rng", None), "bit_generator", None)
        rng_state = bitgen.state if bitgen is not None else None
        set_budget = getattr(self.evaluator, "set_budget", None)
        if set_budget is not None:
            set_budget(budget)
        caller_soft = budget.soft
        if self.soft is not None:         # cfg "soft": on the budget for this call only
            budget.soft = self.soft
        try:
            return self._analyse(board, budget)
        finally:
            budget.soft = caller_soft
            if set_budget is not None:
                set_budget(None)
            if rng_state is not None:     # root children are copies, and board.copy() draws a seed
                bitgen.state = rng_state
            self._set_aside_search_memory()   # freed in observe(), outside the move clock

    def _analyse(self, board, budget: Budget) -> list[Candidate]:
        t0 = time.perf_counter()
        nodes0 = budget.used_nodes
        self._reset_search_state()
        self._me = board.turn
        self._budget = budget
        self._ab = self.prune != "none"
        self._star1 = self.prune == "star1"
        self._opp_minimax = gametheory.opp_is_minimax(self.opp_cfg)
        self._opp_cfg_now = dict(self.opp_cfg or {})
        root_mode = self.root_mode
        if root_mode == "auto":
            rule = (self.decide_cfg or {}).get("rule", "argmax")
            root_mode = "pvs" if rule in ("argmax", "visits") else "exact"
        self._pvs_root = root_mode == "pvs" and self._ab

        rkey = exact_key(board)
        moves = self._moves(board, rkey, True)
        if not moves:
            return []
        # iteration 1 at the root is the 1-ply evaluator score, so "eval" ordering is redundant
        # there; the cheap heuristic order also keeps the root inside a tiny budget
        order = list(moves) if self.ordering == "none" else self._heuristic_order(board, moves)
        partial: list[Candidate] = []
        completed: list[Candidate] | None = None
        done_depth = 0
        last_time = last_nodes = None
        for d in range(1, self.max_depth + 1):
            if d > 1:
                if budget.soft_expired() or self._overrun_predicted(budget, last_time, last_nodes):
                    break
            ti, ni = time.perf_counter(), budget.used_nodes
            try:
                cands = self._search_root(board, rkey, order, d, partial if d == 1 else None)
            except OutOfTime:
                break
            last_time, last_nodes = time.perf_counter() - ti, budget.used_nodes - ni
            completed, done_depth = cands, d
            order = [c.move for c in sorted(cands, key=lambda c: -c.mean)]
            if len(moves) == 1:
                break
        if completed is None:
            # no root value at all: a placeholder, which log_decision logs as a move without a search
            completed = partial or [Candidate(order[0], 0.0, extra={"depth": 0, "fallback": True})]
        self._record_stats(done_depth, time.perf_counter() - t0, budget.used_nodes - nodes0)
        return completed

    def _overrun_predicted(self, budget: Budget, last_time, last_nodes) -> bool:
        if self.id_growth <= 0 or last_time is None:
            return False
        if last_time * self.id_growth > budget.seconds - budget.elapsed():
            return True
        if budget.nodes is not None and last_nodes * self.id_growth > budget.nodes - budget.used_nodes:
            return True
        return False

    def _record_stats(self, depth: int, secs: float, nodes: int) -> None:
        s = self._stats
        s["searches"] = s.get("searches", 0) + 1
        s["nodes"] = s.get("nodes", 0) + nodes
        s["time"] = s.get("time", 0.0) + secs
        s["depth_sum"] = s.get("depth_sum", 0) + depth
        s["last_depth"] = depth
        s["max_depth"] = max(s.get("max_depth", 0), depth)
        s["min_depth"] = min(s.get("min_depth", depth), depth)
        for k in ("tt_hits", "cutoffs", "chance_cutoffs", "evals"):
            s[k] = s.get(k, 0) + self._n[k]
        s["last_nodes"] = nodes

    def _search_root(self, board, rkey, order: list[Move], d: int, partial) -> list[Candidate]:
        cands = []
        alpha = LO
        for i, m in enumerate(order):
            outs = self._children(board, rkey, m, True)
            if self._pvs_root and i > 0:
                rec: list = []
                v = self._chance(outs, d, 0, alpha, HI, rec)
                # without Star1 a multi-outcome chance node searches every outcome with a full
                # window, so only a single-outcome move or a Star1 node can fail low
                can_fail_low = self._star1 or len(outs) == 1
                if can_fail_low and v < alpha - _TIE:
                    # strictly worse than the best move: an upper bound, worst unknown
                    c = Candidate(m, v, LO, 0.0, 1, 0.0, extra={"depth": d, "bound": "upper"})
                elif (can_fail_low and v <= alpha) or len(rec) != len(outs):
                    # a tie with the best value (fail-soft often returns alpha itself) or a Star1
                    # fail-high on a root outcome (its value is right, its record incomplete)
                    rec = [(p, self._node(child, d - 1, 1, LO, HI)) for p, child in outs]
                    c = self._root_candidate(m, rec, d)
                else:
                    c = self._root_candidate(m, rec, d)
            else:
                rec = [(p, self._node(child, d - 1, 1, LO, HI)) for p, child in outs]
                c = self._root_candidate(m, rec, d)
            alpha = max(alpha, c.mean)
            cands.append(c)
            if partial is not None:
                partial.append(c)
        return cands

    @staticmethod
    def _root_candidate(m: Move, rec: list, d: int) -> Candidate:
        mean = sum(p * v for p, v in rec)
        worst = min(v for _, v in rec)
        var = sum(p * (v - mean) ** 2 for p, v in rec)
        p_loss = sum(p for p, v in rec if v <= P_LOSS_VALUE)
        return Candidate(m, mean, worst, math.sqrt(max(var, 0.0)), 1, p_loss,
                         extra={"depth": d, "outcomes": [(p, v) for p, v in rec]})

    def _tick(self) -> None:
        b = self._budget
        b.tick()
        self._n["nodes"] += 1
        if b.expired():
            raise OutOfTime

    def _node(self, board, depth: int, ply: int, alpha: float, beta: float) -> float:
        """Value of a decision node (``depth > 0``) or quiescence/leaf node (``depth <= 0``)."""
        self._tick()
        tv = terminal_value(board, self._me)
        if tv is not None:
            return tv * (1.0 - self.discount * ply)
        ekey = exact_key(board)
        if depth <= 0:
            return self._quiesce(board, ekey, depth, ply, alpha, beta)
        maximizing = board.turn == self._me
        minimax_node = maximizing or self._opp_minimax
        use_ab = self._ab and minimax_node

        tkey = None
        tt_move = None
        if self.use_tt:
            pk = ekey if self.tt_key == "exact" else position_key(board)
            tkey = (pk, depth, ply)
            ent = self._tt.get(tkey)
            if ent is not None:
                val, flag = ent
                if flag == EXACT or (flag == LOWER and val >= beta) or (flag == UPPER and val <= alpha):
                    self._n["tt_hits"] += 1
                    return val
            tt_move = self._best.get(pk)

        moves = self._moves(board, ekey, maximizing)
        if not moves:
            return king_balance(board, self._me) * (1.0 - self.discount * ply)
        moves = self._order(board, ekey, moves, ply, maximizing, tt_move)
        if self.beam:
            moves = moves[: self.beam]

        best_move = None
        if maximizing:
            best = -math.inf
            a = alpha
            for m in moves:
                v = self._move_value(board, ekey, m, depth, ply, a, beta)
                if v > best:
                    best, best_move = v, m
                if use_ab:
                    if best > a:
                        a = best
                    if best >= beta:
                        self._n["cutoffs"] += 1
                        break
        elif minimax_node:
            best = math.inf
            b = beta
            for m in moves:
                v = self._move_value(board, ekey, m, depth, ply, alpha, b)
                if v < best:
                    best, best_move = v, m
                if use_ab:
                    if best < b:
                        b = best
                    if best <= alpha:
                        self._n["cutoffs"] += 1
                        break
        else:
            vals = [self._move_value(board, ekey, m, depth, ply, LO, HI) for m in moves]
            best = gametheory.opp_combine(vals, self._opp_cfg_now)
            best_move = moves[int(np.argmin(vals))]

        if tkey is not None:
            if not use_ab:
                flag = EXACT
            elif best <= alpha:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            else:
                flag = EXACT
            if len(self._tt) < _TT_CAP:
                self._tt[tkey] = (best, flag)
                self._best[tkey[0]] = best_move
        return best

    def _quiesce(self, board, ekey, depth: int, ply: int, alpha: float, beta: float) -> float:
        stand = self._eval(board, ekey) * (1.0 - self.discount * ply)
        if self.qdepth + depth <= 0:
            return stand
        maximizing = board.turn == self._me
        caps = self._captures(board, ekey)
        if not caps:
            return stand
        caps.sort(key=lambda m: -self._capture_score(board, m))
        if maximizing:
            best = stand
            if self._ab and best >= beta:
                return best
            a = max(alpha, best)
            for m in caps:
                v = self._move_value(board, ekey, m, depth, ply, a, beta)
                if v > best:
                    best = v
                if self._ab:
                    if best > a:
                        a = best
                    if best >= beta:
                        self._n["cutoffs"] += 1
                        break
            return best
        if self._opp_minimax:
            best = stand
            if self._ab and best <= alpha:
                return best
            b = min(beta, best)
            for m in caps:
                v = self._move_value(board, ekey, m, depth, ply, alpha, b)
                if v < best:
                    best = v
                if self._ab:
                    if best < b:
                        b = best
                    if best <= alpha:
                        self._n["cutoffs"] += 1
                        break
            return best
        vals = [stand] + [self._move_value(board, ekey, m, depth, ply, LO, HI) for m in caps]
        return gametheory.opp_combine(vals, self._opp_cfg_now)

    def _move_value(self, board, ekey, m: Move, depth: int, ply: int, alpha: float, beta: float) -> float:
        outs = self._children(board, ekey, m, depth >= 2)
        if len(outs) > 1 and self.chance_mode == "sample":
            outs = self._sample(outs)
        if len(outs) == 1:
            return self._node(outs[0][1], depth - 1, ply + 1, alpha, beta)
        return self._chance(outs, depth, ply, alpha, beta)

    def _chance(self, outs, depth: int, ply: int, alpha: float, beta: float, rec: list | None = None) -> float:
        """Chance node over ``outs = [(p, child)]`` (children are at ``depth - 1``, ``ply + 1``)."""
        if len(outs) == 1:
            v = self._node(outs[0][1], depth - 1, ply + 1, alpha, beta)
            if rec is not None:
                rec.append((outs[0][0], v))
            return v
        if not self._star1:
            total = 0.0
            for p, child in outs:
                v = self._node(child, depth - 1, ply + 1, LO, HI)
                if rec is not None:
                    rec.append((p, v))
                total += p * v
            return total
        done = 0.0
        rest = sum(p for p, _ in outs)
        for p, child in outs:
            rest -= p
            if rest < 0.0:
                rest = 0.0
            a_i = (alpha - done - HI * rest) / p
            b_i = (beta - done - LO * rest) / p
            if a_i >= HI:
                self._n["chance_cutoffs"] += 1
                return done + (p + rest) * HI
            if b_i <= LO:
                self._n["chance_cutoffs"] += 1
                return done + (p + rest) * LO
            v = self._node(child, depth - 1, ply + 1, max(a_i, LO), min(b_i, HI))
            if v <= a_i:
                self._n["chance_cutoffs"] += 1
                return done + p * v + rest * HI
            if v >= b_i:
                self._n["chance_cutoffs"] += 1
                return done + p * v + rest * LO
            if rec is not None:
                rec.append((p, v))
            done += p * v
        return done

    def _sample(self, outs):
        probs = np.array([p for p, _ in outs], dtype=float)
        probs /= probs.sum()
        k = self.chance_samples
        counts = np.bincount(self.rng.choice(len(outs), size=k, p=probs), minlength=len(outs))
        return [(c / k, child) for (_, child), c in zip(outs, counts) if c > 0]

    # ---- positions, moves, children, leaves -----------------------------------------------------
    def _moves(self, board, ekey, maximizing: bool) -> list[Move]:
        ent = self._moves_cache.get(ekey)
        if ent is None:
            ent = [_fastgen.fast_legal_moves(board) if self.fast else board.legal_moves(), None]
            if len(self._moves_cache) < _MOVES_CAP:
                self._moves_cache[ekey] = ent
        ms = ent[0]
        allow = self.allow_quantum if maximizing else self.opp_quantum
        if not allow:
            cl = [m for m in ms if not is_quantum(m)]
            return cl or ms
        return ms

    def _captures(self, board, ekey) -> list[Move]:
        """Legal capturing moves of the side to move (never splits/merges).  Uses the cached legal
        move list when there is one, else a geometric generator plus ``board.is_legal`` (much
        cheaper than ``legal_moves``, which tests every split and merge)."""
        ent = self._moves_cache.get(ekey)
        if ent is not None:
            return [m for m in ent[0] if not is_quantum(m) and board.classify(m) in CAPTURES]
        caps = self._caps_cache.get(ekey)
        if caps is None:
            caps = generate_captures(board, _fastgen.fast_is_legal if self.fast else None)
            if len(self._caps_cache) < _MOVES_CAP:
                self._caps_cache[ekey] = caps
        return list(caps)

    def _children(self, board, ekey, m: Move, store: bool) -> list:
        key = (ekey, m)
        outs = self._child_cache.get(key)
        if outs is not None:
            return outs
        outs = []
        for o, p in outcome_probs(board, m):
            if self.fast:
                child = _fastgen.fast_apply(board, m, o)
            else:                             # reference path: the board's own push
                child = board.copy()
                if o is not None:
                    child.rng = ForcedRNG(o)
                child.push(m)
            child.rng = self._shared_rng      # the per-copy Generator is half a board's memory
            outs.append((p, child))
        if len(outs) > 1:
            outs.sort(key=lambda t: -t[0])
        if store and len(self._child_cache) < self.cache_boards:
            self._child_cache[key] = outs
        return outs

    def _eval(self, board, ekey) -> float:
        # history length is part of the key: horizon-aware evaluators (tablebase) depend on it
        k = (ekey, self._me, len(board.history))
        v = self._eval_cache.get(k)
        if v is None:
            self._n["evals"] += 1
            v = _clip(float(self.evaluator(board, self._me)))
            if len(self._eval_cache) < _EVAL_CAP:
                self._eval_cache[k] = v
        return v

    def _leaf_value(self, child, ply: int) -> float:
        """Static value of a child (terminal-aware), used by ``"ordering": "eval"``."""
        tv = terminal_value(child, self._me)
        if tv is not None:
            return tv * (1.0 - self.discount * ply)
        return self._eval(child, exact_key(child)) * (1.0 - self.discount * ply)

    # ---- move ordering ---------------------------------------------------------------------------
    def _order(self, board, ekey, moves: list[Move], ply: int, maximizing: bool, tt_move) -> list[Move]:
        if self.ordering == "none":
            return list(moves)
        if self.ordering == "eval":
            scored = []
            budget = self._budget
            for i, m in enumerate(moves):
                if budget.expired():
                    raise OutOfTime
                outs = self._children(board, ekey, m, True)
                s = sum(p * self._leaf_value(c, ply + 1) for p, c in outs)
                scored.append((-s if maximizing else s, i, m))
            scored.sort(key=lambda t: (t[0], t[1]))
            ordered = [m for _, _, m in scored]
        else:
            ent = self._moves_cache.get(ekey)
            if ent is None:              # only when the moves cache is full
                ordered = self._heuristic_order(board, moves)
            else:
                if ent[1] is None:
                    ent[1] = self._heuristic_order(board, ent[0])
                if len(moves) == len(ent[0]):
                    ordered = list(ent[1])
                else:                    # a quantum-filtered list
                    allowed = set(moves)
                    ordered = [m for m in ent[1] if m in allowed]
        if tt_move is not None and ordered and ordered[0] != tt_move and tt_move in ordered:
            ordered.remove(tt_move)
            ordered.insert(0, tt_move)
        return ordered

    def _heuristic_order(self, board, moves: list[Move]) -> list[Move]:
        """Captures (most valuable likely victim first, king captures first), promotions, moves that
        attack an enemy king square, splits, then the rest; stable within a class."""
        me = board.turn
        types = board.types
        enemy_king = "k" if me == "w" else "K"
        kings = [q for q, p in enumerate(types) if p == enemy_king]
        probs = {}

        def prob(q):
            v = probs.get(q)
            if v is None:
                v = probs[q] = board.prob(q)
            return v

        attacks = _attack_table(board.n)
        last = board.geo.last_rank(me)
        n = board.n
        scored = []
        for i, m in enumerate(moves):
            s, t = m.source, m.target
            piece = types[s]
            if m.is_split:
                sc = 150.0 if piece in "Kk" else 100.0
            elif m.is_merge:
                sc = 0.0
            else:
                victim = types[t]
                pawn = piece in "Pp"
                if victim is not None and (victim.isupper() != piece.isupper()) and (not pawn or s % n != t % n):
                    sc = 10000.0 + 100.0 * PIECE_VALUE[victim.upper()] * prob(t) * prob(s) - PIECE_VALUE[piece.upper()]
                else:
                    sc = 0.0
                    if pawn and t // n == last:
                        sc += 2000.0
                    if kings:
                        for q, path in attacks[(piece, t)]:
                            if q in kings and not any(prob(x) >= 1 - 1e-10 for x in path):
                                sc += 1000.0
                                break
            scored.append((-sc, i, m))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [m for _, _, m in scored]

    def _capture_score(self, board, m: Move) -> float:
        victim = board.types[m.target]
        attacker = board.types[m.source]
        pv = PIECE_VALUE.get(victim.upper(), 1.0) if victim else 0.0
        pa = PIECE_VALUE.get(attacker.upper(), 1.0) if attacker else 0.0
        return 100.0 * pv * board.prob(m.target) * board.prob(m.source) - pa


_ATTACK_TABLES: dict = {}


def _attack_table(n: int) -> dict:
    """``{(piece letter, square): [(attacked square, path squares)]}`` on an empty n x n board
    (pawns: their two capture squares; sliders: every ray square with the squares in between)."""
    tab = _ATTACK_TABLES.get(n)
    if tab is not None:
        return tab
    from ..geometry import Geometry
    geo = Geometry(n)
    tab = {}
    for piece in "KQRBNPkqrbnp":
        for sq in geo.squares():
            k = piece.upper()
            if k == "P":
                color = "w" if piece.isupper() else "b"
                lst = [(q, ()) for q in geo.pawn_captures(sq, color)]
            elif k in "NK":
                lst = [(q, ()) for q in geo.jump_targets(sq, piece)]
            else:
                lst = [(q, tuple(path)) for q, path in geo.ray_targets(sq, piece)]
            tab[(piece, sq)] = lst
    _ATTACK_TABLES[n] = tab
    return tab


__all__ = ["ExpectimaxEngine", "exact_key", "generate_captures", "check_opponent", "check_soft",
           "is_fallback", "log_decision", "decide_and_log", "stats_copy"]
