"""Game-theory layer: how a searcher turns root statistics into a move, and how it models the
opponent at MIN nodes.

Decision rules (``cfg["decide"]["rule"]``), applied to the root ``Candidate`` list:

* ``argmax``   - highest ``mean`` (expected value; the expectiminimax choice).
* ``maximin``  - highest ``worst`` (safest across measurement outcomes / branches), ties by mean.
* ``meanstd``  - highest ``mean - lam * std`` (risk-averse for lam > 0, risk-seeking for lam < 0).
* ``cvar``     - highest ``(1 - lam) * mean + lam * worst`` (a cheap CVaR-style blend).
* ``softmax``  - a *mixed strategy*: sample a move with probability proportional to
                 ``exp(mean / temp)`` (quantal response); ``temp -> 0`` is argmax.
* ``visits``   - MCTS robust child: most visits, ties by mean.
* ``adaptive`` - a *state-dependent risk attitude* (below): seek a swing when behind, minimise
                 variance when ahead, argmax in between.

``check_decide(cfg)`` validates a decide config (known rule, numeric ``lam`` / ``temp``, the
``adaptive`` parameters).  ``ExpectimaxEngine`` and the ``trap`` / ``bayes`` wrappers call it when
they are built (the wrappers also on their inner engine's rule and on the rules they write into
it), so a bad config fails in ``make_engine`` rather than in ``choose`` after a whole search.

The ``adaptive`` rule
---------------------
The game-theoretic lesson of finding F1 (``reports/engines/FINDINGS.md``): from a classically lost
position a measuring move is a Born-weighted lottery that a branch-uniform opponent cannot refuse,
so gambling helps when behind, and variance only hurts when ahead.  With ``v*`` = the highest
``mean`` among the candidates:

1. a candidate with ``extra["proven"] == "win"`` (MCTS: every outcome captures the king) is always
   taken (the highest mean among them);
2. ``v* <= lose_below`` (default -0.3), behind: look for a swing.

   * ``seek = "gain"`` (default): the highest ``mean + seek_lam * gain`` (``seek_lam`` 1.0), where
     ``gain = sum p * max(0, v - v*)`` over the candidate's own measurement outcomes
     (``extra["outcomes"]``: expectimax and greedy ``[(p, v)]``, MCTS ``[(outcome, p, v,
     visits)]``) is how far, in expectation, the move's outcome beats the best mean on offer.  It
     is 0 for a unitary move (one outcome, its mean) and for a candidate without an outcome record,
     and an outcome MCTS has not valued yet adds nothing.
     Both terms grow with every outcome value, so a move that another candidate first-order
     stochastically dominates (every outcome worse or equal) is never preferred to it.
   * ``seek = "upside"``: the highest best-outcome value, the largest ``v`` with ``p > 0`` in the
     outcome record (``mean + std`` without one).  It ignores how likely that outcome is (a 1 %
     shot ties a 50 % one; the tie goes to the higher mean).
   * ``seek = "std"``: the highest ``mean + seek_lam * std``.  Downside spread counts like upside
     spread, so it can prefer a dominated move (80 % -0.5 / 20 % -1.0 over a sure -0.5), and see
     the caveats on what ``std`` measures.
3. ``v* >= win_above`` (default 0.3), ahead: minimise variance among the moves that keep me ahead,
   those with ``mean >= max(win_above, v* - safe_margin)`` (``safe_margin`` 0.2; ``None``: no
   margin, only ``win_above``).  The move that made ``v*`` is always one of them, so a lottery that
   alone puts me ahead is never traded for a move below ``win_above``.  ``safe = "maximin"``
   (default) takes the highest ``worst`` among them; ``safe = "cvar"`` the highest ``(1 -
   safe_lam) * mean + safe_lam * worst`` (``safe_lam`` 0.5).  The margin bounds what the safe
   choice gives up in expectation: maximin compares a measuring move's worst *root* outcome with a
   unitary move's backed-up mean, in which later risk is already averaged away, so it leans
   towards unitary moves;
4. otherwise ``argmax`` (``mid = "visits"``, an extra key, uses the MCTS robust child instead).

Config: ``{"rule": "adaptive", "lose_below": -0.3, "win_above": 0.3, "seek": "gain" | "upside" |
"std", "seek_lam": 1.0, "safe": "maximin" | "cvar", "safe_lam": 0.5, "safe_margin": 0.2}``;
``ValueError`` for an unknown ``seek`` / ``safe`` / ``mid``, ``lose_below > win_above``, ``seek_lam
< 0``, ``safe_lam`` outside [0, 1] or a negative ``safe_margin``.  Ties go to the higher mean, then
to the earliest candidate in the list: the rule never draws from ``rng``, so it is deterministic.
Candidates marked ``extra["bound"]`` (expectimax ``"root": "pvs"`` upper bounds) are ignored when
exact ones exist; expectimax's ``"root": "auto"`` already searches exactly under this rule.
``decide_adaptive`` returns the choice together with the regime used (``"proven"``, ``"seek"``,
``"safe"``, ``"neutral"``); ``adaptive_rank`` lists every candidate in the rule's order of
preference (its choice first), which wrappers use to pick the candidates worth extra work.

Scale.  ``lose_below`` / ``win_above`` / ``safe_margin`` are in the evaluator's value units, not
game results, so they should be set per evaluator.  F1's endgames are scored by the ``tablebase``
evaluator, which values every classical branch of a solved endgame as decided (about +-0.99) or
drawn (0), so a position's value is roughly P(win) - P(loss) and the defaults read "behind" /
"ahead" in game terms (branches outside the solved families fall back to a heuristic evaluator).  The heuristic evaluators (``material``, ``fsf``) squash a balance of x pawns to ``0.85 *
tanh(x / 4)``: there 0.2 is about 1.0 pawn, 0.3 about 1.5, 0.5 about 2.7, 0.7 about 4.7 and 0.8
about 7.0, and only forced results (a capturable king, an engine mate: 0.93-0.98) and decided ones
lie beyond 0.85.  With those evaluators the defaults gamble whenever the best move leaves the
engine 1.5 pawns down and play safe whenever 1.5 pawns up; to gamble only where the evaluator calls
the position forced or decided, as in F1, use ``lose_below = -0.9`` (``win_above = 0.9``).

Caveats: what the statistics measure depends on the searcher.  ``std``: the Born-weighted spread
over the root move's own measurement outcomes for expectimax, greedy and MCTS with ``chance =
"expect"``; with MCTS ``chance = "sample"`` it is the spread of every value backed up through the
edge (later replies, exploration, leaf noise), non-zero for unitary moves, so ``seek = "std"``
there rewards exploration variance rather than Born lotteries (``gain`` and ``upside`` read the
outcome record, which holds a single entry for a unitary move).  ``worst``: the lowest root
outcome value for expectimax and MCTS; for greedy also the lowest *classical branch* value inside
each outcome, so greedy's maximin is branch-wise.  A lottery set up by a unitary move (a king
split measured plies later) is averaged into ``mean`` and invisible to ``seek``.  ``v*`` from MCTS
candidates with few visits is noisy.

Opponent models (``cfg["opponent"]["model"]``), used by searchers at the opponent's decision nodes
(values given from *my* point of view, so the opponent prefers low values):

* ``minimax``  - the opponent plays my worst reply: ``min(values)``.
* ``eps``      - the opponent is omega_eps: with probability eps a uniformly random reply,
                 otherwise the best one: ``(1 - eps) * min + eps * mean``.
* ``softmax``  - quantal-response opponent: picks reply v with probability ~ exp(-v / temp).

Extended by ``qchess.engines.bayes`` (opponent posteriors) and by ``nash_root`` below.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Candidate


# ---- decision rules ------------------------------------------------------------------------------
def _key(c: Candidate, rule: str, lam: float) -> tuple:
    if rule == "maximin":
        return (round(c.worst, 9), round(c.mean, 9))
    if rule == "meanstd":
        return (round(c.mean - lam * c.std, 9), round(c.mean, 9))
    if rule == "cvar":
        return (round((1 - lam) * c.mean + lam * c.worst, 9), round(c.mean, 9))
    if rule == "visits":
        return (c.visits, round(c.mean, 9))
    return (round(c.mean, 9), round(c.worst, 9))


def decide(cands: list[Candidate], cfg: dict | None, rng: np.random.Generator) -> Candidate:
    """Pick a root candidate according to ``cfg`` (``{"rule": ..., "lam": ..., "temp": ...}``)."""
    if not cands:
        raise ValueError("no candidates")
    cfg = cfg or {}
    rule = cfg.get("rule", "argmax")
    if rule == "adaptive":
        return decide_adaptive(cands, cfg)[0]
    if rule == "softmax":
        temp = max(float(cfg.get("temp", 0.05)), 1e-6)
        means = np.array([c.mean for c in cands])
        z = (means - means.max()) / temp
        p = np.exp(z)
        p /= p.sum()
        return cands[int(rng.choice(len(cands), p=p))]
    lam = float(cfg.get("lam", 0.5))
    best = max(_key(c, rule, lam) for c in cands)
    top = [c for c in cands if _key(c, rule, lam) == best]
    return top[int(rng.integers(len(top)))]


# ---- the adaptive (state-dependent risk attitude) rule --------------------------------------------
ADAPTIVE_DEFAULTS = {"lose_below": -0.3, "win_above": 0.3, "seek": "gain", "seek_lam": 1.0,
                     "safe": "maximin", "safe_lam": 0.5, "safe_margin": 0.2, "mid": "argmax"}
_SEEK = ("gain", "upside", "std")
_SAFE = ("maximin", "cvar")
_MID = ("argmax", "visits")
DECIDE_RULES = ("argmax", "maximin", "meanstd", "cvar", "softmax", "visits", "adaptive")


def _number(x) -> float:
    """``float(x)`` for a real number (or numeric string); ``TypeError`` for a bool, which
    ``float`` would silently turn into 0 or 1."""
    if isinstance(x, bool):
        raise TypeError(f"{x!r} is a bool, not a number")
    return float(x)


def adaptive_params(cfg: dict | None) -> dict:
    """The ``adaptive`` rule's parameters (defaults filled in), checked; ``ValueError`` if bad."""
    p = {**ADAPTIVE_DEFAULTS, **{k: v for k, v in (cfg or {}).items() if k in ADAPTIVE_DEFAULTS}}
    num = ["lose_below", "win_above", "seek_lam", "safe_lam"]
    if p["safe_margin"] is not None:
        num.append("safe_margin")
    try:
        for k in num:
            p[k] = _number(p[k])
    except (TypeError, ValueError) as e:
        raise ValueError(f"adaptive rule: bad numeric parameter in {cfg!r}") from e
    if any(p[k] != p[k] for k in num):
        raise ValueError(f"adaptive rule: NaN parameter in {cfg!r}")
    if p["seek"] not in _SEEK:
        raise ValueError(f"adaptive rule: seek={p['seek']!r}, expected one of {_SEEK}")
    if p["safe"] not in _SAFE:
        raise ValueError(f"adaptive rule: safe={p['safe']!r}, expected one of {_SAFE}")
    if p["mid"] not in _MID:
        raise ValueError(f"adaptive rule: mid={p['mid']!r}, expected one of {_MID}")
    if p["lose_below"] > p["win_above"]:
        raise ValueError(f"adaptive rule: lose_below={p['lose_below']} above win_above={p['win_above']}")
    if p["seek_lam"] < 0:
        raise ValueError(f"adaptive rule: seek_lam={p['seek_lam']} must be >= 0 (risk seeking)")
    if not 0.0 <= p["safe_lam"] <= 1.0:
        raise ValueError(f"adaptive rule: safe_lam={p['safe_lam']} must lie in [0, 1]")
    if p["safe_margin"] is not None and p["safe_margin"] < 0:
        raise ValueError(f"adaptive rule: safe_margin={p['safe_margin']} must be >= 0 (or None)")
    return p


def check_decide(cfg, who: str = "decide") -> dict:
    """Validate a decide config (``None`` = argmax): a known ``rule``, numeric ``lam`` / ``temp``
    when given, and the ``adaptive`` parameters.  Returns ``cfg`` (or ``{}``); ``ValueError`` if
    bad.  Engines call it at construction, so a bad config fails before any search."""
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{who}: decide config must be a dict, got {type(cfg).__name__}")
    rule = cfg.get("rule", "argmax")
    if rule not in DECIDE_RULES:
        raise ValueError(f"{who}: decide rule={rule!r}, expected one of {DECIDE_RULES}")
    for k in ("lam", "temp"):
        if k in cfg:
            try:
                v = _number(cfg[k])
            except (TypeError, ValueError) as e:
                raise ValueError(f"{who}: decide {k}={cfg[k]!r} must be a number") from e
            if v != v:
                raise ValueError(f"{who}: decide {k} is NaN")
    if rule == "adaptive":
        try:
            adaptive_params(cfg)
        except ValueError as e:
            raise ValueError(f"{who}: {e}") from e
    return cfg


def _outcome_values(c: Candidate) -> list[tuple[float, float]]:
    """``[(p, v)]`` over a candidate's own measurement outcomes with ``p > 0`` (``extra["outcomes"]``:
    expectimax / greedy ``[(p, v)]``, MCTS ``[(outcome, p, v, visits)]``); malformed entries are
    skipped, so the list can be empty."""
    out = []
    for t in (c.extra or {}).get("outcomes") or ():
        try:
            p, v = (t[0], t[1]) if len(t) == 2 else (t[1], t[2])
            if p is not None and v is not None and float(p) > 0.0:
                out.append((float(p), float(v)))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def best_outcome(c: Candidate) -> float:
    """The best value over a candidate's own measurement outcomes with positive probability
    (``extra["outcomes"]``: expectimax ``[(p, v)]``, MCTS ``[(outcome, p, v, visits)]``);
    ``mean + std`` when the candidate has no usable outcome record."""
    vals = [v for _, v in _outcome_values(c)]
    return max(vals) if vals else c.mean + c.std


def expected_gain(c: Candidate, ref: float) -> float:
    """``sum p * max(0, v - ref)`` over a candidate's own measurement outcomes (``_outcome_values``):
    by how much, in expectation, the move's outcome beats ``ref``.  Without a usable outcome
    record the candidate counts as one outcome at its mean, ``max(0, mean - ref)``."""
    outs = _outcome_values(c)
    if not outs:
        return max(0.0, c.mean - ref)
    return sum(p * max(0.0, v - ref) for p, v in outs)


def adaptive_rank(cands: list[Candidate], cfg: dict | None) -> tuple[list[Candidate], str]:
    """Every candidate in the ``adaptive`` rule's order of preference (module docstring) and the
    regime: ``(ranked, regime)`` with ``ranked[0]`` the rule's choice.  Within the regime's pool
    the order is the regime's key (exact ties: higher mean, then list order); candidates outside
    the pool follow by mean (the safe regime's moves below the margin, then any bound candidates).
    Deterministic: nothing is drawn."""
    if not cands:
        raise ValueError("no candidates")
    p = adaptive_params(cfg)
    exact = [c for c in cands if "bound" not in (c.extra or {})]
    pool = exact or list(cands)
    in_pool = {id(c) for c in pool}
    bounds = [c for c in cands if id(c) not in in_pool]
    r = lambda x: round(float(x), 9)   # noqa: E731 - same rounding as _key
    by_mean = lambda c: (r(c.mean), r(c.worst))   # noqa: E731

    def ranked(first, key):
        """``first`` by ``key`` (stable: equal keys keep list order), then the rest of the pool
        and the bounds by mean."""
        ids = {id(c) for c in first}
        rest = [c for c in pool if id(c) not in ids]
        return (sorted(first, key=key, reverse=True) + sorted(rest, key=by_mean, reverse=True)
                + sorted(bounds, key=by_mean, reverse=True))

    wins = [c for c in pool if (c.extra or {}).get("proven") == "win"]
    if wins:
        return ranked(wins, by_mean), "proven"
    vstar = max(c.mean for c in pool)
    if vstar <= p["lose_below"]:
        if p["seek"] == "gain":
            lam = p["seek_lam"]
            key = lambda c: (r(c.mean + lam * expected_gain(c, vstar)), r(c.mean))   # noqa: E731
        elif p["seek"] == "std":
            lam = p["seek_lam"]
            key = lambda c: (r(c.mean + lam * c.std), r(c.mean))   # noqa: E731
        else:
            key = lambda c: (r(best_outcome(c)), r(c.mean))   # noqa: E731
        return ranked(pool, key), "seek"
    if vstar >= p["win_above"]:
        floor = p["win_above"] if p["safe_margin"] is None else max(p["win_above"], vstar - p["safe_margin"])
        keep = [c for c in pool if r(c.mean) >= r(floor)]      # never empty: v*'s move is in it
        if p["safe"] == "maximin":
            key = lambda c: (r(c.worst), r(c.mean))   # noqa: E731
        else:
            lam = p["safe_lam"]
            key = lambda c: (r((1 - lam) * c.mean + lam * c.worst), r(c.mean))   # noqa: E731
        return ranked(keep, key), "safe"
    if p["mid"] == "visits":
        return ranked(pool, lambda c: (c.visits, r(c.mean))), "neutral"
    return ranked(pool, by_mean), "neutral"


def decide_adaptive(cands: list[Candidate], cfg: dict | None) -> tuple[Candidate, str]:
    """The ``adaptive`` rule (module docstring): ``(chosen candidate, regime)`` with regime
    ``"proven"`` | ``"seek"`` | ``"safe"`` | ``"neutral"``.  Deterministic: exact ties (keys
    rounded to 9 decimals) go to the higher mean, then to the earliest candidate."""
    ranked, regime = adaptive_rank(cands, cfg)
    return ranked[0], regime


# ---- opponent models -----------------------------------------------------------------------------
def opp_combine(values: list[float], cfg: dict | None) -> float:
    """Value of an opponent decision node from my view, given my values of its replies."""
    if not values:
        raise ValueError("no replies")
    cfg = cfg or {}
    model = cfg.get("model", "minimax")
    lo = min(values)
    if model == "minimax":
        return lo
    if model == "eps":
        eps = float(cfg.get("eps", 0.1))
        return (1 - eps) * lo + eps * (sum(values) / len(values))
    if model == "softmax":
        temp = max(float(cfg.get("temp", 0.05)), 1e-6)
        v = np.array(values)
        z = -(v - lo) / temp
        p = np.exp(z)
        p /= p.sum()
        return float((p * v).sum())
    raise ValueError(f"unknown opponent model {model}")


def opp_is_minimax(cfg: dict | None) -> bool:
    """Alpha-beta pruning at opponent nodes is only sound for the pure minimax model."""
    return (cfg or {}).get("model", "minimax") == "minimax"


def entropy(probs) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))
