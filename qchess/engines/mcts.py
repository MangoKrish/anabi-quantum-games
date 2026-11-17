"""Monte Carlo Tree Search for Cantwell quantum chess with explicit chance nodes.

Tree layout: a *decision node* holds a position (its board is stored in the node), an *edge*
holds a move played there and owns one child decision node per measurement outcome (a single
child, key ``None``, for a unitary move; children keyed ``1`` / ``0`` for a measuring move, with
their Born probabilities).  All values are from the point of view of ``me`` = the side to move at
the root, in [-1, 1]; terminal values carry a small ply discount ``v * (1 - 0.001 * ply)`` so that
faster wins and slower losses are preferred.

Config keys (besides ``family``/``name``):

* ``select``  - ``"uct"`` (default): ``Q + c * sqrt(ln N / n)``, unvisited children first in
                heuristic order.  ``"puct"``: ``Q + c * P * sqrt(N) / (1 + n)`` with a prior P =
                softmax of the exact 1-ply child evaluations (temperature ``prior_temp``, default
                0.2), first-play urgency = that 1-ply value.  ``"thompson"``: Bayesian MCTS; each
                child has a Gaussian posterior over its mean value (prior mean = the parent's value,
                or the 1-ply value when known, with ``prior_n`` pseudo-visits, default 1, and
                observation noise ``sigma``, default 0.5, mixed with the sample variance); a value
                is drawn per child and the max (my nodes) / min (opponent nodes) is taken.
                ``"bts"`` / ``"ments"``: Boltzmann tree search and maximum-entropy MCTS, see
                *Boltzmann selection* below.
* ``c``       - exploration constant for uct / puct (default 1.0, values live in [-1, 1]).
* ``dirichlet`` - puct only: ``{"alpha": 0.3, "frac": 0.25}`` (``true`` = these defaults) mixes
                AlphaZero root noise (Silver et al., arXiv:1712.01815: alpha 0.3 for chess;
                frac 0.25 as in AlphaGo Zero, Nature 550, 2017) into the root prior,
                ``P' = (1 - frac) P + frac eta`` with
                ``eta ~ Dirichlet(alpha, ..., alpha)`` over every root move, drawn from the
                engine ``rng`` once per search (so it is fresh at every move and reproducible from
                the game seed).  Only the root's prior changes: the root's move order (and so
                progressive widening) stays the order of the clean prior, every other node keeps
                its clean prior, and a root analysed again (``reuse`` on the same position) gets
                new noise mixed into its clean prior, not into the previous noisy one
                (``Node.prior0``).  ``frac = 0`` draws nothing and is exactly plain puct.  Absent
                or ``None`` = off; any other select rule raises ``ValueError``.
* ``chance``  - ``"expect"`` (default, exact as the package convention asks): the first visit of a
                measuring edge expands every outcome child, later visits descend into the outcome
                that is most under-visited relative to its probability, and the edge value is the
                Born-weighted average of the outcome children's current means.  ``"sample"``:
                closed-loop MCTS, each visit samples an outcome by its Born probability and backs
                up the sampled value.
* ``leaf``    - ``"eval"`` (default): evaluator value of the new leaf.  ``"rollout"``: a playout of
                ``rollout_depth`` plies (default 4) with ``rollout_policy`` ``"random"`` (default) or
                ``"greedy"`` (epsilon-greedy on the move heuristic, ``rollout_eps`` default 0.1;
                ``rollout_quantum`` default False keeps playouts free of splits/merges), outcomes
                sampled by Born probability, then the evaluator.  ``"minimax1"``: exact 1-ply
                expectimax at the leaf (max for me, ``gametheory.opp_combine`` for the opponent).
* ``widen_k``, ``widen_alpha`` - progressive widening: only the first ``ceil(k * N^alpha)``
                children (heuristic order: king captures, captures, promotions, king attacks,
                splits; prior order under puct) are eligible.  ``widen_k = 0`` (default) = off;
                ``widen_alpha`` default 0.5.
* ``final``   - default root decision when the config has no ``decide`` key: ``"visits"`` (default,
                gametheory rule ``visits``), ``"mean"`` (rule ``argmax``), ``"robust"`` (rule
                ``visits`` = max visits, tie by mean, plus max-robust stopping: the search may stop
                once the soft budget is used *and* the most-visited root child is also the
                best-mean child; otherwise it continues to the hard budget).
* ``reuse``   - bool (default False): keep the subtree between moves.  At the next search the
                engine follows the moves and measurement outcomes recorded in ``board.history``
                from its previous root and reuses the node it reaches if its quantum state equals
                the game board's.
* ``opponent``- opponent model at opponent decision nodes (selection minimises my value):
                ``minimax`` (default); ``eps`` = with probability ``eps`` a uniformly random legal
                reply is followed (a fallible opponent); ``softmax`` = once every eligible reply
                has been tried, replies are followed with probability ~ exp(-Q / temp).  Any
                other model raises ``ValueError`` (at construction and again at every search,
                since wrapper engines replace ``opp_cfg`` between moves).
* ``minimax_mix`` - implicit minimax backups (Lanctot et al. 2014, arXiv:1406.0486), extra to
                the brief: the value used for selection and reported at the root is
                ``(1 - mix) * mean + mix * mm`` where ``mm`` is the minimax/expectimax value of
                the explored subtree (leaf evaluations backed up with max / opponent model / Born
                averages).  Default 0 (off).
* ``max_iter`` - optional cap on iterations per search (in addition to the Budget).
* ``eval``, ``decide`` - as in the package conventions; ``decide`` is checked with
                ``gametheory.check_decide`` at construction (before the evaluator is built), so a
                bad rule fails in ``make_engine`` instead of in ``choose`` after a whole search.
* ``quantum`` - allow *my* splits/merges (default True); ``opp_quantum`` - let the modelled
                opponent split/merge (default True).  Same meaning as in ``expectimax``.  Playouts
                use ``rollout_quantum`` for both sides.

Boltzmann selection (``select = "bts"`` / ``"ments"``; Painter, Baioumy, Hawes & Lacerda,
"Monte Carlo Tree Search with Boltzmann Exploration", NeurIPS 2023, arXiv:2404.07732, eq. 15-24 and
Appendix C.2 for two-player games; Xiao, Huang, Mei, Schuurmans & Mueller, "Maximum Entropy
Monte-Carlo Planning", NeurIPS 2019):

* Like puct, a node's first expansion values every legal move exactly one ply deep (``ply1``,
  the Born-weighted evaluator value of its outcome children) and orders the moves by it.  A move
  that has not been visited yet has ``Q = ply1[m]``; a visited one has the edge value below.
* Search policy at a node with ``N`` visits over its eligible moves ``A`` (after widening), with
  ``s = +1`` at my nodes and ``-1`` at the opponent's:
  ``pi(a) = (1 - lam) * softmax_a(s * (Q(a) + beta * H_Q(a)) / T) + lam / |A|`` with
  ``lam = min(1, eps_explore / log(e + N))`` (the E2W mixing in the form Painter et al. restate;
  Xiao et al.'s ``eps |A| / log(N + 1)`` exceeds 1 early).  The move is drawn from ``pi`` with one
  ``rng.random()`` draw (deterministic under a node budget for a fixed game seed).  The
  ``eps`` / ``softmax`` opponent models still act first at opponent nodes.
* ``"bts"``: ``T = temp`` (default 0.1; ``temp_decay`` ``"sqrt"`` gives ``temp / sqrt(N)``,
  ``"log"`` ``temp / log(e + N)``, default none).  ``backup`` ``"bellman"`` (default, BTS proper):
  a decision node's value is the max (opponent nodes: ``gametheory.opp_combine``, the min for
  the minimax model) of ``Q`` over its eligible moves, where ``Q`` of a visited move is the
  Born-weighted average of its outcome children's values and ``ply1`` for an unvisited move; a
  leaf's value is its leaf value, a terminal's its terminal value, and an outcome child not
  visited yet (only possible with ``chance = "sample"``) counts at the value the 1-ply expansion
  gave it, so the average stays a full Born average (Painter et al. weight outcomes by visit
  counts instead; qchess knows the Born probabilities).  ``backup = "mean"``
  (AR-BTS, the variant Painter et al. used in Go): ``Q`` is the usual MCTS mean (``_edge_q``,
  with ``minimax_mix`` if set), pair it with a decaying ``temp_decay``.  DENTS: ``beta0`` > 0
  (default 0 = plain BTS) adds the entropy bonus ``beta = beta0 / log(e + N)`` (``beta_decay =
  "log"``, default, the gridworld schedule) or ``beta0 / sqrt(N)`` (``"sqrt"``, the Go schedule),
  with ``H_V(node) = s * H(pi) + sum_a pi(a) H_Q(a)`` and ``H_Q(edge) = sum_o p_o H_V(child_o)``
  (0 for leaves, terminals and unvisited moves; the sign ``s`` is the paper's min-node form).
* ``"ments"``: soft backups.  A decision node's value is ``s * T * log sum_a exp(s * Q(a) / T)``
  (my nodes ``T log sum exp(Q / T)``, opponent nodes ``-T log sum exp(-Q / T)``, Painter et al.
  eq. 58) over the same ``Q`` as Bellman BTS, clipped to [-1, 1]; ``T = tau / (1 + tau_decay * N)``
  (``tau`` default 0.05, ``tau_decay`` default 0 = no annealing).  The same ``T`` is the
  selection temperature, as in E2W.  ``beta0`` and ``backup`` are not used.  Opponent nodes follow
  the ``opponent`` model in the backup as well as in selection (``soft_opp_value``):
  ``gametheory.opp_combine`` with its hard min replaced by the soft min, i.e. the soft min for
  ``minimax``, ``(1 - eps) * soft min + eps * mean`` for ``eps``, and ``opp_combine`` itself for
  ``softmax``.  As ``T -> 0`` the backup becomes Bellman BTS's (max / ``opp_combine``) under
  every model.
  The soft value is biased: the soft max exceeds the hard max by up to ``T log |A|`` per ply (at
  ``tau`` 0.05 with the moves spread over 0.05: about 0.06 for 5 moves, 0.2 for 118).  That is
  enough to change the argmax of the root ``Q`` (in the stat-mech probe the soft choice differed
  from the hard argmax of the same tree in about a fifth of the searches, at no measurable regret
  cost), and the clip to [-1, 1] makes several forced wins back up the same 1.0, so their ply
  discount no longer ranks them (a king capture at the root is still a proven win, see below).
  ``tau_decay`` > 0 anneals the bias away at well-visited nodes.
* ``eps_explore`` default 0.1 (both).  The root decision defaults to ``final = "mean"`` for both
  rules: the argmax of the root ``Q``, the recommendation policy of the papers' theory (MENTS
  eq. 14, as Painter et al. restate Xiao et al.; BTS eq. 19), which is not what Painter et al.
  played in Go (see *Defaults* below); ``"visits"`` and ``"robust"`` work as for the other rules,
  and an explicit ``decide`` overrides as usual.
  Root candidates' ``mean`` is the root ``Q`` (Bellman or soft value; the mean for AR-BTS) and
  their outcome record holds the outcome children's values of the same kind.  ``minimax_mix`` > 0
  with Bellman BTS or MENTS raises ``ValueError`` (their values are already backed-up values).
* Defaults: ``temp`` 0.1, ``tau`` 0.05 and ``eps_explore`` 0.1 are qchess's own choices for
  values in [-1, 1], not the papers' settings.  ``temp`` 0.1 equals the BTS temperature Painter et
  al. picked in Go (Table 5), but their Go results were scaled to +-100 (Appendix D.5: parameters
  "about 100 times larger"), so on qchess's scale that pick is about 0.001 (their MENTS pick,
  1.0, about 0.01).  In Go they also chose ``eps`` as low as they tried (0.001-0.003, Tables
  9-10), recommended the most-visited move (``argmax N``, more robust to network noise) rather
  than the argmax of ``Q``, used average-return backups (AR-*: Bellman backups propagated the
  network's noise) and mixed the prior policy into the search policy (eq. 52-53, not implemented
  here).  On the probe's 60 positions (``reports/engines/statmech/probe.py``; review follow-up of
  20 October 2026, mean regret vs a 2-ply reference) colder settings were not better (``temp``
  0.1: 0.052, 0.01: 0.054, 0.001 with ``eps_explore`` 0.03: 0.063; puct 0.047), and ``final =
  "visits"`` vs ``"mean"`` gave 0.046 vs 0.052 (0.032 vs 0.036 against a 3-ply reference; not
  significant), so the defaults stay.
* Bookkeeping: ``Node.sv`` / ``Edge.sv`` hold the Bellman or soft values, ``Node.hv`` /
  ``Edge.hq`` the DENTS entropies; every node on the path is updated after each visit (O(|A|)).

Information-theoretic options (added 20 October 2026; both absent = off, and the search is then
bitwise the one before they existed):

* ``prior_info`` - puct only: ``{"gain": g, "std": s}`` (both default 0) adds, before the prior's
  softmax, ``g * h(p1) + s * sigma`` (value units) to every move's 1-ply value, where ``h(p1)`` is the
  binary entropy in bits of a measuring move's outcome (the Shannon information the measurement
  reveals; 0 for a unitary move) and ``sigma`` the Born-weighted spread of its outcome children's
  1-ply values.  The bonus is added for the side to move at every node (my moves and the modelled
  opponent's alike): it widens the search towards lotteries and informative measurements.  Only the
  prior P and the move order change; first-play urgency stays the plain 1-ply value.
* ``time_manager`` - adaptive time per move with a per-game bank, used by ``choose`` only (a wrapper
  that calls ``analyse`` gets a plain search).  ``{}`` or ``true`` = the defaults ``TM_DEFAULTS``.
  The resource is wall-clock seconds, or nodes when the arena's ``Budget`` has a node cap.  With
  ``base`` = the move's budget from the arena and ``bank`` = the sum over this game's earlier moves of
  ``base - used`` (starts at 0 in ``new_game``):

  1. ``hi = min(max * base, base + bank) - reserve`` (seconds only), ``lo = min(min * base, hi)``;
     the search first runs to ``lo``;
  2. at ``lo`` it reads the root: ``unstable`` (most-visited move is not the best-Q move), ``close``
     (``max(0, 1 - gap / close_gap)``, gap = best minus second-best Q), the normalised entropy of the
     root visit distribution, and the Shannon entropy of the root board's classical branches
     (``min(1, H / branch_bits)``); ``u = clip(w_unstable * unstable + w_close * close + w_visits *
     H_visits + w_branch * H_branch, 0, 1)``, ``factor = min + (max - min) * u``;
  3. it continues to ``target = clip(factor * base + bank / spend_moves, lo, hi)``.

  So the move never uses more than ``base`` plus what earlier moves saved, and the game's mean time
  per move cannot exceed ``base`` except by the search's own overrun of a deadline (charged to the
  bank as well, since ``used`` is measured).  Moves played without a search (one legal move) and
  searches that stop early (proven win) add their unused time to the bank.  ``stats()`` logs per
  ``choose`` call ``tm_u``, ``tm_factor`` (target / base), ``tm_used`` (used / base) and ``tm_bank``
  (bank / base after the move); a move without a search logs ``None`` for ``tm_u`` / ``tm_factor``.

Positions with no legal move (rare: every piece blocked) are scored like the arena adjudicates
them, ``king_balance(board, me)`` with the ply discount (0 on a classical board), as soon as the
node is first visited; every later visit backs up that same value.

Root candidates (all from my view): ``mean`` = Q (with ``minimax_mix`` > 0, the mixed value);
``visits`` = N; ``worst`` = min over the outcome children's values (the same, mixed, values that
make up ``mean``; ``mean`` for a unitary move), never above ``mean``; ``p_loss`` = Born mass of
outcome children whose value is <= -0.9; ``std`` = with ``chance = "expect"``, the Born-weighted
spread over the measurement outcomes at the root, ``sqrt(sum p (v_o - mean)^2)``, 0 for a unitary
move (as ``ExpectimaxEngine`` reports it); with ``chance = "sample"``, the spread of the sampled
values backed up through the edge.  ``extra`` = ``{"iterations", "root_visits", "prior", "proven",
"outcomes"}`` (``outcomes`` = ``[(outcome, p, value, visits)]``).

Proven wins: a root move whose every outcome captures the opponent's king outright has
``extra["proven"] == "win"``.  The search stops as soon as one is found, and ``analyse`` then
returns *only* the proven-win candidates, so every decision rule (and every wrapper that applies
one to ``analyse``'s output) plays a king capture.  ``choose`` returns at once, without searching,
when only one move is legal.

Decision log: ``stats()`` holds, per ``choose`` call, v* and the chosen candidate's mean / worst /
std, and under the ``adaptive`` rule the regime per move and per-regime counts (``vstar``,
``chosen_*``, ``regimes``, ``regime_*``; see the ``expectimax`` module docstring, also for why
an engine wrapped by ``trap`` logs nothing: the wrapper calls ``analyse``, not ``choose``).  A
move played without a search logs ``None`` and no regime: one legal move, or the heuristic
fallback when not even one iteration finished (its placeholder candidate, ``extra["fallback"]``,
has ``mean`` 0.0, which is not a search value).  A proven win taken by the fast path in
``choose`` logs regime ``"proven"`` under the adaptive rule, the regime ``decide_adaptive`` gives
any list of proven wins.  The log draws nothing from ``self.rng``, which the search itself draws
from (thompson selection, sampled chance, rollouts, the ``eps`` opponent).

Budget: one ``tick`` per iteration and one per generated board (tree children, playout plies,
1-ply children); the budget is checked before every generated board, and an interrupted
iteration is discarded.  ``choose`` always returns a legal move (the heuristic-best move if not
even one iteration finished).  Boards are generated with ``base.apply`` (the cached-matrix fast
path of ``qchess.engines.fastgen``), bitwise identical to ``copy`` + ``ForcedRNG`` + ``push``.
"""
from __future__ import annotations

import math

import numpy as np

from ..board import QuantumBoard
from ..geometry import JUMPERS, SLIDERS, WHITE, kind
from ..moves import Move, MoveType
from . import gametheory
from .base import (Budget, Candidate, Engine, OutOfTime, apply, king_balance, legal_moves,
                   outcome_probs, terminal_value)
from .expectimax import decide_and_log, log_decision, stats_copy
from .infotheory import binary_entropy, branch_entropy

PLY_DISCOUNT = 0.001
PIECE_VALUE = {"K": 0.0, "Q": 9.0, "R": 5.0, "B": 3.0, "N": 3.0, "P": 1.0}
CAPTURES = {MoveType.CAPTURE_JUMP, MoveType.CAPTURE_SLIDE, MoveType.PAWN_CAPTURE}
BLOCKED = {MoveType.BLOCKED_JUMP, MoveType.BLOCKED_SLIDE, MoveType.BLOCKED_PAWN_STEP}

SELECT_RULES = ("uct", "puct", "thompson", "bts", "ments")
BOLTZMANN_RULES = ("bts", "ments")
BTS_BACKUPS = ("bellman", "mean")
TEMP_DECAYS = (None, "sqrt", "log")
BETA_DECAYS = ("log", "sqrt")
DIRICHLET_DEFAULTS = {"alpha": 0.3, "frac": 0.25}
CHANCE_MODES = ("expect", "sample")
LEAF_MODES = ("eval", "rollout", "minimax1")
FINAL_RULES = {"visits": {"rule": "visits"}, "mean": {"rule": "argmax"}, "robust": {"rule": "visits"}}
OPP_MODELS = ("minimax", "eps", "softmax")
PRIOR_INFO_KEYS = ("gain", "std")
TM_DEFAULTS = {"min": 0.5, "max": 2.0, "w_unstable": 0.5, "w_close": 0.25, "close_gap": 0.05,
               "w_visits": 0.25, "w_branch": 0.25, "branch_bits": 1.0, "spend_moves": 4.0,
               "reserve": 0.01}
TM_LISTS = ("tm_u", "tm_factor", "tm_used", "tm_bank")


def check_opponent(opp_cfg: dict | None) -> str:
    """The opponent model name of ``opp_cfg``; ``ValueError`` for a model MCTS does not know."""
    model = (opp_cfg or {}).get("model", "minimax")
    if model not in OPP_MODELS:
        raise ValueError(f"unknown opponent model {model!r}")
    return model


def _num(cfg: dict, key: str, default: float, lo: float | None = None, lo_open: bool = False) -> float:
    """``float(cfg[key])`` (or ``default``), checked: not a bool, finite (not NaN, not +-inf),
    ``>= lo`` (``> lo`` when ``lo_open``); ``ValueError`` otherwise.  An infinite temperature,
    entropy weight or Dirichlet alpha turns the search's arithmetic into NaN (``inf * 0``,
    ``inf - inf``), so it is refused here instead of crashing or silently degrading the search."""
    x = cfg.get(key, default)
    if isinstance(x, bool):
        raise ValueError(f"mcts: {key}={x!r} must be a number")
    try:
        v = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"mcts: {key}={x!r} must be a number") from e
    if not math.isfinite(v):
        raise ValueError(f"mcts: {key}={x!r} must be a finite number")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ValueError(f"mcts: {key}={v} must be {'>' if lo_open else '>='} {lo}")
    return v


def check_dirichlet(d, select: str) -> dict | None:
    """The root-noise config (``None`` = off) with defaults filled in; ``ValueError`` if bad or if
    ``select`` is not ``"puct"``."""
    if d is None or d is False:
        return None
    if d is True:
        d = {}
    if not isinstance(d, dict):
        raise ValueError(f"mcts: dirichlet must be a dict like {DIRICHLET_DEFAULTS}, got {d!r}")
    unknown = set(d) - set(DIRICHLET_DEFAULTS)
    if unknown:
        raise ValueError(f"mcts: unknown dirichlet keys {sorted(unknown)}")
    if select != "puct":
        raise ValueError(f"mcts: dirichlet root noise needs select 'puct', not {select!r}")
    alpha = _num(d, "alpha", DIRICHLET_DEFAULTS["alpha"], 0.0, lo_open=True)
    frac = _num(d, "frac", DIRICHLET_DEFAULTS["frac"], 0.0)
    if frac > 1.0:
        raise ValueError(f"mcts: dirichlet frac={frac} must lie in [0, 1]")
    return {"alpha": alpha, "frac": frac}


def check_prior_info(d, select: str) -> dict | None:
    """The ``prior_info`` config (``None`` = off) with defaults filled in; ``ValueError`` if bad or
    if ``select`` is not ``"puct"``."""
    if d is None or d is False:
        return None
    if not isinstance(d, dict):
        raise ValueError(f"mcts: prior_info must be a dict like {{'gain': 0.05, 'std': 0.5}}, got {d!r}")
    unknown = set(d) - set(PRIOR_INFO_KEYS)
    if unknown:
        raise ValueError(f"mcts: unknown prior_info keys {sorted(unknown)}")
    if select != "puct":
        raise ValueError(f"mcts: prior_info needs select 'puct', not {select!r}")
    return {"gain": _num(d, "gain", 0.0), "std": _num(d, "std", 0.0)}


def check_time_manager(d) -> dict | None:
    """The ``time_manager`` config (``None`` = off) with ``TM_DEFAULTS`` filled in; ``ValueError``
    if a key is unknown or a value is out of range."""
    if d is None or d is False:
        return None
    if d is True:
        d = {}
    if not isinstance(d, dict):
        raise ValueError(f"mcts: time_manager must be a dict (see TM_DEFAULTS), got {d!r}")
    unknown = set(d) - set(TM_DEFAULTS)
    if unknown:
        raise ValueError(f"mcts: unknown time_manager keys {sorted(unknown)}")
    out = {k: _num(d, k, v, 0.0) for k, v in TM_DEFAULTS.items()}
    if not 0.0 < out["min"] <= 1.0 <= out["max"]:
        raise ValueError(f"mcts: time_manager needs 0 < min <= 1 <= max, got {out['min']}, {out['max']}")
    for k in ("close_gap", "branch_bits", "spend_moves"):
        if out[k] <= 0.0:
            raise ValueError(f"mcts: time_manager {k} must be > 0")
    return out


def soft_value(qs, temp: float, sign: float = 1.0) -> float:
    """``sign * temp * log sum exp(sign * q / temp)``: the soft max (``sign = 1``) or soft min
    (``sign = -1``) of ``qs`` at temperature ``temp`` (stable for any ``temp`` > 0)."""
    z = [sign * q / temp for q in qs]
    m = max(z)
    return sign * temp * (m + math.log(sum(math.exp(x - m) for x in z)))


def soft_opp_value(qs, temp: float, opp_cfg: dict | None) -> float:
    """MENTS's value (my view) of an opponent node whose replies have my values ``qs``:
    ``gametheory.opp_combine`` with its hard min replaced by the soft min at ``temp``.

    * ``minimax``: ``soft_value(qs, temp, -1)`` (the two-player MENTS backup, Painter et al. eq. 58);
    * ``eps``: ``(1 - eps) * soft_min + eps * mean(qs)`` (the rational share plays the soft min,
      the random share a uniform reply, as the ``eps`` selection does);
    * ``softmax``: ``opp_combine`` unchanged, ``E_p[q]`` with ``p ~ exp(-q / temp_opp)`` (the model
      fixes the opponent's whole policy at its own temperature; it has no min to soften).

    As ``temp -> 0`` every case becomes ``opp_combine(qs, opp_cfg)``, the Bellman BTS backup."""
    model = check_opponent(opp_cfg)
    if model == "softmax":
        return float(gametheory.opp_combine(qs, opp_cfg))
    smin = soft_value(qs, temp, -1.0)
    if model == "eps":
        eps = float(opp_cfg.get("eps", 0.1))
        return (1.0 - eps) * smin + eps * (sum(qs) / len(qs))
    return smin


# ---- tree ----------------------------------------------------------------------------------------
class Node:
    """Decision node.  ``tv`` is the (ply-discounted) terminal value or None; ``raw`` the
    undiscounted +1/-1/0.  ``N``/``W`` count visits and the sum of values backed up through it."""
    __slots__ = ("board", "turn", "ply", "raw", "tv", "legal", "moves", "edges", "N", "W", "ev",
                 "ply1", "prior", "mm", "sv", "hv", "prior0")

    def __init__(self, board: QuantumBoard | None = None, me: str | None = None, root_ply: int = 0):
        self.board = board
        self.turn = board.turn if board is not None else None
        self.ply = len(board.history) if board is not None else 0
        raw = terminal_value(board, me) if board is not None else None
        self.raw = raw
        self.tv = None if raw is None else raw * (1.0 - PLY_DISCOUNT * max(0, self.ply - root_ply))
        self.legal: list[Move] | None = None     # legal moves (generation order)
        self.moves: list[Move] | None = None     # legal moves in search order
        self.edges: dict[Move, Edge] = {}
        self.N = 0
        self.W = 0.0
        self.ev: float | None = None             # cached evaluator value
        self.ply1: dict[Move, float] | None = None  # exact 1-ply expected evaluations
        self.prior: dict[Move, float] | None = None
        self.mm: float | None = None             # minimax value of the explored subtree
        self.sv: float | None = None             # bts / ments: Bellman or soft value
        self.hv = 0.0                            # bts (DENTS): entropy value H_V
        self.prior0: dict[Move, float] | None = None   # root prior before Dirichlet noise

    def q(self) -> float | None:
        if self.tv is not None:
            return self.tv
        if self.N:
            return self.W / self.N
        return self.ev


class Edge:
    """A move at a decision node; ``outcomes`` = ``[(outcome, probability)]`` (outcome None for a
    unitary move) and ``children[outcome]`` the resulting decision nodes."""
    __slots__ = ("move", "outcomes", "children", "N", "W", "W2", "mm", "sv", "hq")

    def __init__(self, move: Move, outcomes: list[tuple[int | None, float]]):
        self.move = move
        self.outcomes = outcomes
        self.children: dict[int | None, Node] = {}
        self.N = 0
        self.W = 0.0
        self.W2 = 0.0
        self.mm: float | None = None
        self.sv: float | None = None             # bts / ments: Born average of children's sv
        self.hq = 0.0                            # bts (DENTS): Born average of children's hv

    def expect_q(self) -> float | None:
        """Born-weighted average of the outcome children's current means (None until every
        outcome child has a value)."""
        tot = 0.0
        for o, p in self.outcomes:
            c = self.children.get(o)
            if c is None:
                return None
            q = c.q()
            if q is None:
                return None
            tot += p * q
        return tot

    def sample_q(self) -> float | None:
        return self.W / self.N if self.N else None


# ---- cheap move-ordering heuristic ---------------------------------------------------------------
def _attacks(geo, piece: str, frm: int, to: int, color: str) -> bool:
    k = kind(piece)
    if k in JUMPERS:
        return to in geo.jump_targets(frm, piece)
    if k in SLIDERS:
        return frm != to and geo.valid(to, frm, piece)
    return to in geo.pawn_captures(frm, color)


def heuristic_scores(board: QuantumBoard, moves: list[Move]) -> list[float]:
    """Move-ordering scores: king captures >> captures (by victim value) > promotions > moves
    that attack the enemy king > splits > merges > quiet moves > blocked moves."""
    geo, me = board.geo, board.turn
    enemy_k = "k" if me == WHITE else "K"
    ksq = [(q, board.prob(q)) for q in geo.squares() if board.types[q] == enemy_k]
    out = []
    for m in moves:
        try:
            mt = board.classify(m)
        except ValueError:
            out.append(-1e9)
            continue
        piece = board.types[m.source]
        s = 0.0
        if mt in CAPTURES:
            victim = board.types[m.target]
            pt = board.prob(m.target)
            ps = board.prob(m.source)
            if kind(victim) == "K":
                s += 1000.0 * pt * ps
            else:
                s += 10.0 + PIECE_VALUE[kind(victim)] * pt * ps
        elif mt in BLOCKED:
            s -= 2.0
        if kind(piece) == "P" and mt not in BLOCKED and geo.xy(m.target)[1] == geo.last_rank(me):
            s += 8.0
        targets = [m.target] if m.target2 is None else [m.target, m.target2]
        for t in targets:
            for q, pk in ksq:
                if _attacks(geo, piece, t, q, me):
                    s += 5.0 * pk / len(targets)
        if m.is_split:
            s += 1.0
        elif m.is_merge:
            s += 0.5
        out.append(s)
    return out


def _clip(v: float) -> float:
    return -1.0 if v < -1.0 else 1.0 if v > 1.0 else float(v)


def same_position(a: QuantumBoard, b: QuantumBoard, tol: float = 1e-9) -> bool:
    """Same side to move, piece types and quantum state (amplitudes, incl. ancillas)."""
    if a.turn != b.turn or a.types != b.types or a.state.nq != b.state.nq:
        return False
    if set(a.state.amps) != set(b.state.amps):
        return False
    return all(abs(a.state.amps[k] - b.state.amps[k]) <= tol for k in a.state.amps)


class _Done(Exception):
    """Internal: the root has no legal moves."""


# ---- the engine ----------------------------------------------------------------------------------
class MCTSEngine(Engine):
    def __init__(self, cfg: dict | None = None, evaluator=None):
        super().__init__(cfg)
        cfg = self.cfg
        self.select = cfg.get("select", "uct")
        if self.select not in SELECT_RULES:
            raise ValueError(f"unknown select rule {self.select!r}")
        self.c = float(cfg.get("c", 1.0))
        self.prior_temp = max(float(cfg.get("prior_temp", 0.2)), 1e-6)
        self.prior_n = max(float(cfg.get("prior_n", 1.0)), 1e-6)
        self.sigma = float(cfg.get("sigma", 0.5))
        self.chance = cfg.get("chance", "expect")
        if self.chance not in CHANCE_MODES:
            raise ValueError(f"unknown chance mode {self.chance!r}")
        self.leaf = cfg.get("leaf", "eval")
        if self.leaf not in LEAF_MODES:
            raise ValueError(f"unknown leaf mode {self.leaf!r}")
        self.rollout_depth = int(cfg.get("rollout_depth", 4))
        self.rollout_policy = cfg.get("rollout_policy", "random")
        if self.rollout_policy not in ("random", "greedy"):
            raise ValueError(f"unknown rollout policy {self.rollout_policy!r}")
        self.rollout_eps = float(cfg.get("rollout_eps", 0.1))
        self.rollout_quantum = bool(cfg.get("rollout_quantum", False))
        self.widen_k = float(cfg.get("widen_k", 0.0))
        self.widen_alpha = float(cfg.get("widen_alpha", 0.5))
        self.final = cfg.get("final", "mean" if self.select in BOLTZMANN_RULES else "visits")
        if self.final not in FINAL_RULES:
            raise ValueError(f"unknown final rule {self.final!r}")
        self.reuse = bool(cfg.get("reuse", False))
        self.minimax_mix = float(cfg.get("minimax_mix", 0.0))
        # Boltzmann selection (bts / ments) and root noise (puct); see the module docstring
        self._boltz = self.select in BOLTZMANN_RULES
        self.temp = _num(cfg, "temp", 0.1, 0.0, lo_open=True)
        self.temp_decay = cfg.get("temp_decay")
        if self.temp_decay not in TEMP_DECAYS:
            raise ValueError(f"mcts: temp_decay={self.temp_decay!r}, expected one of {TEMP_DECAYS}")
        self.eps_explore = _num(cfg, "eps_explore", 0.1, 0.0)
        self.beta0 = _num(cfg, "beta0", 0.0, 0.0)
        self.beta_decay = cfg.get("beta_decay", "log")
        if self.beta_decay not in BETA_DECAYS:
            raise ValueError(f"mcts: beta_decay={self.beta_decay!r}, expected one of {BETA_DECAYS}")
        self.backup = cfg.get("backup", "bellman")
        if self.backup not in BTS_BACKUPS:
            raise ValueError(f"mcts: backup={self.backup!r}, expected one of {BTS_BACKUPS}")
        self.tau = _num(cfg, "tau", 0.05, 0.0, lo_open=True)
        self.tau_decay = _num(cfg, "tau_decay", 0.0, 0.0)
        # values read from Node.sv / Edge.sv instead of visit means: Bellman BTS and MENTS
        self._sv_mode = self.select == "ments" or (self.select == "bts" and self.backup == "bellman")
        self._dents = self.select == "bts" and self.beta0 > 0
        if self._sv_mode and self.minimax_mix > 0:
            raise ValueError(f"mcts: minimax_mix does not apply to select {self.select!r} with "
                             f"backed-up (Bellman / soft) values")
        self.dirichlet = check_dirichlet(cfg.get("dirichlet"), self.select)
        self._noised = False
        self.max_iter = cfg.get("max_iter")
        self.prior_info = check_prior_info(cfg.get("prior_info"), self.select)
        self.tm = check_time_manager(cfg.get("time_manager"))
        self._tm_bank = 0.0
        self._tm_live: dict | None = None
        self._game_info: dict = {}
        self.allow_quantum = cfg.get("quantum", True)
        self.opp_quantum = cfg.get("opp_quantum", True)
        self.decide_cfg = cfg.get("decide", dict(FINAL_RULES[self.final]))
        self.opp_cfg = cfg.get("opponent", {"model": "minimax"})
        check_opponent(self.opp_cfg)
        gametheory.check_decide(self.decide_cfg, "mcts")
        if evaluator is None:
            from .evaluators import make_evaluator
            evaluator = make_evaluator(cfg.get("eval", {"kind": "material"}))
            self._owns_eval = True
        else:
            self._owns_eval = False
        self.evaluator = evaluator
        self._prev: Node | None = None
        self._prev_me: str | None = None
        self._spent: Node | None = None          # last search tree, freed outside the move clock
        self._me = WHITE
        self._root_ply = 0
        self._budget: Budget | None = None
        self._stats = self._fresh_stats()

    @staticmethod
    def _fresh_stats() -> dict:
        return {"searches": 0, "iterations": 0, "reused": 0, "boards": 0, "aborted": 0, "time": 0.0}

    # ---- Engine interface ----------------------------------------------------------------------
    def new_game(self, color: str, seed: int) -> None:
        super().new_game(color, seed)
        self._stats = self._fresh_stats()
        self._prev = None
        self._prev_me = None
        self._spent = None
        self._tm_bank = 0.0
        self._tm_live = None

    def observe(self, before, move, result, after) -> None:
        # Free the previous search tree here (the arena calls observe outside the move clock):
        # deallocating a few thousand nodes and boards took 2-5 ms on an idle machine and up to
        # ~30 ms under load, all charged as an overrun when it happened on analyse's return.
        self._spent = None

    def choose(self, board: QuantumBoard, budget: Budget) -> Move:
        if self.tm is None:
            return self._choose(board, budget)
        self._tm_begin(budget)
        try:
            return self._choose(board, budget)
        finally:
            self._tm_end(board, budget)

    def _choose(self, board: QuantumBoard, budget: Budget) -> Move:
        ms = legal_moves(board, self.allow_quantum)
        if len(ms) == 1:
            log_decision(self._stats, [], None)       # no search, nothing to log but the move
            return ms[0]
        cands = self.analyse(board, budget)
        # analyse() already keeps only the proven wins when there is one; this is a fast path
        for c in cands:
            if c.extra.get("proven") == "win":
                adaptive = (self.decide_cfg or {}).get("rule", "argmax") == "adaptive"
                log_decision(self._stats, cands, c, "proven" if adaptive else None)
                return c.move
        return decide_and_log(cands, self.decide_cfg, self.rng, self._stats).move

    def stats(self) -> dict:
        return stats_copy(self._stats)

    def set_game_info(self, info: dict) -> None:
        """Arena hook (ply cap, start ply): forwarded to the evaluator, which needs it for the
        tablebase kind (plies left in the game)."""
        self._game_info = dict(info or {})
        hook = getattr(self.evaluator, "set_game_info", None)
        if hook is not None:
            hook(info)

    def close(self) -> None:
        if self._owns_eval and hasattr(self.evaluator, "close"):
            self.evaluator.close()
        self._prev = None
        self._spent = None

    # ---- search driver ---------------------------------------------------------------------------
    def analyse(self, board: QuantumBoard, budget: Budget) -> list[Candidate]:
        check_opponent(self.opp_cfg)          # wrappers may have replaced it since __init__
        self._spent = None                     # no observe() since the last search: free it now
        self._budget = budget
        self._me = board.turn
        self._root_ply = len(board.history)
        self._noised = False                   # Dirichlet root noise: drawn once per search
        if self._tm_live is not None:
            self._tm_live["searched"] = True
        root = self._reused_root(board) if self.reuse else None
        if root is None:
            root = Node(self._clone(board), self._me, self._root_ply)
        else:
            self._stats["reused"] += 1
        if root.N == 0:          # the root's own leaf value is never needed
            root.N, root.W = 1, 0.0
        iters = 0
        while True:
            if budget.expired() and not self._tm_extend(root, budget):
                break
            if self.max_iter is not None and iters >= int(self.max_iter):
                break
            if self.final == "robust" and budget.soft_expired() and self._robust_settled(root):
                break
            try:
                edge = self._iterate(root)
            except OutOfTime:
                if self._tm_extend(root, budget):
                    continue
                self._stats["aborted"] += 1
                break
            except _Done:
                break
            iters += 1
            if self._proven(edge) == "win":
                break
        self._stats["searches"] += 1
        self._stats["iterations"] += iters
        self._stats["time"] += budget.elapsed()
        if self.reuse:
            self._prev, self._prev_me = root, self._me
        out = self._candidates(root, board, iters)
        self._spent = root                     # freed by observe() / the next search, off the clock
        return out

    def _iterate(self, root: Node) -> Edge:
        self._budget.tick()
        self._ensure_moves(root)
        if not root.moves:
            raise _Done
        if self.dirichlet is not None and not self._noised:
            self._noise_root(root)
        edge = self._select(root)
        v = self._visit_edge(root, edge)
        root.N += 1
        root.W += v
        if self.minimax_mix > 0:
            self._update_node_mm(root)
        if self._boltz:
            self._update_node_boltz(root)
        return edge

    def _noise_root(self, root: Node) -> None:
        """AlphaZero root noise: ``prior = (1 - frac) * prior0 + frac * Dirichlet(alpha)`` over the
        root moves (in the clean prior's order), one draw from ``self.rng`` per search."""
        self._noised = True
        if root.prior is None:
            return
        if root.prior0 is None:
            root.prior0 = dict(root.prior)
        frac = self.dirichlet["frac"]
        if frac <= 0.0:
            return
        eta = self.rng.dirichlet([self.dirichlet["alpha"]] * len(root.moves))
        root.prior = {m: (1.0 - frac) * root.prior0[m] + frac * float(x) for m, x in zip(root.moves, eta)}

    def _robust_settled(self, root: Node) -> bool:
        vis = [e for e in root.edges.values() if e.N > 0]
        if not vis:
            return False
        most = max(vis, key=lambda e: e.N)
        best = max(vis, key=lambda e: self._edge_q(e))
        return most is best or self._edge_q(most) >= self._edge_q(best) - 1e-12

    # ---- recursion -------------------------------------------------------------------------------
    def _visit(self, node: Node) -> float:
        """One visit of a decision node; returns the value backed up to its parent edge."""
        if node.tv is None and node.N == 0:
            self._mark_no_moves(node)
        if node.tv is not None:       # terminal, or no legal move (a fixed adjudicated value)
            node.N += 1
            node.W += node.tv
            node.mm = node.tv
            if self._boltz:
                node.sv, node.hv = node.tv, 0.0
            return node.tv
        if node.N == 0:
            v = self._leaf_value(node)
            node.N, node.W = 1, v
            if node.mm is None:
                node.mm = v
            if self._boltz and node.sv is None:
                node.sv = v
            return v
        self._ensure_moves(node)
        if not node.moves:            # defensive: the first-visit check already caught this
            self._mark_no_moves(node)
            return self._visit(node)
        edge = self._select(node)
        v = self._visit_edge(node, edge)
        node.N += 1
        node.W += v
        if self.minimax_mix > 0:
            self._update_node_mm(node)
        if self._boltz:
            self._update_node_boltz(node)
        return v

    def _visit_edge(self, node: Node, edge: Edge) -> float:
        if self.chance == "expect":
            fresh = [o for o, _ in edge.outcomes
                     if o not in edge.children or edge.children[o].N == 0]
            if fresh:
                for o in fresh:
                    self._visit(self._child(node, edge, o))
            else:
                tot = sum(edge.children[o].N for o, _ in edge.outcomes)
                o = max(edge.outcomes, key=lambda op: op[1] * (tot + 1) - edge.children[op[0]].N)[0]
                self._visit(edge.children[o])
            v = edge.expect_q()
        else:
            if len(edge.outcomes) == 1:
                o = edge.outcomes[0][0]
            else:
                ps = np.array([p for _, p in edge.outcomes])
                o = edge.outcomes[int(self.rng.choice(len(ps), p=ps / ps.sum()))][0]
            v = self._visit(self._child(node, edge, o))
        edge.N += 1
        edge.W += v
        edge.W2 += v * v
        if self.minimax_mix > 0:
            self._update_edge_mm(edge)
        if self._boltz:
            self._update_edge_boltz(edge)
        return v

    # ---- tree construction -----------------------------------------------------------------------
    def _clone(self, board: QuantumBoard) -> QuantumBoard:
        """``board.copy()`` without advancing the caller's board RNG."""
        st = board.rng.bit_generator.state
        c = board.copy()
        board.rng.bit_generator.state = st
        return c

    def _apply(self, board: QuantumBoard, m: Move, outcome: int | None) -> QuantumBoard:
        """The child board, via ``base.apply`` (fastgen: bitwise the same as ``copy`` +
        ``ForcedRNG`` + ``push``, 7-15x faster).  ``board`` is always one of the search's own
        boards (the root is a ``_clone`` of the caller's), so the caller's RNG is untouched."""
        if self._budget.expired():
            raise OutOfTime
        self._budget.tick()
        self._stats["boards"] += 1
        return apply(board, m, outcome)

    def _allow(self, node: Node) -> bool:
        """Split/merge moves allowed at this node: mine (``quantum``) or the opponent's
        (``opp_quantum``)."""
        return bool(self.allow_quantum if node.turn == self._me else self.opp_quantum)

    def _mark_no_moves(self, node: Node) -> None:
        """If the side to move has no legal move, give the node a fixed value: the arena's
        adjudication ``king_balance`` (0 on a classical board), ply-discounted like ``tv``."""
        if node.legal is None:
            node.legal = legal_moves(node.board, self._allow(node))
        if not node.legal:
            node.moves = []
            disc = 1.0 - PLY_DISCOUNT * max(0, node.ply - self._root_ply)
            node.tv = _clip(king_balance(node.board, self._me) * disc)

    def _edge(self, node: Node, m: Move) -> Edge:
        e = node.edges.get(m)
        if e is None:
            e = Edge(m, outcome_probs(node.board, m))
            node.edges[m] = e
        return e

    def _child(self, node: Node, edge: Edge, o: int | None) -> Node:
        c = edge.children.get(o)
        if c is None:
            c = Node(self._apply(node.board, edge.move, o), self._me, self._root_ply)
            edge.children[o] = c
        return c

    def _ensure_moves(self, node: Node) -> None:
        if node.moves is not None:
            return
        if node.legal is None:
            node.legal = legal_moves(node.board, self._allow(node))
        legal = node.legal
        if not legal:
            node.moves = []
            return
        if self.select == "puct" or self._boltz:
            vals = self._one_ply(node)
            sgn = 1.0 if node.turn == self._me else -1.0
            if self.prior_info is not None:
                bonus = self._info_bonus(node)
                z = np.array([sgn * vals[m] + bonus[m] for m in legal]) / self.prior_temp
            else:
                z = np.array([sgn * vals[m] for m in legal]) / self.prior_temp
            p = np.exp(z - z.max())
            p /= p.sum()
            node.prior = {m: float(pi) for m, pi in zip(legal, p)}
            order = sorted(range(len(legal)), key=lambda i: -p[i])
        else:
            sc = heuristic_scores(node.board, legal)
            order = sorted(range(len(legal)), key=lambda i: -sc[i])
        node.moves = [legal[i] for i in order]

    def _one_ply(self, node: Node) -> dict[Move, float]:
        """Exact expected evaluator value (my view) of every legal move; resumable if the budget
        interrupts it (children and their evaluations stay cached in the tree)."""
        if node.ply1 is not None:
            return node.ply1
        if node.legal is None:
            node.legal = legal_moves(node.board, self._allow(node))
        out = {}
        for m in node.legal:
            e = self._edge(node, m)
            tot = 0.0
            for o, p in e.outcomes:
                c = self._child(node, e, o)
                tot += p * (c.tv if c.tv is not None else self._ev(c))
            out[m] = tot
        node.ply1 = out
        return out

    def _info_bonus(self, node: Node) -> dict[Move, float]:
        """``prior_info``: ``gain * h(p1) + std * sigma`` per legal move (value units), from the
        edges and outcome children ``_one_ply`` has already built and evaluated."""
        g, s = self.prior_info["gain"], self.prior_info["std"]
        out = {}
        for m in node.legal:
            e = node.edges[m]
            b = 0.0
            if len(e.outcomes) > 1:
                p1 = sum(p for o, p in e.outcomes if o == 1)
                if g:
                    b += g * binary_entropy(p1)
                if s:
                    vs = [(p, c.tv if c.tv is not None else c.ev)
                          for o, p in e.outcomes for c in (e.children[o],)]
                    mu = sum(p * v for p, v in vs)
                    b += s * math.sqrt(max(0.0, sum(p * (v - mu) ** 2 for p, v in vs)))
            out[m] = b
        return out

    # ---- time manager (time_manager; module docstring) -------------------------------------------
    def _tm_begin(self, budget: Budget) -> None:
        tm = self.tm
        nodes = budget.nodes is not None
        base = float(budget.nodes if nodes else budget.seconds)
        bank = self._tm_bank
        hi = min(tm["max"] * base, base + bank)
        if not nodes:
            hi -= tm["reserve"]
        hi = max(hi, 0.05 * base)
        lo = min(tm["min"] * base, hi)
        self._tm_live = {"nodes": nodes, "base": base, "lo": lo, "hi": hi, "decided": False, "u": None,
                         "target": lo}
        self._tm_set(budget, lo)

    def _tm_set(self, budget: Budget, x: float) -> None:
        if self._tm_live["nodes"]:
            budget.nodes = max(1, int(round(x)))
        else:
            budget.seconds = x

    def _tm_extend(self, root: Node, budget: Budget) -> bool:
        """At ``lo``: decide this move's target once; True if the search should go on."""
        live = self._tm_live
        if live is None or live["decided"]:
            return False
        live["decided"] = True
        tm = self.tm
        u = self._tm_uncertainty(root)
        factor = tm["min"] + (tm["max"] - tm["min"]) * u
        target = factor * live["base"] + max(self._tm_bank, 0.0) / tm["spend_moves"]
        target = min(max(target, live["lo"]), live["hi"])
        live["u"], live["target"] = u, target
        self._tm_set(budget, target)
        return not budget.expired()

    def _tm_uncertainty(self, root: Node) -> float:
        tm = self.tm
        vis = [e for e in root.edges.values() if e.N > 0]
        hb = branch_entropy(root.board)
        u = tm["w_branch"] * min(1.0, hb / tm["branch_bits"])
        if len(vis) >= 2:
            most = max(vis, key=lambda e: e.N)
            qs = sorted(((self._edge_q(e), i) for i, e in enumerate(vis)), reverse=True)
            best = vis[qs[0][1]]
            if most is not best and self._edge_q(most) < qs[0][0] - 1e-12:
                u += tm["w_unstable"]
            gap = qs[0][0] - qs[1][0]
            u += tm["w_close"] * max(0.0, 1.0 - gap / tm["close_gap"])
            n = float(sum(e.N for e in vis))
            h = -sum((e.N / n) * math.log(e.N / n) for e in vis)
            u += tm["w_visits"] * h / math.log(len(vis))
        return min(max(u, 0.0), 1.0)

    def _tm_end(self, board: QuantumBoard, budget: Budget) -> None:
        live = self._tm_live
        self._tm_live = None
        if live is None:
            return
        used = float(budget.used_nodes) if live["nodes"] else budget.elapsed()
        self._tm_bank += live["base"] - used
        base = live["base"] or 1.0
        searched = live["decided"] or live.get("searched", False)
        st = self._stats
        for k, v in zip(TM_LISTS, (live["u"], live["target"] / base if searched else None, used / base,
                                   self._tm_bank / base)):
            st.setdefault(k, []).append(None if v is None else round(float(v), 4))

    # ---- leaf values -----------------------------------------------------------------------------
    def _ev(self, node: Node) -> float:
        if node.ev is None:
            node.ev = _clip(self.evaluator(node.board, self._me))
        return node.ev

    def _leaf_value(self, node: Node) -> float:
        if self.leaf == "rollout":
            return self._rollout(node)
        if self.leaf == "minimax1":
            return self._minimax1(node)
        return self._ev(node)

    def _minimax1(self, node: Node) -> float:
        vals = self._one_ply(node)
        if not vals:                  # no legal move (normally caught by _mark_no_moves)
            disc = 1.0 - PLY_DISCOUNT * max(0, node.ply - self._root_ply)
            return _clip(king_balance(node.board, self._me) * disc)
        v = list(vals.values())
        if node.turn == self._me:
            return max(v)
        return float(gametheory.opp_combine(v, self.opp_cfg))

    def _rollout(self, node: Node) -> float:
        b = node.board
        depth = node.ply - self._root_ply
        for _ in range(self.rollout_depth):
            raw = terminal_value(b, self._me)
            if raw is not None:
                return raw * (1.0 - PLY_DISCOUNT * depth)
            ms = legal_moves(b, self.rollout_quantum)
            if not ms:                # adjudicated as the arena does
                return _clip(king_balance(b, self._me) * (1.0 - PLY_DISCOUNT * depth))
            if self.rollout_policy == "greedy" and self.rng.random() >= self.rollout_eps:
                sc = np.array(heuristic_scores(b, ms))
                top = np.flatnonzero(sc == sc.max())
                m = ms[int(top[self.rng.integers(len(top))])]
            else:
                m = ms[int(self.rng.integers(len(ms)))]
            outs = outcome_probs(b, m)
            if len(outs) == 1:
                o = outs[0][0]
            else:
                o = outs[0][0] if self.rng.random() < outs[0][1] else outs[1][0]
            b = self._apply(b, m, o)
            depth += 1
        raw = terminal_value(b, self._me)
        if raw is not None:
            return raw * (1.0 - PLY_DISCOUNT * depth)
        return _clip(self.evaluator(b, self._me))

    # ---- values and selection --------------------------------------------------------------------
    def _edge_q(self, e: Edge) -> float:
        if self._sv_mode and e.sv is not None:   # Bellman BTS / MENTS: the backed-up value
            return e.sv
        q = e.expect_q() if self.chance == "expect" else None
        if q is None:
            q = e.sample_q()
        if q is None:
            q = 0.0
        if self.minimax_mix > 0 and e.mm is not None:
            q = (1.0 - self.minimax_mix) * q + self.minimax_mix * e.mm
        return q

    def _update_edge_mm(self, e: Edge) -> None:
        tot = wsum = 0.0
        for o, p in e.outcomes:
            c = e.children.get(o)
            if c is not None and c.mm is not None:
                tot += p * c.mm
                wsum += p
        e.mm = tot / wsum if wsum > 0 else None

    def _update_node_mm(self, node: Node) -> None:
        vals = [e.mm for e in node.edges.values() if e.mm is not None and e.N > 0]
        if not vals:
            return
        node.mm = max(vals) if node.turn == self._me else float(gametheory.opp_combine(vals, self.opp_cfg))

    def _width(self, node: Node) -> int:
        n = len(node.moves)
        if self.widen_k <= 0:
            return n
        return max(1, min(n, int(math.ceil(self.widen_k * max(node.N, 1) ** self.widen_alpha))))

    def _select(self, node: Node) -> Edge:
        mine = node.turn == self._me
        moves = node.moves
        allowed = moves[: self._width(node)]
        model = check_opponent(self.opp_cfg)
        if not mine and model == "eps":
            if self.rng.random() < float(self.opp_cfg.get("eps", 0.1)):
                return self._edge(node, moves[int(self.rng.integers(len(moves)))])
        if not mine and model == "softmax":
            es = [node.edges.get(m) for m in allowed]
            if all(e is not None and e.N > 0 for e in es):
                temp = max(float(self.opp_cfg.get("temp", 0.05)), 1e-6)
                q = np.array([self._edge_q(e) for e in es])
                p = np.exp(-(q - q.min()) / temp)
                return es[int(self.rng.choice(len(es), p=p / p.sum()))]
        if self._boltz:
            return self._select_boltzmann(node, allowed)
        if self.select == "thompson":
            return self._select_thompson(node, allowed, mine)
        if self.select == "puct":
            return self._select_puct(node, allowed, mine)
        return self._select_uct(node, allowed, mine)

    def _select_uct(self, node: Node, allowed: list[Move], mine: bool) -> Edge:
        log_n = math.log(max(node.N, 1))
        best, best_s = None, -math.inf
        for m in allowed:
            e = node.edges.get(m)
            if e is None or e.N == 0:
                return self._edge(node, m)
            q = self._edge_q(e)
            s = (q if mine else -q) + self.c * math.sqrt(log_n / e.N)
            if s > best_s:
                best, best_s = e, s
        return best

    def _select_puct(self, node: Node, allowed: list[Move], mine: bool) -> Edge:
        sqrt_n = math.sqrt(max(node.N, 1))
        best_m, best_s = None, -math.inf
        for m in allowed:
            e = node.edges.get(m)
            n = e.N if e is not None else 0
            q = self._edge_q(e) if n else node.ply1[m]
            s = (q if mine else -q) + self.c * node.prior[m] * sqrt_n / (1 + n)
            if s > best_s:
                best_m, best_s = m, s
        return self._edge(node, best_m)

    def _select_thompson(self, node: Node, allowed: list[Move], mine: bool) -> Edge:
        parent_q = node.q() or 0.0
        n0, s0 = self.prior_n, self.sigma * self.sigma
        best_m, best_t = None, -math.inf
        for m in allowed:
            e = node.edges.get(m)
            mu0 = node.ply1[m] if node.ply1 is not None and m in node.ply1 else parent_q
            n = e.N if e is not None else 0
            if n == 0:
                mean, var = mu0, s0 / n0
            else:
                q = self._edge_q(e)
                ss = max(0.0, e.W2 - e.W * e.W / n)       # sum of squared deviations
                var = ((n0 * s0 + ss) / (n0 + n)) / (n0 + n)
                mean = (n0 * mu0 + n * q) / (n0 + n)
            t = mean + math.sqrt(var) * float(self.rng.standard_normal())
            t = t if mine else -t
            if t > best_t:
                best_m, best_t = m, t
        return self._edge(node, best_m)

    # ---- Boltzmann selection and backups (bts / ments) ----------------------------------------------
    def _temperature(self, node: Node) -> float:
        """Selection temperature at ``node`` (and MENTS's backup temperature)."""
        n = max(node.N, 1)
        if self.select == "ments":
            t = self.tau / (1.0 + self.tau_decay * node.N)
        elif self.temp_decay == "sqrt":
            t = self.temp / math.sqrt(n)
        elif self.temp_decay == "log":
            t = self.temp / math.log(math.e + node.N)
        else:
            t = self.temp
        return max(t, 1e-12)

    def _beta(self, node: Node) -> float:
        """DENTS entropy weight at ``node`` (0 unless ``select = "bts"`` and ``beta0`` > 0)."""
        if not self._dents:
            return 0.0
        if self.beta_decay == "sqrt":
            return self.beta0 / math.sqrt(max(node.N, 1))
        return self.beta0 / math.log(math.e + node.N)

    def _explore_lambda(self, node: Node) -> float:
        """E2W uniform-mixing weight ``min(1, eps / log(e + N))``."""
        return min(1.0, self.eps_explore / math.log(math.e + node.N))

    def _boltz_q(self, node: Node, allowed: list[Move]) -> tuple[list[float], list[float]]:
        """``(Q, H_Q)`` of the eligible moves: a visited edge's value (``_edge_q``: Bellman / soft
        value, or the mean for AR-BTS) and entropy, else the exact 1-ply value and 0."""
        qs, hs = [], []
        for m in allowed:
            e = node.edges.get(m)
            if e is not None and e.N > 0:
                qs.append(self._edge_q(e))
                hs.append(e.hq)
            else:
                qs.append(node.ply1[m])
                hs.append(0.0)
        return qs, hs

    def _boltz_policy(self, node: Node, qs: list[float], hs: list[float]) -> list[float]:
        """``pi = (1 - lam) softmax(s (Q + beta H_Q) / T) + lam / |A|`` (module docstring)."""
        sgn = 1.0 if node.turn == self._me else -1.0
        t = self._temperature(node)
        beta = self._beta(node)
        z = [sgn * (q + beta * h) / t for q, h in zip(qs, hs)]
        mx = max(z)
        w = [math.exp(x - mx) for x in z]
        tot = sum(w)
        lam = self._explore_lambda(node)
        k = len(qs)
        return [(1.0 - lam) * x / tot + lam / k for x in w]

    def _select_boltzmann(self, node: Node, allowed: list[Move]) -> Edge:
        qs, hs = self._boltz_q(node, allowed)
        pi = self._boltz_policy(node, qs, hs)
        u = float(self.rng.random())
        acc, i = 0.0, len(pi) - 1
        for j, p in enumerate(pi):
            acc += p
            if u < acc:
                i = j
                break
        return self._edge(node, allowed[i])

    @staticmethod
    def _sv_of(c: Node) -> float | None:
        """An outcome child's Bellman / soft value: its ``sv`` once visited, else the value the
        1-ply expansion gave it (terminal value or evaluator value), else None."""
        if c.sv is not None:
            return c.sv
        return c.tv if c.tv is not None else c.ev

    def _update_edge_boltz(self, e: Edge) -> None:
        """``sv`` / ``hq`` = Born-weighted average of the outcome children's values (``_sv_of``)
        / ``hv``, renormalised over the children that have a value."""
        tot = h = wsum = 0.0
        for o, p in e.outcomes:
            c = e.children.get(o)
            v = None if c is None else self._sv_of(c)
            if v is not None:
                tot += p * v
                h += p * c.hv
                wsum += p
        if wsum > 0:
            e.sv = tot / wsum
            e.hq = h / wsum

    def _update_node_boltz(self, node: Node) -> None:
        """Back up an expanded decision node: Bellman (max / opponent model) or soft value over
        its eligible moves, and the DENTS entropy value."""
        if not node.moves or node.ply1 is None or not (self._sv_mode or self._dents):
            return
        allowed = node.moves[: self._width(node)]
        qs, hs = self._boltz_q(node, allowed)
        mine = node.turn == self._me
        if self.select == "ments":
            t = self._temperature(node)
            node.sv = _clip(soft_value(qs, t, 1.0) if mine else soft_opp_value(qs, t, self.opp_cfg))
        elif self.backup == "bellman":
            node.sv = max(qs) if mine else float(gametheory.opp_combine(qs, self.opp_cfg))
        if self._dents:
            pi = self._boltz_policy(node, qs, hs)
            ent = -sum(p * math.log(p) for p in pi if p > 0)
            node.hv = (ent if mine else -ent) + sum(p * h for p, h in zip(pi, hs))

    # ---- root bookkeeping ------------------------------------------------------------------------
    def _proven(self, e: Edge | None) -> str | None:
        if e is None:
            return None
        raws = []
        for o, _ in e.outcomes:
            c = e.children.get(o)
            if c is None or c.raw is None:
                return None
            raws.append(c.raw)
        if all(r > 0 for r in raws):
            return "win"
        if all(r < 0 for r in raws):
            return "loss"
        return None

    def _child_value(self, c: Node) -> float | None:
        """An outcome child's value as it enters its edge's ``_edge_q``: the mean, mixed with
        the subtree's minimax value when ``minimax_mix`` > 0 (Bellman BTS / MENTS: its ``sv``)."""
        if self._sv_mode:
            v = self._sv_of(c)
            if v is not None:
                return v
        q = c.q()
        if q is None:
            return None
        if self.minimax_mix > 0 and c.mm is not None:
            return (1.0 - self.minimax_mix) * q + self.minimax_mix * c.mm
        return q

    def _candidates(self, root: Node, board: QuantumBoard, iters: int) -> list[Candidate]:
        out, wins = [], []
        order = root.moves or []
        for m in order:
            e = root.edges.get(m)
            if e is None:
                continue
            proven = self._proven(e)
            if e.N == 0 and proven != "win":   # (a proven win can be unvisited under puct)
                continue
            kq = []
            for o, p in e.outcomes:
                c = e.children.get(o)
                v = None if c is None else self._child_value(c)
                if v is not None:
                    kq.append((o, p, float(v), c.N))
            if proven == "win":
                mean = e.expect_q()                  # exact: every outcome is terminal
            else:
                mean = self._edge_q(e)
            p_loss = sum(p for _, p, v, _ in kq if v <= -0.9)
            if len(e.outcomes) > 1 and kq:
                worst = min(mean, min(v for _, _, v, _ in kq))
            else:
                worst = mean
            if self.chance == "expect" or proven == "win":
                # Born-weighted spread over the measurement outcomes (0 for a unitary move)
                wsum = sum(p for _, p, _, _ in kq)
                var = 0.0
                if len(kq) > 1 and wsum > 0:
                    mu = sum(p * v for _, p, v, _ in kq) / wsum
                    var = sum(p * (v - mu) ** 2 for _, p, v, _ in kq) / wsum
            else:
                var = max(0.0, e.W2 / e.N - (e.W / e.N) ** 2)
            cand = Candidate(m, float(mean), float(worst), float(math.sqrt(max(var, 0.0))), int(e.N),
                             float(p_loss),
                             {"iterations": iters, "root_visits": root.N,
                              "prior": None if root.prior is None else root.prior.get(m),
                              "proven": proven, "outcomes": kq})
            (wins if proven == "win" else out).append(cand)
        if wins:
            # A certain king capture: return only those moves, so that any decision rule applied
            # to this list (by choose() or by a wrapper engine) plays one.
            return wins
        if out:
            return out
        # not even one iteration finished: fall back to the cheap heuristic (a placeholder, which
        # log_decision logs as a move without a search)
        ms = root.moves if root.moves else legal_moves(board, self.allow_quantum)
        if not ms:
            raise ValueError("no legal moves in this position")
        if root.moves is None:
            sc = heuristic_scores(board, ms)
            ms = [ms[int(np.argmax(sc))]]
        return [Candidate(ms[0], 0.0, 0.0, 0.0, 0, 0.0, {"iterations": iters, "fallback": True})]

    def _reused_root(self, board: QuantumBoard) -> Node | None:
        prev = self._prev
        if prev is None or self._prev_me != self._me or len(board.history) < prev.ply:
            return None
        node = prev
        for mr in board.history[prev.ply:]:
            e = node.edges.get(mr.move)
            if e is None:
                return None
            node = e.children.get(mr.outcome)
            if node is None:
                return None
        if node.turn != board.turn or not same_position(node.board, board):
            return None
        return node


__all__ = ["MCTSEngine", "Node", "Edge", "heuristic_scores", "same_position", "soft_value", "soft_opp_value",
           "check_dirichlet", "check_prior_info", "check_time_manager", "TM_DEFAULTS"]
