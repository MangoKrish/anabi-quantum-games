"""Static evaluators for quantum-chess positions: material, Fairy-Stockfish, tablebase.

``make_evaluator(cfg)`` builds an ``Evaluator`` from a plain dict.  Every evaluator

* ``ev(board, me)`` -> float in [-1, 1] from the point of view of colour ``me`` ("w" / "b");
  exact (+1 / -1 / 0) when the game is over under Cantwell Rule 10;
* ``ev.branch_values(board, me)`` -> ``[(weight, value)]`` over the classical branches of the
  superposition (Born weights; a finished game gives ``[(1.0, terminal)]``); a branch with a
  missing king is terminal (+1 / -1 / 0) and is never sent to an engine;
* ``ev.aggregate(branch_values)`` -> the value ``__call__`` returns;
* ``ev.stats()`` / ``ev.close()``;
* ``ev.reset()``: per-game reset (engine state, caches, counters; search engines call it from
  ``new_game`` so that a game's evaluations do not depend on the games played before it);
* ``ev.set_game_info(info)``: arena hook before a game (``ply_cap``, ``start_ply``, ``n``);
* ``ev.set_budget(budget | None)``: while a ``base.Budget`` is set, an evaluator that calls an
  engine (``fsf``) raises ``base.OutOfTime`` instead of starting a new engine search once the
  budget has expired, so the caller can drop the unfinished evaluation.

Value bands (from ``me``'s view), which ``Candidate.p_loss`` and the search engines rely on:

* decided   ``|v| >= LOSS_THRESHOLD`` (0.985): +1 / -1 / 0 for a branch with a king missing, or a
  tablebase-proven result (``[0.990, 1]``);
* forced    ``FORCED_MIN <= |v| <= FORCED_MAX`` ([0.93, 0.98]): in this classical branch the side
  to move can take the enemy king (``hang``) or Fairy-Stockfish reports a mate.  A result ``d``
  plies away is ``forced_value(d) = 0.98 - 0.0025 * min(d, 20)`` (faster is larger);
* heuristic ``|v| <= HEUR_MAX`` (0.85): ``HEUR_MAX * tanh(score / scale)`` (engine centipawns of
  the ``fsf`` kind: ``HEUR_MAX * tanh(cp / cp_scale)``).

So every forced result outranks every heuristic score, however lopsided the material, and a
heuristic score can never be mistaken for a decided one.  (With the ``settled`` option below this
holds on classical boards only; see there for how the bands shift.)

Kinds (``cfg["kind"]``):

``material`` (default)
    Born-weighted material (P1 N3 B3 R5 Q9, ``cfg["values"]`` overrides) per classical branch,
    plus a king-safety term computed exactly in each branch: if the side to move can capture the
    enemy king the branch is a forced result worth ``cfg["hang"]`` (default
    ``forced_value(1)`` = 0.9775; must lie in the forced band) to the mover; otherwise the mover
    loses ``cfg["check"]`` (default 0.5) pawns per enemy piece attacking its king.
    ``cfg["king_safety"]`` (default 1) scales the check term; ``hang`` is a forced result and is
    not scaled, and ``king_safety = 0`` switches both off (the branch is then scored by material
    alone).  Optional mobility ``cfg["mobility"]`` (pawns per pseudo-legal move,
    probability-weighted, default 0) and optional static threats ``cfg["threats"]`` (default 0):
    the mover gains that fraction of the best capture available in the branch (victim value if
    undefended, else victim minus least valuable attacker), a one-move quiescence for shallow
    searches.  A branch is squashed as ``HEUR_MAX * tanh(score / cfg["scale"])`` (default scale 4
    pawns, the same as the ``HEUR_MAX * tanh(cp / 400)`` of the ``fsf`` kind).  Pure Python on bit
    masks; no engine.

``kings``
    Only the kings: playable branches are 0, so the ``mean`` value is ``base.king_balance``.  An
    ablation baseline.

``fsf``
    Fairy-Stockfish in the king-capture variant (``gardnerkc`` on 5x5, ``gardner4kc`` on 4x4)
    on each playable branch.  If there are at most ``cfg["exact_max"]`` (default 16) playable
    branches they are all evaluated with their Born weights; otherwise ``cfg["samples"]``
    (default 8) are drawn by weight (Monte Carlo, the terminal branches stay exact).  The draw
    uses common random numbers: the generator is seeded from ``cfg["seed"]`` and the position,
    so a position always gets the same sample (``cfg["crn"] = False`` uses one running
    generator instead).  Search limit ``cfg["depth"]`` (default 6) or ``cfg["nodes"]``.
    ``cp`` -> ``HEUR_MAX * tanh(cp / cfg["cp_scale"])``; ``mate n`` (side to move wins in ``2n - 1``
    plies, or for ``n < 0`` loses in ``2|n|`` plies) -> ``±forced_value(plies)``.  ``cp_scale``
    (centipawns, > 0) defaults to ``100 * cfg["scale"]`` = 400, the original curve; with the key
    absent the computation is bitwise the one before the key existed.  It only changes engine
    ``cp`` branches: mates stay in the forced band and branches that fall back to ``material``
    keep ``scale``.  Board-level ``mobility`` / ``qfeat`` terms (pawns) are added to the engine's
    score before squashing, ``tanh((cp / 100 + off) / (cp_scale / 100))``.  Only this kind reads
    ``cp_scale`` (directly or as a tablebase's fallback): ``material`` and ``kings``, and a
    ``tablebase`` whose fallback is not ``fsf``, raise ``ValueError`` for it instead of ignoring
    it.  Fitted scale (for this form, ``0.5 + 0.5 * HEUR_MAX * tanh(cp / S)`` read as an expected
    score): ``reports/engines/statmech/fd_calibration.md``.  FENs are validated
    with pyffish; invalid ones (and everything, if the binary is missing) fall back to
    ``material``.  LRU cache keyed by FEN (``cfg["cache"]``, default 50000).

    Engine state and reproducibility: each evaluator owns private engine subprocesses (one per
    board size), so its values never depend on another evaluator's queries.  ``reset()`` sends
    ``ucinewgame`` to them and clears the FEN cache and the counters, so a game's values depend
    only on that game's own sequence of queries (deterministic under node budgets).  With
    ``cfg["fresh"] = True`` the engine also gets ``ucinewgame`` before every uncached query, so
    every value is a function of the FEN and the search limit alone (about twice the engine time
    at depth 6).

    Processes: the engine for ``cfg["n"]`` (default 5) starts at construction unless
    ``cfg["lazy"]``; ``set_game_info({"n": ...})`` starts the engine for the game's board size
    before the game (outside any move budget) and stops engines for other sizes.  At most
    ``max_processes()`` engine subprocesses (default 6, env ``QCHESS_FSF_MAX_PROCS``,
    ``set_max_processes(k)``) live in one Python process: starting another stops the least
    recently used one, which restarts (with a fresh state, the same as after ``reset()``) when
    its evaluator next needs it.  Idle pooled evaluators are the ones stopped; only if more
    evaluators than the cap are used within one game do restarts happen mid-game (raise the cap
    then).  Engine handles inherited through ``fork`` are never used or closed by the child.
    ``close()`` stops them.

``tablebase``
    For the endgame suite.  A ``qchess.experiments.tablebase.Tablebase`` is attached with
    ``ev.set_tablebase(tb, plies_left_fn)`` or registered for all evaluators with
    ``register_tablebase(tb, plies_left_fn=None, horizon=None, name="default")`` (selected with
    ``cfg["tb"]``).  Each branch is valued like ``qchess.experiments.cantwell.utility`` with
    ``r = plies_left_fn(board)`` plies left (``None`` = unbounded; ``horizon`` gives
    ``horizon - len(board.history)``), divided by 1.01 so it stays in [-1, 1] (faster wins and
    slower losses still score higher).  Without either, the evaluator uses the arena's
    tablebase (``qchess.engines.baselines.get_tablebase()``) and the plies left from the arena's
    ``set_game_info`` hook (forwarded by the engines), else ``cfg["horizon"]``, else unbounded.
    Branches outside the solved families use ``cfg["fallback"]`` ("material" default, "fsf",
    or a full evaluator config; never a tablebase kind); top-level ``qfeat``, ``mobility``,
    ``scale`` and ``cp_scale`` are passed to the fallback unless it sets its own.  A top-level
    ``cp_scale`` is always checked (``check_cp_scale``) and needs an ``fsf`` fallback
    (``ValueError`` otherwise, since no other fallback would use it).

Quantum-aware terms, any heuristic kind: ``cfg["qfeat"] = {"king_spread": w1,
"opp_uncertainty": w2, "branch_count": w3}`` (weights in pawns, added to every heuristic
branch's score before squashing, like ``mobility``; forced branches (``hang``, engine mates)
and decided ones (terminal, tablebase) ignore both, so that they stay in their bands):

* ``king_spread``: ``spread(me) * threat(me) - spread(opp) * threat(opp)``, where ``spread`` is
  ``1 - max_sq P(king on sq) / P(king on board)`` (0 for a classical king, 1/2 for an even
  split) and ``threat`` is the Born mass of branches in which that king is attacked.
* ``opp_uncertainty``: ``u(opp) - u(me)`` with ``u(c) = sum 4 p (1 - p)`` over the non-king
  squares of colour ``c`` (pieces in superposition).
* ``branch_count``: ``(n_good - n_bad) / n_branches`` over *distinct* classical branches
  (unweighted), good/bad by the sign of the branch material (terminal branches by their
  result): rewards having many different branches that favour me.

All three are antisymmetric, so ``ev(b, "w") == -ev(b, "b")`` for ``agg = "mean"``.

Information-theoretic ``qfeat`` terms (added 20 October 2026; ``qchess.engines.infotheory``; weights
in pawns, any sign; absent or 0 = off and bitwise the old values; same exemptions as above):

* ``king_entropy``: ``H(me) - H(opp)``, the Shannon entropy in bits of each king's location over its
  squares plus "absent" (captured mass).  0 for classical kings, 1 bit for an even split.
* ``king_variance``: ``V(me) - V(opp)``, ``V = sum_q mu_q (1 - mu_q)`` over the king's squares.
* ``entanglement``: ``S * behind(me)``, ``S`` = von Neumann entropy (bits) of the reduced state of one
  colour's occupancy qubits (average of the white and the black cut; ancillas on the other side, so
  for the pure global state ``S(rho_A) = S(rho_B)``), computed from the sparse amplitudes;
  ``behind(me) = -tanh(a / 2)`` with ``a`` = my Born-mean material balance plus 10 pawns per unit of
  king-probability balance (+1 far behind, -1 far ahead): entanglement is scored as good for the
  side that is behind and bad for the side that is ahead.  ``S`` itself is colour-symmetric; the
  ``behind`` factor makes the term antisymmetric.
* ``impurity``: the same with ``1 - Tr rho_A^2`` (linear entropy) instead of ``S``.

Above ``cfg["qfeat_cap"]`` (default 256) non-zero amplitudes ``entanglement`` / ``impurity`` are 0
(counter ``qfeat_cap_skips``).  In the champion's 200 games vs B1tm the reached positions had at most 5
amplitudes (``reports/engines/infotheory.md``).

``settled`` (``material`` and ``fsf`` kinds only; added 20 October 2026; experimental, no fitted
values are recommended): ``cfg["settled"] = {"king_gone": k_t, "mate": k_m}``, both numbers in
[0, 1] and both required.  Key absent = off, and the values are then bitwise the ones before the
key existed (``{"king_gone": 1, "mate": 1}`` is bitwise the same as absent too).  Under Rule 10 the
game goes on until a king's *total* probability is 0, so in a superposed position one classical
branch is not a result yet; the option values a branch with the enemy king gone at k_t of a win
and a branch with a forced result at k_m of its forced value.

What the factors should be is open.  The first fit against B1tm (k_t = 0.21, k_m = 0.31 at
``cp_scale`` 1320, ``reports/engines/statmech/settled_calibration.md``) is withdrawn (review of 20
October 2026): it left out player strength.  B1tm never splits its king, so every king-missing
branch in those games is the challenger's own king and the sign of that mass is set by who is to
move; the challenger, the stronger player, scored 0.593 when it was to move and B1tm 0.413, and a
fit without a who-to-move term loads that gap onto k_t (and partly k_m).  With the term the same
data give k_t = 0.84, k_m = 0.56, and out of sample (who-to-move term and one global scale in the
model) those beat k = 1, i.e. no ``settled``, by only 0.0015 nats per position, while k = 1 beats
(0.21, 0.31) by 0.012.  The earlier readings "the hybrid converts its own forced branches at full
value, B1tm hardly any" and "a king taken by exactly half is worth about nothing" came mostly from
the same confound.  On a board with two or more classical branches that Rule 10 has not ended:

* a branch with exactly one king missing is valued ``k_t * (+-1)`` instead of +-1 (both kings
  missing: 0, as before);
* a branch in the forced band (``hang``, an engine mate, also in a ``material`` fallback branch of
  the ``fsf`` kind) keeps its sign and its order by distance and is scaled to ``k_m * v``;
* heuristic branches are unchanged (``cp_scale`` / ``scale`` still set their curve), and the
  aggregation (``agg``) then runs over the settled values;
* except where the side to move can end the game next ply with certainty, by a move after which
  every outcome is a game it has won (a classical rook takes a classical king along a classically
  empty file whatever else is in superposition, or takes the last part of a partly taken king).
  Such a board keeps its default branch values, as a classical hang does, so superposing an
  unrelated piece no longer turns a lost position into a third of one (review of 20 October 2026;
  before, ``n3k/5/5/5/K3R b`` after ``a5^b3c4`` was -0.30 for black instead of -0.9775, and greedy
  split the knight while its king hung).  Exact test: ``_mover_wins_next`` plays the mover's moves
  onto the enemy king's square out with ``base.chance_children``, once every branch holding the
  enemy king has it on one square and attacked; counter ``settled_certain``.  In the calibration
  games the side to move with such a capture (403 superposed positions, read from their branch
  FENs) scored 1.000 for both players; (0.21, 0.31) predicted 0.63.  A forced result that is not
  certain one ply ahead (the king split over two squares, a path blocked in some branch, different
  attackers in different branches, an engine mate in more than one move) is still scaled.

Unchanged: a finished board (Rule 10) is exactly +1 / -1 / 0 (``branch_values`` returns
``[(1.0, terminal)]`` before any branch is scored), and so is every classical board (one branch):
there a ``hang`` is a king capture next ply that ends the game, and an engine mate is a classical
mate (in the calibration games the side with a classical engine mate scored 0.96, the mated side
0.08).  ``kings`` and ``tablebase`` raise ``ValueError`` for the key (also inside a tablebase's
fallback, whose branches the tablebase scores itself).

How the bands shift under ``settled`` (k_t, k_m < 1): decided ``|v| >= LOSS_THRESHOLD`` = finished
boards, plus the king-missing branches of a board kept at its default values because the mover
wins next ply with certainty; forced ``[FORCED_MIN, FORCED_MAX]`` = classical boards and those
certain boards; every other superposed board's branch values lie within ``max(HEUR_MAX, k_t,
k_m * FORCED_MAX)``, so its value can be smaller in size than a heuristic score (a mate branch at
k_m = 0.4 is worth 0.39, which a heuristic branch reaches at 2 pawns with ``scale`` 4 /
``cp_scale`` 400, 8.7 pawns at ``cp_scale`` 1740).
Consequences: ``GreedyEngine``'s ``p_loss`` counts the mass with my king gone structurally (``Evaluator.lost_mass``), so it is the same with and without the key; the
``p_loss`` of expectimax and MCTS reads backed-up root outcome values (<= -0.95 / <= -0.9), so it
then counts only outcomes the search finds lost under Rule 10 (finished boards within the
horizon) or forced on a classical board, not superposed leaves with my king partly gone; and an
adaptive ``lose_below = -0.9`` / ``win_above = 0.9`` (gamble only where the evaluator calls the
position forced or decided) no longer fires on superposed king-gone / mate mass.  No search engine
detects a finished game by its value: expectimax, MCTS (proven wins), the rollouts, greedy and
the trap analysis call ``base.terminal_value`` (``board.result()``), and the Star1 bounds stay
[-1, 1], so they are unaffected (``tests/engines/test_settled.py``).

Aggregation over branches, ``cfg["agg"]``: ``mean`` (default, Born expectation), ``min`` (worst
branch), ``cvar`` (mean of the worst ``cfg["alpha"]`` mass, default 0.25), ``meanstd``
(``mean - cfg["lam"] * std``, default lam 0.5).
"""
from __future__ import annotations

import math
import os
import shutil
import weakref
import zlib
from collections import Counter, OrderedDict

import numpy as np

from ..geometry import BLACK, WHITE, Geometry, to_fen
from . import infotheory as _it
from .base import LOSS, WIN, OutOfTime, chance_children, legal_moves, terminal_value

# qfeat terms: the original three and the information-theoretic ones (20 October 2026)
INFO_TERMS = frozenset({"king_entropy", "king_variance", "entanglement", "impurity"})
QFEAT_TERMS = frozenset({"king_spread", "opp_uncertainty", "branch_count"}) | INFO_TERMS

PIECE_VALUES = {"P": 1.0, "N": 3.0, "B": 3.0, "R": 5.0, "Q": 9.0, "K": 0.0}
W_EPS = 1e-14          # branches lighter than this are ignored

# Value bands (see the module docstring): heuristic < forced < decided.
HEUR_MAX = 0.85         # |heuristic value| <= HEUR_MAX  (HEUR_MAX * tanh(score / scale))
FORCED_MAX = 0.98       # forced results (king capturable in the branch, engine mate) lie in
FORCED_STEP = 0.0025    #   [FORCED_MIN, FORCED_MAX], FORCED_STEP lower per ply, for up to 20 plies
FORCED_MIN = FORCED_MAX - FORCED_STEP * 20      # 0.93
LOSS_THRESHOLD = 0.985  # |value| at or above this counts as a decided branch (terminal / proven)


def forced_value(plies: int) -> float:
    """Magnitude of a forced result ``plies`` plies away: 0.98 - 0.0025 * min(plies, 20)."""
    return FORCED_MAX - FORCED_STEP * min(max(int(plies), 0), 20)


HANG = forced_value(1)  # the side to move takes the enemy king next ply


def check_cp_scale(x) -> float:
    """The fsf kind's ``cp_scale`` as a float: a finite number of centipawns > 0 (not a bool)."""
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise ValueError(f"cp_scale must be a positive number of centipawns, not {x!r}")
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"cp_scale must be a positive finite number of centipawns, not {x!r}")
    return v


SETTLED_KEYS = ("king_gone", "mate")


def check_settled(x) -> tuple[float, float]:
    """The heuristic kinds' ``settled`` option as ``(k_t, k_m)``: a dict with exactly the keys
    ``"king_gone"`` and ``"mate"``, each a finite number in [0, 1] (not a bool)."""
    if not isinstance(x, dict):
        raise ValueError(f"settled must be a dict {{'king_gone': k_t, 'mate': k_m}}, not {x!r}")
    missing = [k for k in SETTLED_KEYS if k not in x]
    unknown = sorted(set(map(str, x)) - set(SETTLED_KEYS))
    if missing or unknown:
        raise ValueError(f"settled needs exactly the keys {list(SETTLED_KEYS)} (missing {missing}, "
                         f"unknown {unknown})")
    out = []
    for k in SETTLED_KEYS:
        v = x[k]
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"settled[{k!r}] must be a number in [0, 1], not {v!r}")
        f = float(v)
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise ValueError(f"settled[{k!r}] must be a number in [0, 1], not {v!r}")
        out.append(f)
    return out[0], out[1]


def _reject_settled(cfg: dict, kind: str) -> None:
    """``ValueError`` if a kind without heuristic branches of its own is given ``settled``."""
    if "settled" in cfg:
        raise ValueError(f"settled ({cfg['settled']!r}) only applies to the heuristic kinds material and "
                         f"fsf; the {kind} kind would ignore it")


def _reject_cp_scale(cfg: dict, kind: str) -> None:
    """``ValueError`` if a kind that never sees engine centipawns is given ``cp_scale``: it would be
    accepted and silently have no effect (review of 20 October 2026)."""
    if "cp_scale" in cfg:
        raise ValueError(f"cp_scale ({cfg['cp_scale']!r}) only applies to the fsf kind (engine centipawns, "
                         f"directly or as a tablebase's fallback); the {kind} kind would ignore it (its "
                         f"curve is set by 'scale', in pawns)")

KC4_INI = """[gardner4kc:gardner4]
king = -
commoner = k
extinctionValue = loss
extinctionPieceTypes = k
promotionPieceTypes = q
"""


# ---- per-board-size attack tables -----------------------------------------------------------------
class _Tables:
    """Bit masks and rays used to find attacks on a square in a classical branch."""

    _cache: dict[int, "_Tables"] = {}

    def __init__(self, n: int):
        g = Geometry(n)
        self.n = n
        R = n * n
        self.knight = [0] * R
        self.king = [0] * R
        self.pawn_from = {WHITE: [0] * R, BLACK: [0] * R}
        self.ortho: list[list[list[int]]] = [[] for _ in range(R)]
        self.diag: list[list[list[int]]] = [[] for _ in range(R)]
        for t in range(R):
            for s in g.jump_targets(t, "N"):
                self.knight[t] |= 1 << s
            for s in g.jump_targets(t, "K"):
                self.king[t] |= 1 << s
            x, y = g.xy(t)
            for dx in (-1, 1):
                if g.in_board(x + dx, y - 1):          # a white pawn one rank below attacks t
                    self.pawn_from[WHITE][t] |= 1 << g.idx(x + dx, y - 1)
                if g.in_board(x + dx, y + 1):
                    self.pawn_from[BLACK][t] |= 1 << g.idx(x + dx, y + 1)
            for dirs, out in (((1, 0), (-1, 0), (0, 1), (0, -1)), self.ortho[t]), \
                             (((1, 1), (1, -1), (-1, 1), (-1, -1)), self.diag[t]):
                for dx, dy in dirs:
                    ray = []
                    cx, cy = x + dx, y + dy
                    while g.in_board(cx, cy):
                        ray.append(g.idx(cx, cy))
                        cx, cy = cx + dx, cy + dy
                    if ray:
                        out.append(ray)

    @classmethod
    def get(cls, n: int) -> "_Tables":
        t = cls._cache.get(n)
        if t is None:
            t = cls._cache[n] = _Tables(n)
        return t


class _Scan:
    """One pass over a board: its classical branches and per-letter square masks."""

    def __init__(self, board):
        self.board = board
        self.n = board.n
        self.T = _Tables.get(board.n)
        mask = (1 << board.nsq) - 1
        acc: dict[int, float] = {}
        for b, a in board.state.amps.items():
            bits = b & mask
            acc[bits] = acc.get(bits, 0.0) + float(a.real * a.real + a.imag * a.imag)
        self.branches = [(w, bits) for bits, w in acc.items() if w > W_EPS]
        self.masks: dict[str, int] = {}
        for q, p in enumerate(board.types):
            if p is not None:
                self.masks[p] = self.masks.get(p, 0) | (1 << q)
        self._probs: list[float] | None = None

    def m(self, letter: str) -> int:
        return self.masks.get(letter, 0)

    @property
    def probs(self) -> list[float]:
        if self._probs is None:
            pr = [0.0] * (self.n * self.n)
            for w, bits in self.branches:
                b, q = bits, 0
                while b:
                    if b & 1:
                        pr[q] += w
                    b >>= 1
                    q += 1
            self._probs = pr
        return self._probs

    def king_sq(self, bits: int, color: str) -> int:
        """Square of ``color``'s king in a branch (-1 if absent)."""
        kb = bits & self.m("K" if color == WHITE else "k")
        return kb.bit_length() - 1 if kb else -1

    def attackers(self, bits: int, t: int, color: str) -> int:
        """Number of ``color`` pieces attacking square ``t`` in the classical branch ``bits``."""
        T = self.T
        if color == WHITE:
            mN, mK, mP, mB, mR, mQ = (self.m(c) for c in "NKPBRQ")
        else:
            mN, mK, mP, mB, mR, mQ = (self.m(c) for c in "nkpbrq")
        cnt = ((bits & T.knight[t] & mN).bit_count() + (bits & T.king[t] & mK).bit_count()
               + (bits & T.pawn_from[color][t] & mP).bit_count())
        rq, bq = (mR | mQ) & bits, (mB | mQ) & bits
        if rq:
            for ray in T.ortho[t]:
                for s in ray:
                    if (bits >> s) & 1:
                        if (rq >> s) & 1:
                            cnt += 1
                        break
        if bq:
            for ray in T.diag[t]:
                for s in ray:
                    if (bits >> s) & 1:
                        if (bq >> s) & 1:
                            cnt += 1
                        break
        return cnt

    def min_attacker(self, bits: int, t: int, color: str, values: dict) -> float | None:
        """Value of ``color``'s least valuable piece attacking ``t`` (the king counts as 50,
        since it may only take undefended pieces), or None."""
        T = self.T
        up = color == WHITE
        L = (lambda c: c) if up else str.lower
        if bits & T.pawn_from[color][t] & self.m(L("P")):
            return values["P"]
        best = math.inf
        if bits & T.knight[t] & self.m(L("N")):
            best = values["N"]
        for rays, slider in ((T.diag[t], "B"), (T.ortho[t], "R")):
            ms, mq = self.m(L(slider)) & bits, self.m(L("Q")) & bits
            if not (ms | mq):
                continue
            for ray in rays:
                for s in ray:
                    if (bits >> s) & 1:
                        if (ms >> s) & 1:
                            best = min(best, values[slider])
                        elif (mq >> s) & 1:
                            best = min(best, values["Q"])
                        break
        if best == math.inf and bits & T.king[t] & self.m(L("K")):
            best = 50.0
        return None if best == math.inf else best


def _sign(me: str) -> float:
    return 1.0 if me == WHITE else -1.0


def _mover_wins_next(board, scan: _Scan) -> bool:
    """True if the side to move has a move after which every outcome is a game it has won under Rule 10 (the
    enemy king's total probability is 0): the result is certain one ply ahead, whatever is measured.  Used by
    the ``settled`` option only (review of 20 October 2026).

    A necessary condition is checked from the branches first: in every branch where the enemy king is present
    it stands on one and the same square and a piece of the mover attacks it there.  Only then are the mover's
    classical moves onto that square played out exactly (``base.chance_children``)."""
    mover = board.turn
    kmask = scan.m("k" if mover == WHITE else "K")
    target = -1
    for _, bits in scan.branches:
        kb = bits & kmask
        if not kb:
            continue
        t = kb.bit_length() - 1
        if target < 0:
            target = t
        elif t != target:
            return False
        if not scan.attackers(bits, t, mover):
            return False
    if target < 0:
        return False
    win = "white" if mover == WHITE else "black"
    for m in legal_moves(board):
        if m.target != target or m.is_split or m.is_merge:
            continue
        if all(child.result() == win for _, child in chance_children(board, m)):
            return True
    return False


# ---- the evaluator interface ----------------------------------------------------------------------
class Evaluator:
    kind = "base"

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(cfg or {})
        self.agg = self.cfg.get("agg", "mean")
        if self.agg not in ("mean", "min", "cvar", "meanstd"):
            raise ValueError(f"unknown aggregation {self.agg!r}")
        self.alpha = float(self.cfg.get("alpha", 0.25))
        self.lam = float(self.cfg.get("lam", 0.5))
        self.scale = float(self.cfg.get("scale", 4.0))
        self.mobility = float(self.cfg.get("mobility", 0.0))
        self.qfeat = {k: float(v) for k, v in (self.cfg.get("qfeat") or {}).items() if float(v) != 0.0}
        unknown = set(self.qfeat) - QFEAT_TERMS
        if unknown:
            raise ValueError(f"unknown qfeat terms {sorted(unknown)}")
        for k, v in self.qfeat.items():
            if k in INFO_TERMS and not math.isfinite(v):
                raise ValueError(f"qfeat[{k!r}] must be a finite number, not {v!r}")
        # entanglement / impurity: amplitude cap above which the cut spectrum is not computed (term 0)
        self.qfeat_cap = int(self.cfg.get("qfeat_cap", _it.DEFAULT_CAP))
        vals = dict(PIECE_VALUES)
        vals.update({k.upper(): float(v) for k, v in (self.cfg.get("values") or {}).items()})
        self.values = vals
        self.hang = float(self.cfg.get("hang", HANG))
        if not FORCED_MIN <= self.hang <= FORCED_MAX:
            raise ValueError(f"hang must lie in the forced band [{FORCED_MIN}, {FORCED_MAX}], "
                             f"not {self.hang}")
        # (k_t, k_m) or None (key absent: off); the kinds without heuristic branches reject the key
        self.settled = check_settled(self.cfg["settled"]) if "settled" in self.cfg else None
        self.counters: Counter = Counter()
        self._budget = None

    # -- public API --
    def __call__(self, board, me: str) -> float:
        return self.aggregate(self.branch_values(board, me))

    value = __call__

    def branch_values(self, board, me: str) -> list[tuple[float, float]]:
        self.counters["calls"] += 1
        tv = terminal_value(board, me)
        if tv is not None:
            self.counters["terminal"] += 1
            return [(1.0, tv)]
        scan = _Scan(board)
        return self._from_scan(board, scan, me)

    def aggregate(self, wv: list[tuple[float, float]]) -> float:
        return aggregate(wv, self.agg, self.alpha, self.lam)

    def stats(self) -> dict:
        return dict(self.counters)

    def set_game_info(self, info: dict | None) -> None:
        """Arena hook before a game (ply cap, start ply, board size ``n``)."""
        self.game_info = dict(info or {})

    def reset(self) -> None:
        """Per-game reset: counters (and, per kind, engine state and caches)."""
        self.counters.clear()

    def new_game(self) -> None:
        """Alias of ``reset``."""
        self.reset()

    def set_budget(self, budget) -> None:
        """``base.Budget`` the evaluator may stop on (``OutOfTime``); ``None`` = never stop."""
        self._budget = budget

    def close(self) -> None:
        pass

    def lost_mass(self, board, me: str) -> float:
        """Share of ``board``'s Born mass lost for ``me`` whatever the values: 1.0 / 0.0 for a finished
        game (lost / not lost), else the classical branches with my king missing and the enemy's
        present.  For the ``material`` and ``fsf`` kinds without ``settled`` this is the mass of the
        branches valued ``<= -LOSS_THRESHOLD``; ``GreedyEngine`` reads it for ``p_loss`` under
        ``settled``, where those branches are valued ``-k_t``."""
        tv = terminal_value(board, me)
        if tv is not None:
            return 1.0 if tv < 0 else 0.0
        scan = _Scan(board)
        mine, theirs = (scan.m("K"), scan.m("k")) if me == WHITE else (scan.m("k"), scan.m("K"))
        tot = lost = 0.0
        for w, bits in scan.branches:
            tot += w
            if not bits & mine and bits & theirs:
                lost += w
        return lost / tot if tot > 0 else 0.0

    # -- internals --
    def _from_scan(self, board, scan: _Scan, me: str) -> list[tuple[float, float]]:
        mK, mk = scan.m("K"), scan.m("k")
        win_w = WIN if me == WHITE else LOSS
        # settled: only on a superposed board (a finished one never gets here, see branch_values) whose result
        # the side to move cannot make certain next ply (that one keeps its default values, like a classical hang)
        settle = self.settled is not None and len(scan.branches) > 1
        if settle:
            self.counters["settled_positions"] += 1
            if _mover_wins_next(board, scan):
                self.counters["settled_certain"] += 1
                settle = False
        if settle:
            win_w = self.settled[0] * win_w
        out: list[tuple[float, float]] = []
        playable: list[tuple[float, int]] = []
        for w, bits in scan.branches:
            hk, hb = bits & mK, bits & mk
            if hk and hb:
                playable.append((w, bits))
            elif not hk and not hb:
                out.append((w, 0.0))
            elif hk:
                out.append((w, win_w))          # black's king is gone in this branch
            else:
                out.append((w, -win_w))
        self.counters["branches"] += len(scan.branches)
        if playable:
            pv = self._playable_values(board, scan, playable, me)
            if settle:
                # for the material and fsf kinds a playable value above HEUR_MAX is a forced one
                # (hang, engine mate; heuristic values are HEUR_MAX * tanh, at most HEUR_MAX)
                k_m = self.settled[1]
                pv = [(w, k_m * v) if abs(v) > HEUR_MAX else (w, v) for w, v in pv]
            out.extend(pv)
        return out

    def _playable_values(self, board, scan: _Scan, playable, me: str) -> list[tuple[float, float]]:
        raise NotImplementedError

    def _material_white(self, scan: _Scan, bits: int) -> float:
        tot = 0.0
        for letter, mask in scan.masks.items():
            v = self.values.get(letter.upper(), 0.0)
            if v and bits & mask:
                c = (bits & mask).bit_count()
                tot += v * c if letter.isupper() else -v * c
        return tot

    def _offset(self, board, scan: _Scan, me: str) -> float:
        """Board-level terms in pawns from ``me``'s view (mobility and qfeat)."""
        if not self.qfeat and not self.mobility:
            return 0.0
        opp = BLACK if me == WHITE else WHITE
        off = 0.0
        if self.mobility:
            off += self.mobility * (_mobility(board, scan, me) - _mobility(board, scan, opp))
        w = self.qfeat.get("king_spread")
        if w:
            off += w * (_spread(scan, me) * _threat(scan, me) - _spread(scan, opp) * _threat(scan, opp))
        w = self.qfeat.get("opp_uncertainty")
        if w:
            off += w * (_uncertainty(scan, opp) - _uncertainty(scan, me))
        w = self.qfeat.get("branch_count")
        if w:
            good = bad = 0
            mK, mk = scan.m("K"), scan.m("k")
            for _, bits in scan.branches:
                hk, hb = bool(bits & mK), bool(bits & mk)
                if hk and hb:
                    s = self._material_white(scan, bits) * _sign(me)
                elif hk or hb:
                    s = _sign(me) * (1.0 if hk else -1.0)
                else:
                    s = 0.0
                good += s > 0
                bad += s < 0
            off += w * (good - bad) / max(1, len(scan.branches))
        if self.qfeat.keys() & INFO_TERMS:
            off += self._info_offset(board, scan, me, opp)
        return off

    def _info_offset(self, board, scan: _Scan, me: str, opp: str) -> float:
        """The information-theoretic qfeat terms (pawns, from ``me``'s view; module docstring)."""
        off = 0.0
        pr = scan.probs
        w = self.qfeat.get("king_entropy")
        if w:
            off += w * (_it.king_entropy(board, me, pr) - _it.king_entropy(board, opp, pr))
        w = self.qfeat.get("king_variance")
        if w:
            off += w * (_it.king_variance(board, me, pr) - _it.king_variance(board, opp, pr))
        we, wi = self.qfeat.get("entanglement"), self.qfeat.get("impurity")
        if (we or wi) and len(board.state.amps) > 1:
            sp = _it.colour_cut_spectra(board, self.qfeat_cap)
            if sp is None:
                self.counters["qfeat_cap_skips"] += 1
            else:
                amount = 0.0
                if we:
                    amount += we * 0.5 * (_it.vn_entropy(sp[0]) + _it.vn_entropy(sp[1]))
                if wi:
                    amount += wi * (1.0 - 0.5 * (_it.purity(sp[0]) + _it.purity(sp[1])))
                if amount:
                    off += amount * self._behind(scan, me)
        return off

    def _behind(self, scan: _Scan, me: str) -> float:
        """``-tanh(a / 2)`` with ``a`` = my advantage in pawns: the Born-mean material balance over
        all branches plus 10 pawns per unit of king-probability balance.  +1 far behind, -1 far
        ahead, 0 level: the sign the entanglement / impurity terms carry (complexity favours the
        side that is behind)."""
        mK, mk = scan.m("K"), scan.m("k")
        tot = mat = kw = kb = 0.0
        for w, bits in scan.branches:
            tot += w
            mat += w * self._material_white(scan, bits)
            kw += w * bool(bits & mK)
            kb += w * bool(bits & mk)
        if tot <= 0:
            return 0.0
        a = (mat + 10.0 * (kw - kb)) / tot * _sign(me)
        return -math.tanh(a / 2.0)

    def _heuristic_values(self, board, scan: _Scan, playable, me: str, off: float):
        """Material + king safety per playable branch (also the fallback of the other kinds).

        A branch where the mover can take the enemy king is a forced result (``±self.hang``,
        not scaled by ``king_safety`` and without the ``off`` terms); every other branch is
        ``HEUR_MAX * tanh(score / scale)``, strictly below the forced band."""
        ks = float(self.cfg.get("king_safety", 1.0))
        hang = self.hang
        check = float(self.cfg.get("check", 0.5))
        threats = float(self.cfg.get("threats", 0.0))
        sg = _sign(me)
        mover = board.turn
        waiter = BLACK if mover == WHITE else WHITE
        msign = 1.0 if mover == me else -1.0
        victims = [(q, self.values.get(p.upper(), 0.0)) for q, p in enumerate(board.types)
                   if p is not None and p.upper() != "K" and (p.isupper() == (waiter == WHITE))] \
            if threats else []
        out = []
        for w, bits in playable:
            if ks:
                ek = scan.king_sq(bits, waiter)
                if scan.attackers(bits, ek, mover):
                    out.append((w, msign * hang))
                    continue
            raw = self._material_white(scan, bits) * sg + off
            if ks and check:
                n_att = scan.attackers(bits, scan.king_sq(bits, mover), waiter)
                raw -= msign * ks * check * n_att
            if threats:
                gain = 0.0
                for q, v in victims:
                    if v <= gain or not (bits >> q) & 1:
                        continue
                    a = scan.min_attacker(bits, q, mover, self.values)
                    if a is None:
                        continue
                    g = v if not scan.attackers(bits, q, waiter) else v - a
                    gain = max(gain, g)
                raw += msign * threats * gain
            out.append((w, HEUR_MAX * math.tanh(raw / self.scale)))
        return out


def aggregate(wv: list[tuple[float, float]], agg: str = "mean", alpha: float = 0.25,
              lam: float = 0.5) -> float:
    """Combine ``[(weight, value)]`` into one value in [-1, 1]."""
    if not wv:
        return 0.0
    tot = sum(w for w, _ in wv)
    if tot <= 0:
        return 0.0
    if agg == "min":
        live = [v for w, v in wv if w > W_EPS]
        return float(min(live or [v for _, v in wv]))
    if agg == "cvar":
        target = max(min(alpha, 1.0), 1e-9) * tot
        acc = s = 0.0
        for w, v in sorted(wv, key=lambda x: x[1]):
            take = min(w, target - acc)
            if take <= 0:
                break
            s += take * v
            acc += take
        return float(max(-1.0, min(1.0, s / acc))) if acc > 0 else 0.0
    mean = sum(w * v for w, v in wv) / tot
    if agg == "meanstd":
        var = sum(w * (v - mean) ** 2 for w, v in wv) / tot
        mean -= lam * math.sqrt(max(var, 0.0))
    return float(max(-1.0, min(1.0, mean)))


# ---- quantum-aware and mobility features -------------------------------------------------------------
def _spread(scan: _Scan, color: str) -> float:
    letter = "K" if color == WHITE else "k"
    pr = scan.probs
    ps = [pr[q] for q, p in enumerate(scan.board.types) if p == letter]
    tot = sum(ps)
    if tot <= W_EPS or len(ps) < 2:
        return 0.0
    return 1.0 - max(ps) / tot


def _threat(scan: _Scan, color: str) -> float:
    """Born mass of branches in which ``color``'s king is on the board and attacked."""
    opp = BLACK if color == WHITE else WHITE
    mass = 0.0
    for w, bits in scan.branches:
        ks = scan.king_sq(bits, color)
        if ks >= 0 and scan.attackers(bits, ks, opp):
            mass += w
    return mass


def _uncertainty(scan: _Scan, color: str) -> float:
    pr = scan.probs
    u = 0.0
    for q, p in enumerate(scan.board.types):
        if p is None or p in "Kk" or (p.isupper() != (color == WHITE)):
            continue
        u += 4.0 * pr[q] * (1.0 - pr[q])
    return u


def _mobility(board, scan: _Scan, color: str) -> float:
    """Probability-weighted count of pseudo-legal targets (a square only stops a slide or is
    excluded as a target when it is classically occupied)."""
    g = board.geo
    pr = scan.probs
    types = board.types
    full = 1.0 - 1e-9

    def own(q):
        p = types[q]
        return p is not None and p.isupper() == (color == WHITE)

    tot = 0.0
    for s, p in enumerate(types):
        if p is None or p.isupper() != (color == WHITE) or pr[s] <= W_EPS:
            continue
        k = p.upper()
        cnt = 0
        if k in "NK":
            for t in g.jump_targets(s, p):
                if not (own(t) and pr[t] >= full):
                    cnt += 1
        elif k in "BRQ":
            T = scan.T
            rays = (T.ortho[s] if k in "RQ" else []) + (T.diag[s] if k in "BQ" else [])
            for ray in rays:
                for t in ray:
                    if types[t] is not None and pr[t] >= full:
                        if not own(t):
                            cnt += 1
                        break
                    cnt += 1
        else:
            t = g.pawn_step(s, color)
            if t is not None and not (types[t] is not None and pr[t] >= full):
                cnt += 1
            for t in g.pawn_captures(s, color):
                if types[t] is not None and not own(t):
                    cnt += 1
        tot += pr[s] * cnt
    return tot


# ---- kinds ------------------------------------------------------------------------------------------
class MaterialEvaluator(Evaluator):
    kind = "material"

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        _reject_cp_scale(self.cfg, self.kind)

    def _playable_values(self, board, scan, playable, me):
        return self._heuristic_values(board, scan, playable, me, self._offset(board, scan, me))


class KingsEvaluator(Evaluator):
    """Terminal branches only (playable branches are 0): the mean is ``king_balance``."""
    kind = "kings"

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        _reject_cp_scale(self.cfg, self.kind)
        _reject_settled(self.cfg, self.kind)

    def _playable_values(self, board, scan, playable, me):
        off = self._offset(board, scan, me)
        v = HEUR_MAX * math.tanh(off / self.scale) if off else 0.0
        return [(w, v) for w, _ in playable]


def _kc_variant(n: int) -> str:
    return {5: "gardnerkc", 4: "gardner4kc"}[n]


def _ensure_pyffish_variant(n: int):
    import pyffish as sf
    if n == 5:
        if "gardnerkc" not in sf.variants():
            from ..experiments.fsf_check import KC_INI
            sf.load_variant_config(KC_INI)
    else:
        from ..baseline import ensure_variant
        ensure_variant(4)
        if "gardner4kc" not in sf.variants():
            sf.load_variant_config(KC4_INI)
    return sf


def _start_kc_engine(n: int, path: str, hash_mb: int):
    """A UCI driver playing the king-capture variant on an n x n board."""
    if n == 5:
        from ..experiments.fsf_check import KingCaptureEngine
        return KingCaptureEngine(path, hash_mb=hash_mb)
    import os
    import tempfile

    from ..baseline import UCIEngine
    eng = UCIEngine(path, n=4, hash_mb=hash_mb)
    fd, ini = tempfile.mkstemp(suffix=".ini", prefix="qchess_kc4_")
    with os.fdopen(fd, "w") as fh:
        fh.write(KC4_INI)
    try:
        eng._send(f"load {ini}")
        eng._send("setoption name UCI_Variant value gardner4kc")
        eng._send("isready")
        eng._read_until("readyok")
    finally:
        os.remove(ini)
    return eng


def _mate_value(n: int) -> float:
    """UCI ``mate n`` (side-to-move view) -> forced band: a win in ``2n - 1`` plies for ``n > 0``,
    a loss in ``2|n|`` plies for ``n < 0``; ``mate 0`` (no move left) is a loss now."""
    if n == 0:
        return -FORCED_MAX
    if n > 0:
        return forced_value(2 * n - 1)
    return -forced_value(-2 * n)


# ---- engine processes: at most _MAX_PROCS live per Python process, least recently used stopped first ----
_MAX_PROCS = max(1, int(os.environ.get("QCHESS_FSF_MAX_PROCS", "6") or 6))
_LIVE: "OrderedDict[tuple[int, int], weakref.ref]" = OrderedDict()   # (id(evaluator), n), LRU first
_FOREIGN: list = []    # engine handles inherited through fork: the parent's processes, never touched


def max_processes() -> int:
    return _MAX_PROCS


def set_max_processes(k: int) -> None:
    """Cap on live Fairy-Stockfish evaluator processes in this Python process (>= 1)."""
    global _MAX_PROCS
    _MAX_PROCS = max(1, int(k))
    _make_room(None, 0)


def live_processes() -> int:
    """Live evaluator engine processes started by this Python process."""
    _prune()
    return len(_LIVE)


def _prune() -> None:
    pid = os.getpid()
    for key in list(_LIVE):
        ev = _LIVE[key]()
        if ev is None or key[1] not in ev._engines or ev._pids.get(key[1]) != pid:
            del _LIVE[key]


def _make_room(keep, extra: int = 1) -> None:
    """Stop least recently used engines (never ``keep``'s) until ``extra`` more fit under the cap."""
    _prune()
    while len(_LIVE) + extra > _MAX_PROCS:
        victim = next((k for k in _LIVE if keep is None or k[0] != id(keep)), None)
        if victim is None:
            return
        ev = _LIVE[victim]()
        if ev is None:
            del _LIVE[victim]
            continue
        ev._release(victim[1])
        ev.counters["engine_evictions"] += 1


class FSFEvaluator(Evaluator):
    kind = "fsf"

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        c = self.cfg
        self.exact_max = int(c.get("exact_max", 16))
        self.samples = max(1, int(c.get("samples", 8)))
        self.depth = int(c.get("depth", 6))
        self.nodes = c.get("nodes")
        self.go = f"go nodes {int(self.nodes)}" if self.nodes else f"go depth {self.depth}"
        # engine cp -> HEUR_MAX * tanh(cp / cp_scale); without the key, cp / (100 * scale) exactly as before
        if "cp_scale" in c:
            self.cp_scale = check_cp_scale(c["cp_scale"])
            self._cp_pawns = self.cp_scale / 100.0
        else:
            self.cp_scale = 100.0 * self.scale
            self._cp_pawns = self.scale
        self.cache_size = int(c.get("cache", 50000))
        self.seed = int(c.get("seed", 0))
        self.crn = bool(c.get("crn", True))
        self.fresh = bool(c.get("fresh", False))
        self.rng = np.random.default_rng(self.seed)
        self.path = c.get("path", "fairy-stockfish")
        self.hash_mb = int(c.get("hash", 16))
        self.board_n = int(c.get("n", 5))
        self._cache: OrderedDict[str, tuple[str, int]] = OrderedDict()
        self._fallback = MaterialEvaluator({k: v for k, v in c.items()
                                            if k in ("scale", "values", "king_safety", "hang", "check",
                                                     "threats")})
        self._engines: dict[int, object] = {}
        self._pids: dict[int, int] = {}      # board size -> pid that started the engine
        self._used: set[int] = set()         # board sizes queried since the engine started
        self._sf = None
        self._dead: set[int] = set()
        if c.get("lazy") is not True and shutil.which(self.path):
            self._engine(self.board_n)

    # -- engine management --
    def _engine(self, n: int):
        if n in self._dead:
            return None
        eng = self._engines.get(n)
        if eng is not None and self._pids.get(n) != os.getpid():
            # inherited through fork: that pipe belongs to the parent's process; leave it alone
            _FOREIGN.append(self._engines.pop(n))
            self._pids.pop(n, None)
            eng = None
        if eng is None:
            for m in [m for m in self._engines if m != n and m not in self._used]:
                self._release(m)                 # started eagerly for another board size, never used
            _make_room(self)
            try:
                self._sf = _ensure_pyffish_variant(n)
                eng = self._engines[n] = _start_kc_engine(n, self.path, self.hash_mb)
            except (FileNotFoundError, OSError, RuntimeError, ImportError):
                self._dead.add(n)
                self.counters["engine_unavailable"] += 1
                return None
            self._pids[n] = os.getpid()
            self._used.discard(n)
            _LIVE[(id(self), n)] = weakref.ref(self)
            self.counters["engine_starts"] += 1
        elif (id(self), n) in _LIVE:
            _LIVE.move_to_end((id(self), n))     # most recently used
        return eng

    def _release(self, n: int) -> None:
        """Stop the engine for board size ``n`` (restarted on demand)."""
        eng = self._engines.pop(n, None)
        pid = self._pids.pop(n, None)
        self._used.discard(n)
        _LIVE.pop((id(self), n), None)
        if eng is None:
            return
        if pid != os.getpid():
            _FOREIGN.append(eng)
            return
        try:
            eng.close()
        except Exception:
            pass

    def _new_game(self, n: int, eng) -> bool:
        try:
            eng._send("ucinewgame")
            eng._send("isready")
            eng._read_until("readyok")
            return True
        except (RuntimeError, OSError, BrokenPipeError, ValueError):
            self._release(n)
            return False

    def _query(self, n: int, fen: str) -> tuple[str, int] | None:
        hit = self._cache.get(fen)
        if hit is not None:
            self._cache.move_to_end(fen)
            self.counters["cache_hits"] += 1
            return hit
        b = self._budget
        if b is not None and b.expired():
            self.counters["budget_stops"] += 1
            raise OutOfTime
        for attempt in (0, 1):
            eng = self._engine(n)
            if eng is None:
                return None
            try:
                if self.fresh:
                    eng._send("ucinewgame")
                    eng._send("isready")
                    eng._read_until("readyok")
                eng._send(f"position fen {fen}")
                eng._send(self.go)
                lines = eng._read_until("bestmove")
                self._used.add(n)
                break
            except (RuntimeError, OSError, BrokenPipeError, ValueError):
                self.counters["engine_restarts"] += 1
                self._release(n)
                if attempt == 1:
                    self._dead.add(n)
                    return None
        res = ("cp", 0)
        for line in lines:
            if line.startswith("info") and " score " in line:
                toks = line.split()
                k = toks.index("score")
                try:
                    res = (toks[k + 1], int(toks[k + 2]))
                except (IndexError, ValueError):
                    pass
        self.counters["engine_calls"] += 1
        self._cache[fen] = res
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return res

    def _valid(self, n: int, fen: str) -> bool:
        if self._sf is None:
            return False
        try:
            return self._sf.validate_fen(fen, _kc_variant(n)) == 1
        except Exception:
            return False

    # -- branch selection --
    def _select(self, board, playable):
        if len(playable) <= self.exact_max:
            return playable
        self.counters["sampled_positions"] += 1
        mass = sum(w for w, _ in playable)
        p = np.array([w for w, _ in playable]) / mass
        if self.crn:
            key = f"{self.seed}|{board.turn}|{board.types}|{sorted(b for _, b in playable)}"
            rng = np.random.default_rng(zlib.crc32(key.encode()))
        else:
            rng = self.rng
        idx = rng.choice(len(playable), size=self.samples, p=p)
        counts = Counter(int(i) for i in idx)
        return [(mass * c / self.samples, playable[i][1]) for i, c in sorted(counts.items())]

    def _playable_values(self, board, scan, playable, me):
        n = board.n
        off = self._offset(board, scan, me)
        stm = board.turn
        sgn = 1.0 if stm == me else -1.0
        out, fallback = [], []
        if n not in (4, 5) or self._engine(n) is None:
            fallback = list(playable)          # material is cheap: no need to sample
        else:
            for w, bits in self._select(board, playable):
                fen = to_fen(board.types, stm, n, occupied=bits)
                if fen not in self._cache and not self._valid(n, fen):
                    self.counters["invalid_fen"] += 1
                    fallback.append((w, bits))
                    continue
                r = self._query(n, fen)
                if r is None:
                    fallback.append((w, bits))
                    continue
                kind, x = r
                if kind == "mate":
                    out.append((w, sgn * _mate_value(x)))          # forced: no offset
                else:
                    out.append((w, HEUR_MAX * math.tanh((sgn * x / 100.0 + off) / self._cp_pawns)))
        if fallback:
            self.counters["fallback_branches"] += len(fallback)
            out.extend(self._fallback._heuristic_values(board, scan, fallback, me, off))
        return out

    # -- per game --
    def reset(self) -> None:
        """``ucinewgame`` to every live engine, empty FEN cache, counters and sampling RNG reset."""
        super().reset()
        self._fallback.reset()
        self._cache.clear()
        self.rng = np.random.default_rng(self.seed)
        pid = os.getpid()
        for n, eng in list(self._engines.items()):
            if self._pids.get(n) != pid:
                self._release(n)
            else:
                self._new_game(n, eng)

    def set_game_info(self, info: dict | None) -> None:
        """Start the engine for the game's board size now (not inside the first move's budget)
        and stop the engines for other sizes."""
        super().set_game_info(info)
        n = self.game_info.get("n")
        if n is None:
            return
        n = int(n)
        for m in [m for m in self._engines if m != n]:
            self._release(m)
        self.board_n = n
        if n in (4, 5) and shutil.which(self.path):
            self._engine(n)

    def stats(self) -> dict:
        d = super().stats()
        d["cache_size"] = len(self._cache)
        return d

    def close(self) -> None:
        for n in list(self._engines):
            self._release(n)


# ---- tablebase ------------------------------------------------------------------------------------
_TABLEBASES: dict[str, tuple[object, object]] = {}


def register_tablebase(tb, plies_left_fn=None, horizon: int | None = None, name: str = "default") -> None:
    """Make a solved ``Tablebase`` available to every ``tablebase`` evaluator (``cfg["tb"]``).

    ``plies_left_fn(board) -> int | None`` gives the plies left in the game; with ``horizon``
    instead it is ``horizon - len(board.history)``; with neither the horizon is unbounded."""
    if plies_left_fn is None and horizon is not None:
        plies_left_fn = lambda b, _h=int(horizon): _h - len(b.history)  # noqa: E731
    _TABLEBASES[name] = (tb, plies_left_fn)


def unregister_tablebase(name: str = "default") -> None:
    _TABLEBASES.pop(name, None)


class TablebaseEvaluator(Evaluator):
    kind = "tablebase"

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        fb = self.cfg.get("fallback", "material")
        fb = {"kind": fb} if isinstance(fb, str) else dict(fb)
        fb_cls = _KINDS.get(fb.get("kind", "material"))
        if fb_cls is not None and issubclass(fb_cls, TablebaseEvaluator):
            raise ValueError("the tablebase fallback cannot be a tablebase")
        # settled: the tablebase scores the king-missing branches itself and calls only the
        # fallback's playable-branch scoring, so the key would be ignored at either level
        _reject_settled(self.cfg, self.kind)
        if "settled" in fb:
            raise ValueError(f"tablebase: settled ({fb['settled']!r}) in the fallback would be ignored (the "
                             f"tablebase values the king-missing branches itself); it applies to the "
                             f"material and fsf kinds used directly")
        if "cp_scale" in self.cfg:
            # checked even when the fallback sets its own (then the top-level value is not used)
            check_cp_scale(self.cfg["cp_scale"])
            if fb_cls is not None and not issubclass(fb_cls, FSFEvaluator):
                raise ValueError(f"tablebase: cp_scale only reaches an fsf fallback; this fallback is "
                                 f"{fb.get('kind', 'material')!r}, which would ignore it")
        for k in ("qfeat", "mobility", "scale", "cp_scale"):
            if k in self.cfg and k not in fb:
                fb[k] = self.cfg[k]
        self.fallback = make_evaluator(fb)
        self.tb_name = self.cfg.get("tb", "default")
        self._tb = None
        self._plies_left = None
        self._local = False
        self._game_plf = None
        # resolved now, not inside the first move's budget (importing baselines loads pyffish)
        from ..experiments.cantwell import utility
        self._utility = utility
        try:
            from .baselines import get_tablebase
        except ImportError:
            get_tablebase = None
        self._arena_tablebase = get_tablebase

    def set_tablebase(self, tb, plies_left_fn=None) -> None:
        self._tb, self._plies_left, self._local = tb, plies_left_fn, True

    def set_game_info(self, info: dict | None) -> None:
        """Arena hook (``{"ply_cap", "start_ply"}``): plies left = cap - plies played so far."""
        super().set_game_info(info)
        info = self.game_info
        cap = info.get("ply_cap")
        if cap is not None:
            start = int(info.get("start_ply", 0))
            self._game_plf = lambda b, _c=int(cap), _s=start: _c - (len(b.history) - _s)
        else:
            self._game_plf = None
        self._prepare(self._current()[0])            # index the tablebase before the game
        self.fallback.set_game_info(info)

    def reset(self) -> None:
        super().reset()
        self.fallback.reset()

    def set_budget(self, budget) -> None:
        super().set_budget(budget)
        self.fallback.set_budget(budget)

    def _current(self):
        """(tablebase, plies_left_fn).  Tablebase: ``set_tablebase`` > ``register_tablebase``
        under ``cfg["tb"]`` > ``qchess.engines.baselines.get_tablebase()`` (the arena's).  Plies
        left: ``set_tablebase``'s / the registry's function > the arena game info >
        ``cfg["horizon"] - len(board.history)`` > unbounded."""
        if self._local:
            tb, plf = self._tb, self._plies_left
        else:
            tb, plf = _TABLEBASES.get(self.tb_name, (None, None))
            if tb is None and self._arena_tablebase is not None:
                tb = self._arena_tablebase()
        if plf is None:
            plf = self._game_plf
        if plf is None and self.cfg.get("horizon") is not None:
            h = int(self.cfg["horizon"])
            plf = lambda b: h - len(b.history)  # noqa: E731
        return tb, plf

    @staticmethod
    def _prepare(tb) -> None:
        if tb is not None and getattr(tb, "_fidx", None) is None:
            tb._fidx = tb.family_index()

    def _playable_values(self, board, scan, playable, me):
        utility = self._utility
        tb, plf = self._current()
        out, rest = [], []
        if tb is None or tb.n != board.n:
            rest = list(playable)
        else:
            self._prepare(tb)
            r = plf(board) if plf is not None else None
            if r is not None:
                r = max(0, int(r))
            side = 0 if board.turn == WHITE else 1
            color = 0 if me == WHITE else 1
            types = board.types
            for w, bits in playable:
                placed = [(types[q], q) for q in range(board.nsq) if (bits >> q) & 1]
                try:
                    node = tb.encode(placed, side)
                except (KeyError, IndexError):
                    rest.append((w, bits))
                    continue
                if not tb.valid[node]:
                    rest.append((w, bits))
                    continue
                out.append((w, utility(tb, node, r, color) / 1.01))
            self.counters["tb_hits"] += len(out)
        if rest:
            self.counters["fallback_branches"] += len(rest)
            out.extend(self.fallback._playable_values(board, scan, rest, me))
        return out

    def stats(self) -> dict:
        d = super().stats()
        d.update({f"fallback_{k}": v for k, v in self.fallback.stats().items()})
        return d

    def close(self) -> None:
        self.fallback.close()


_KINDS = {"material": MaterialEvaluator, "kings": KingsEvaluator, "fsf": FSFEvaluator,
          "stockfish": FSFEvaluator, "tablebase": TablebaseEvaluator, "tb": TablebaseEvaluator}


def make_evaluator(cfg: dict | str | None = None) -> Evaluator:
    """Build an evaluator from ``{"kind": ..., ...}`` (a bare string is the kind)."""
    if cfg is None:
        cfg = {"kind": "material"}
    elif isinstance(cfg, str):
        cfg = {"kind": cfg}
    kind = cfg.get("kind", "material")
    if kind not in _KINDS:
        raise ValueError(f"unknown evaluator kind {kind!r}")
    return _KINDS[kind](cfg)


__all__ = ["Evaluator", "MaterialEvaluator", "KingsEvaluator", "FSFEvaluator", "TablebaseEvaluator",
           "make_evaluator", "aggregate", "register_tablebase", "unregister_tablebase", "PIECE_VALUES",
           "LOSS_THRESHOLD", "HEUR_MAX", "FORCED_MIN", "FORCED_MAX", "HANG", "forced_value", "KC4_INI",
           "check_cp_scale", "check_settled", "SETTLED_KEYS", "max_processes", "set_max_processes",
           "live_processes"]
