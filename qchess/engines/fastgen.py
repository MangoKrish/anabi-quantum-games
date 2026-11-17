"""Fast move generation and move application for ``QuantumBoard`` with unchanged semantics.

``fast_legal_moves(board)`` returns the same list as ``board.legal_moves()`` (same moves, same
order); ``fast_apply(board, m, outcome)`` returns the same child as the original ``base.apply``
(``board.copy()`` + ``ForcedRNG`` + ``board.push``), with bitwise-identical amplitudes;
``fast_push`` is an in-place ``board.push``.  ``tests/engines/test_fastgen.py`` checks all of
them against the originals on thousands of positions from random 5x5 and 4x4 games.

Where the time went (profile of random 5x5 games with splits, 19 October 2026): about 80 % of
``legal_moves`` and ``push`` was Qiskit rebuilding a controlled gate (``gate.control``, ~0.25 ms)
and its ``Operator`` matrix (~2 ms) for every slide with a path control and every pawn capture;
most of the rest was the full-state copy + gate application + comparison in
``QuantumBoard._unitary_changes_state``, run for every candidate move.

What this module does instead:

1. **Cached gate matrices.**  Every matrix comes from ``qchess.gates.matrix`` applied to the gate
   ``QuantumBoard.unitary_ops`` builds, computed once, and is applied by a replica of
   ``SparseState.apply`` that caches the per-column non-zero lists and output masks.  Same
   matrices, same products, same summation order: the amplitudes are bitwise identical.

2. **A cheap "does the unitary change the state" test** (second half of Cantwell's legal-move
   definition).  The jump, slide, capture and pawn-capture unitaries send basis states to basis
   states with a phase, and a basis state they move (an *active* branch) never comes back to
   itself with phase 1: iSWAP cycles |01>,|10> with phase product -1, and the capture moves send
   it to a basis state with a fresh ancilla set, where the old state has no amplitude.  So the
   state changes iff an active branch has non-zero amplitude.  Active branches: jump, pawn step,
   slide: bit s != bit t (slides: every typed path square empty); capture jump / slide: s or t
   occupied (slides: path empty); pawn capture: s and t occupied.

   Split and merge mix amplitudes.  Their one- and two-particle 3x3 blocks each have a
   one-dimensional eigenvalue-1 eigenspace (other eigenvalues at distance 1.514 from 1) whose
   eigenvector has every component non-zero (|components| 0.357, 0.863, 0.357).  So a group of
   branches (equal bits off the three squares) that holds some but not all of the three local
   one-particle (or two-particle) basis states is moved, and only a full triple needs the
   amplitudes; then the 8x8 local unitary is applied to that group.

3. **Support-only decisions on regular states.**  If every branch weight |a|^2 is at least
   ``REG_MIN_W`` (1e-8) and the squared norm is 1 within ``REG_NORM_TOL`` (1e-11), each threshold
   the board uses (p <= EPS and p >= 1 - EPS with EPS = 1e-10; the 1e-9 comparison of
   ``SparseState.close_to``) is decided by which basis states are present, so the move list is a
   function of (n, turn, types, set of basis states), except full-triple splits/merges, which
   are recomputed from the amplitudes.  Proof sketch: p is 0, at least 1e-8, or the full sum;
   an active branch has |a| >= 1e-4, so the deviation is >= 1e-4 (permutations) or >= 2e-5
   (partial triples), far above 1e-9.

   Other states take the exact path: probabilities summed with the board's own terms in the
   board's own order (the same floats); an active branch with |a| > ``ACTIVE_AMP`` (1e-6) means
   "changes", no active branch means "unchanged", and anything between (and any split/merge
   deviation between ``DEV_SAME`` and ``DEV_CHANGED``) runs a replica of
   ``_unitary_changes_state`` on the cached matrices.

4. **A move-list cache** for regular positions keyed on (n, turn, quantum flag, types, sorted
   basis states), FIFO-bounded at ``CACHE_MAX`` entries (``CACHE_MAX <= 0``: no caching).
   Full-triple splits/merges are stored as "recheck" items and re-decided from the amplitudes on
   every hit.  ``set_cache(False)`` turns it off; ``clear_cache()``; ``cache_info()``.

   Keys and values hold only ints, strings, booleans and None (moves are stored as indices
   into the per-board-size table ``_Tables.moves``), in flat tuples, so the garbage collector
   untracks every entry after the first collection it survives: a full cache adds almost
   nothing to collection pauses, which run inside an engine's timed ``choose``.  (With the
   former layout, a frozenset key and tuples of ``Move`` objects, about 8 tracked objects per
   entry, a full collection took ~46 ms at 45,000 entries against ~9 ms empty; review of
   19 October 2026.)  Eviction drops the oldest ``CACHE_MAX // 16`` entries at a time (at
   least enough to stay within ``CACHE_MAX``), so no single store pays for a large purge.

5. **Split / Merge Slide** (boards with ``slider_splits=True``, 20 October 2026).  Their unitary is
   block-diagonal over the control squares of the two arms (``gates.split_slide_plan``): U_split /
   U_merge where both arms are clear, a full iSWAP / iSWAP^dag where one arm is blocked, the identity
   where both are.  Change test: group the branches by everything off the three squares (so a group
   lies in one block); a permutation block changes the state iff it holds an active branch (the two
   squares of its iSWAP differ), the mixing block iff a group is a partial triple (as for split /
   merge jumps); full triples (amplitude-dependent, "recheck" in the cache) and irregular states run
   the exact replica of ``QuantumBoard._unitary_changes_state``.  With the default
   ``slider_splits=False`` none of this code runs and the cache keys are unchanged.

Nothing here mutates the board passed to ``fast_legal_moves`` or ``fast_apply`` except, exactly
as ``board.copy()`` (hence the original ``base.apply``) always did, by drawing one integer from
``board.rng`` to seed the child's generator, so random streams are unchanged as well.
"""
from __future__ import annotations

from itertools import islice

import numpy as np

from .. import gates
from ..board import EPS, SLIDER_QUANTUM, IllegalMove, MoveResult, QuantumBoard
from ..geometry import JUMPERS, SLIDERS, WHITE, Geometry
from ..moves import MEASURING, Move, MoveType, SliderMoveType
from ..sparse import AMP_EPS, SparseState

# ---- tunables (module attributes, so tests can force the exact paths) ---------------------------
REG_MIN_W = 1e-8          # smallest branch weight for the support-only path
REG_NORM_TOL = 1e-11      # |sum of weights - 1| allowed on the support-only path
ACTIVE_AMP = 1e-6         # an active branch with |a| above this certainly changes the state
DEV_CHANGED = 1e-7        # split/merge: max local deviation above this = changes
DEV_SAME = 1e-10          # split/merge: max local deviation below this = unchanged
CACHE_MAX = 50_000        # move-list cache entries (<= 0: no caching)
EVICT_DIV = 16            # a full cache drops its oldest CACHE_MAX // EVICT_DIV entries at once
PLAN_MAX = 20_000         # cached gate-application plans

_MT = MoveType
_JUMPLIKE = (_MT.JUMP, _MT.BLOCKED_JUMP, _MT.PAWN_STEP, _MT.BLOCKED_PAWN_STEP)
_SLIDES = (_MT.SLIDE, _MT.BLOCKED_SLIDE, _MT.CAPTURE_SLIDE)
_NEED = {_MT.CAPTURE_JUMP: 1, _MT.CAPTURE_SLIDE: 1, _MT.PAWN_CAPTURE: 2}  # QuantumBoard._ancillas_needed


# =================================================================================================
# gate matrices (identical to QuantumBoard.unitary_ops + gates.matrix)
# =================================================================================================
_MATS: dict = {}


def _mat(key) -> np.ndarray:
    m = _MATS.get(key)
    if m is None:
        name, k = key
        if name == "iswap":
            g = gates.ISWAP
        elif name == "iswap_dg":
            g = gates.ISWAP_DG
        elif name == "sqrt_iswap":
            g = gates.SQRT_ISWAP
        elif name == "sqrt_iswap_dg":
            g = gates.SQRT_ISWAP_DG
        elif name == "oc_iswap":
            g = gates.open_controlled(gates.ISWAP, k)
        elif name == "cc_x":
            g = gates.closed_controlled(gates.X, 2)
        elif name == "c_iswap":
            g = gates.closed_controlled(gates.ISWAP, 1)
        else:  # pragma: no cover
            raise KeyError(key)
        m = gates.matrix(g)
        _MATS[key] = m
    return m


_ISWAP = ("iswap", 0)
_ISWAP_DG = ("iswap_dg", 0)
_SQRT = ("sqrt_iswap", 0)
_SQRT_DG = ("sqrt_iswap_dg", 0)
_CCX = ("cc_x", 0)
_CISWAP = ("c_iswap", 0)


def _path_controls(board: QuantumBoard, s: int, t: int) -> list[int]:
    """``board.path_controls(s, t)``."""
    types = board.types
    return [q for q in _tables(board.n).path[s][t] if types[q] is not None]


def unitary_op_keys(board: QuantumBoard, m: Move, mt: MoveType, c=None, p=None) -> list[tuple]:
    """``QuantumBoard.unitary_ops`` with matrix-cache keys in place of Qiskit gates."""
    s, t = m.source, m.target
    if mt in _JUMPLIKE:
        return [(_ISWAP, [s, t])]
    if mt is _MT.SPLIT_JUMP:
        return [(_SQRT, [s, t]), (_ISWAP, [s, m.target2])]
    if mt is _MT.MERGE_JUMP:
        return [(_ISWAP_DG, [s, t]), (_SQRT_DG, [m.source2, t])]
    if mt is _MT.SLIDE or mt is _MT.BLOCKED_SLIDE:
        path = _path_controls(board, s, t)
        if not path:
            return [(_ISWAP, [s, t])]
        return [(("oc_iswap", len(path)), [*path, s, t])]
    if mt is _MT.CAPTURE_JUMP:
        return [(_ISWAP, [t, c]), (_ISWAP, [s, t])]
    if mt is _MT.CAPTURE_SLIDE:
        path = _path_controls(board, s, t)
        if not path:
            return [(_ISWAP, [t, c]), (_ISWAP, [s, t])]
        g = ("oc_iswap", len(path))
        return [(g, [*path, t, c]), (g, [*path, s, t])]
    if mt is _MT.PAWN_CAPTURE:
        return [(_CCX, [s, t, p]), (_CISWAP, [p, t, c]), (_CISWAP, [p, s, t])]
    raise ValueError(mt)


def unitary_ops(board: QuantumBoard, m: Move, mt: MoveType, c=None, p=None) -> list[tuple]:
    """``[(matrix, qubits)]``: the matrices of ``board.unitary_ops`` without rebuilding gates."""
    return [(_mat(k), qs) for k, qs in unitary_op_keys(board, m, mt, c, p)]


_PLANS: dict = {}


def _apply(state: SparseState, key, qubits: list[int]) -> None:
    """``state.apply(_mat(key), qubits)`` with the column scans and output masks cached.

    The products ``v * a`` (``v`` the same numpy scalar ``mat[jj, i]``), the accumulation
    ``new.get(nb, 0j) + v * a`` and its order (branches in dict order, ``jj`` ascending) are those
    of ``SparseState.apply``, so the result is bitwise identical."""
    pk = (key, tuple(qubits))
    plan = _PLANS.get(pk)
    if plan is None:
        mat = _mat(key)
        k = len(qubits)
        if mat.shape != (1 << k, 1 << k):  # pragma: no cover - mirrors SparseState.apply
            raise ValueError(f"matrix shape {mat.shape} does not match {k} qubits")
        masks = tuple(1 << q for q in qubits)
        all_mask = 0
        for m in masks:
            all_mask |= m
        outm = []
        for jj in range(1 << k):
            nb = 0
            for j, m in enumerate(masks):
                if (jj >> j) & 1:
                    nb |= m
            outm.append(nb)
        if len(_PLANS) >= PLAN_MAX:
            _PLANS.clear()
        # tuples of ints / numpy scalars only, so the garbage collector can untrack them
        plan = _PLANS[pk] = (all_mask, masks, tuple(outm), mat, {})
    all_mask, masks, outm, mat, cols = plan
    new: dict = {}
    get = new.get
    for b, a in state.amps.items():
        x = b & all_mask
        nz = cols.get(x)
        if nz is None:
            i = 0
            for j, m in enumerate(masks):
                if x & m:
                    i |= 1 << j
            col = mat[:, i]
            nz = cols[x] = tuple((outm[jj], col[jj]) for jj in np.nonzero(np.abs(col) > AMP_EPS)[0])
        base = b ^ x
        for om, v in nz:
            nb = base | om
            new[nb] = get(nb, 0j) + v * a
    state.amps = {b: a for b, a in new.items() if abs(a) > AMP_EPS}


def _apply_restricted(state: SparseState, key, qubits: list[int], zm: int, om: int) -> None:
    """``state.apply_restricted(_mat(key), qubits, zm, om)`` with the column scans cached; the same
    products, sums and dict order, so the result is bitwise identical."""
    pk = (key, tuple(qubits))
    plan = _PLANS.get(pk)
    if plan is None:
        _apply_plan_entry(pk, key, qubits)
        plan = _PLANS[pk]
    all_mask, masks, outm, mat, cols = plan
    if all_mask & (zm | om):  # pragma: no cover - mirrors SparseState.apply_restricted
        raise ValueError("a condition qubit is also a target")
    new: dict = {}
    get = new.get
    for b, a in state.amps.items():
        if (b & zm) or (om and not (b & om)):
            new[b] = get(b, 0j) + a
            continue
        x = b & all_mask
        nz = cols.get(x)
        if nz is None:
            i = 0
            for j, m in enumerate(masks):
                if x & m:
                    i |= 1 << j
            col = mat[:, i]
            nz = cols[x] = tuple((outm[jj], col[jj]) for jj in np.nonzero(np.abs(col) > AMP_EPS)[0])
        base = b ^ x
        for o, v in nz:
            nb = base | o
            new[nb] = get(nb, 0j) + v * a
    state.amps = {b: a for b, a in new.items() if abs(a) > AMP_EPS}


def _apply_plan_entry(pk, key, qubits) -> None:
    mat = _mat(key)
    masks = tuple(1 << q for q in qubits)
    all_mask = 0
    for m in masks:
        all_mask |= m
    outm = []
    for jj in range(1 << len(qubits)):
        nb = 0
        for j, m in enumerate(masks):
            if (jj >> j) & 1:
                nb |= m
        outm.append(nb)
    if len(_PLANS) >= PLAN_MAX:
        _PLANS.clear()
    _PLANS[pk] = (all_mask, masks, tuple(outm), mat, {})


_GATE_KEY: dict = {}


def _gate_key(gate) -> tuple:
    if not _GATE_KEY:
        _GATE_KEY.update({id(gates.ISWAP): _ISWAP, id(gates.ISWAP_DG): _ISWAP_DG,
                          id(gates.SQRT_ISWAP): _SQRT, id(gates.SQRT_ISWAP_DG): _SQRT_DG})
    return _GATE_KEY[id(gate)]


def slider_arms(board: QuantumBoard, m: Move, mt: MoveType) -> tuple[list, list]:
    """``board.slider_arms(m, mt)`` from the path tables."""
    types = board.types
    path = _tables(board.n).path
    if mt is SliderMoveType.SPLIT_SLIDE:
        s, a, b = m.source, m.target, m.target2
        return ([q for q in path[s][a] if types[q] is not None and q != b],
                [q for q in path[s][b] if types[q] is not None and q != a])
    s1, s2, t = m.source, m.source2, m.target
    return ([q for q in path[s1][t] if types[q] is not None and q != s2],
            [q for q in path[s2][t] if types[q] is not None and q != s1])


def slider_plan(board: QuantumBoard, m: Move, mt: MoveType) -> list:
    """``board.slider_plan(m, mt)``."""
    c1, c2 = slider_arms(board, m, mt)
    if mt is SliderMoveType.SPLIT_SLIDE:
        return gates.split_slide_plan(m.source, m.target, m.target2, c1, c2)
    return gates.merge_slide_plan(m.source, m.source2, m.target, c1, c2)


def _apply_slider_plan(state: SparseState, steps: list) -> None:
    """``gates.apply_plan(state, steps)`` on the cached matrices."""
    for gate, zeros, ones, ab in steps:
        zm = 0
        for q in zeros:
            zm |= 1 << q
        om = 0
        for q in ones:
            om |= 1 << q
        _apply_restricted(state, _gate_key(gate), ab, zm, om)


def _exact_changes_state(board: QuantumBoard, m: Move, mt: MoveType) -> bool:
    """Replica of ``QuantumBoard._unitary_changes_state`` on cached matrices (the fallback)."""
    if mt in SLIDER_QUANTUM:
        tmp = board.state.copy()
        _apply_slider_plan(tmp, slider_plan(board, m, mt))
        return not tmp.close_to(board.state)
    state = board.state
    tmp = state.copy()
    extra = [tmp.add_qubit() for _ in range(_NEED.get(mt, 0))]
    for key, qs in unitary_op_keys(board, m, mt, *extra):
        _apply(tmp, key, qs)
    ref = SparseState(tmp.nq, state.amps)
    return not tmp.close_to(ref)


# split (roles s, t1, t2) and merge (roles s1, s2, t): 8x8 local unitaries as sparse columns
_LOCAL: dict = {}


def local_unitary(kind_: str) -> np.ndarray:
    ops = gates.split_jump_ops(0, 1, 2) if kind_ == "split" else gates.merge_jump_ops(0, 1, 2)
    return gates.ops_unitary(ops, [0, 1, 2])


def _local_cols(kind_: str) -> list:
    c = _LOCAL.get(kind_)
    if c is None:
        u = local_unitary(kind_)
        c = _LOCAL[kind_] = [[(j, complex(u[j, i])) for j in range(8) if abs(u[j, i]) > 1e-15]
                             for i in range(8)]
    return c


# =================================================================================================
# per-board-size tables (built from Geometry, so orders match QuantumBoard.legal_moves exactly)
# =================================================================================================
class _Tables:
    """Move tables in ``QuantumBoard.legal_moves`` order.  Every entry ends with its ``Move`` and
    that move's index ``i`` in ``self.moves`` (``self.moves[i] is`` the entry's move), which is
    what the move-list cache stores."""

    def __init__(self, n: int):
        geo = Geometry(n)
        sq = list(geo.squares())
        self.n = n
        self.nsq = n * n
        self.geo = geo
        self.moves: list[Move] = []
        self.path = [[[] for _ in sq] for _ in sq]
        for s in sq:
            for t in sq:
                try:
                    self.path[s][t] = geo.path(s, t)
                except ValueError:
                    pass
        self.path_mask = [[sum(1 << q for q in p) for p in row] for row in self.path]
        self.jumps: dict[str, list] = {}
        self.jump_set: dict[str, list] = {}
        self.splits: dict[str, list] = {}
        self.merges: dict[str, dict] = {}
        reg = self._reg
        for k in "NK":
            jt = [geo.jump_targets(s, k) for s in sq]
            self.jump_set[k] = [set(x) for x in jt]
            self.jumps[k] = [[(t, *reg(Move(s, t))) for t in jt[s]] for s in sq]
            self.splits[k] = [[(t1, t2, *reg(Move(s, t1, target2=t2))) for t1 in jt[s] for t2 in jt[s] if t1 != t2]
                              for s in sq]
            mg = {}
            for s in sq:
                for s2 in sq:
                    if s2 != s:
                        # the very expression legal_moves() iterates, hence the same order
                        common = set(jt[s]) & set(geo.jump_targets(s2, k))
                        mg[(s, s2)] = [(t, *reg(Move(s, t, source2=s2))) for t in common]
            self.merges[k] = mg
        self.rays: dict[str, list] = {}
        self.ray_set: dict[str, list] = {}
        for k in "BRQ":
            self.rays[k] = [[(t, *reg(Move(s, t))) for t, _ in geo.ray_targets(s, k)] for s in sq]
            self.ray_set[k] = [{e[0] for e in r} for r in self.rays[k]]
        self.pawn_step: dict[str, list] = {}
        self.pawn_caps: dict[str, list] = {}
        self.pawn_cap_set: dict[str, list] = {}
        for color in ("w", "b"):
            steps, caps = [], []
            for s in sq:
                t = geo.pawn_step(s, color)
                steps.append(None if t is None else (t, *reg(Move(s, t))))
                caps.append([(t, *reg(Move(s, t))) for t in geo.pawn_captures(s, color)])
            self.pawn_step[color] = steps
            self.pawn_caps[color] = caps
            self.pawn_cap_set[color] = [{e[0] for e in c} for c in caps]
        self.last_rank = {"w": n - 1, "b": 0}
        self.text: dict[Move, str] = {}
        self._slider_q = None

    def slider_q(self) -> tuple[dict, dict]:
        """Split / Merge Slide tables (built on first use, appended to ``self.moves``), in
        ``QuantumBoard.legal_moves`` order: ``splits[k][s] = [(t1, t2, Move, i)]``,
        ``merges[k][(s, s2)] = [(t, Move, i)]``."""
        if self._slider_q is None:
            reg = self._reg
            sq = range(self.nsq)
            splits, merges = {}, {}
            for k in "BRQ":
                rays = [[e[0] for e in self.rays[k][s]] for s in sq]
                splits[k] = [[(t1, t2, *reg(Move(s, t1, target2=t2))) for t1 in rays[s] for t2 in rays[s] if t1 != t2]
                             for s in sq]
                mg = {}
                for s in sq:
                    for s2 in sq:
                        if s2 != s:
                            r2 = self.ray_set[k][s2]
                            mg[(s, s2)] = [(t, *reg(Move(s, t, source2=s2))) for t in rays[s] if t in r2]
                merges[k] = mg
            self._slider_q = (splits, merges)
        return self._slider_q

    def _reg(self, mv: Move) -> tuple[Move, int]:
        """``(mv, its index in self.moves)``."""
        self.moves.append(mv)
        return mv, len(self.moves) - 1

    def move_text(self, m: Move) -> str:
        s = self.text.get(m)
        if s is None:
            s = self.text[m] = m.text(self.geo)
        return s


_TABLES: dict[int, _Tables] = {}


def _tables(n: int) -> _Tables:
    t = _TABLES.get(n)
    if t is None:
        t = _TABLES[n] = _Tables(n)
    return t


# =================================================================================================
# per-position context
# =================================================================================================
def _weights(amps: dict) -> tuple[bool, list]:
    """(regular?, branch weights in dict order)."""
    ws = [abs(a) ** 2 for a in amps.values()]
    return (bool(ws) and min(ws) >= REG_MIN_W and abs(sum(ws) - 1.0) <= REG_NORM_TOL), ws


class _Ctx:
    __slots__ = ("board", "amps", "regular", "any_", "all_", "pats", "bw", "items", "bitp",
                 "cacheable", "ampdep")

    def __init__(self, board: QuantumBoard, regular: bool | None = None, ws: list | None = None):
        self.board = board
        amps = board.state.amps
        self.amps = amps
        if regular is None:
            regular, ws = _weights(amps)
        self.regular = regular
        self.cacheable = True
        self.ampdep = False
        self.bitp: dict[int, float] = {}
        if regular:
            any_, all_ = 0, -1
            for b in amps:
                any_ |= b
                all_ &= b
            self.any_, self.all_ = any_, all_
            bm = (1 << board.nsq) - 1
            self.pats = list({b & bm for b in amps})
            self.bw = self.items = None
        else:
            self.any_ = self.all_ = 0
            self.pats = None
            self.bw = list(zip(amps.keys(), ws))
            self.items = list(amps.items())

    def bit_prob(self, q: int) -> float:
        """``board.prob(q)``: same terms, same order, same float."""
        p = self.bitp.get(q)
        if p is None:
            m = 1 << q
            p = self.bitp[q] = float(sum(w for b, w in self.bw if b & m))
        return p

    # ---- the pieces of QuantumBoard._legal_given_type -------------------------------------------
    def path_blocked(self, s: int, t: int, tab: _Tables) -> bool:
        """``board.path_classically_blocked(s, t)``."""
        if self.regular:
            return bool(self.all_ & tab.path_mask[s][t])
        return any(self.bit_prob(q) >= 1 - EPS for q in tab.path[s][t])

    def measuring(self, kind_: int, s: int, t: int, pmask: int) -> int:
        """Class of the outcome-1 probability p1 of a measuring move: 0 if p1 <= EPS, 2 if
        p1 >= 1 - EPS, else 1.  kind_ 0: blocked moves (M1 = target empty); 1: capture jump and
        pawn capture (M1 = source occupied); 2: capture slide (``m1_predicate``, path ``pmask``)."""
        S, T = 1 << s, 1 << t
        if kind_ == 0:
            def pred(b):
                return not b & T
        elif kind_ == 1:
            def pred(b):
                return b & S
        else:
            def pred(b):
                blocked = b & pmask
                return (not blocked and b & S) or (blocked and not b & T)
        if self.regular:
            n_true = 0
            for p in self.pats:
                if pred(p):
                    n_true += 1
            return 0 if n_true == 0 else 2 if n_true == len(self.pats) else 1
        p1 = float(sum(w for b, w in self.bw if pred(b)))
        return 0 if p1 <= EPS else 2 if p1 >= 1 - EPS else 1

    def perm_changes(self, ctrl: int, M: int, act: tuple, m: Move, mt: MoveType) -> bool:
        """Change test for the phase-permutation unitaries.  Active branch: every control square
        empty (``b & ctrl == 0``) and local pattern ``b & M`` in ``act``."""
        if self.regular:
            for p in self.pats:
                if not p & ctrl and (p & M) in act:
                    return True
            return False
        big = -1.0
        for b, a in self.items:
            if not b & ctrl and (b & M) in act:
                x = abs(a)
                if x > big:
                    big = x
        if big < 0:
            return False
        if big > ACTIVE_AMP:
            return True
        return _exact_changes_state(self.board, m, mt)

    def mix_changes(self, r0: int, r1: int, r2: int, m: Move, mt: MoveType, kind_: str) -> bool:
        """Change test for split (roles s, t1, t2) and merge (roles s1, s2, t); r* are bit masks."""
        M = r0 | r1 | r2
        if self.regular:
            for p in self.pats:
                x = p & M
                if x and x != M:
                    break
            else:
                return False           # no active branch
            if M & ~self.any_:
                return True            # a square of the triple is never occupied: groups partial
            groups: dict[int, set] = {}
            for b in self.amps:
                x = b & M
                if x and x != M:
                    groups.setdefault(b & ~M, set()).add(x)
            for g in groups.values():
                n1 = sum(1 for x in g if x == r0 or x == r1 or x == r2)
                n2 = len(g) - n1
                if 0 < n1 < 3 or 0 < n2 < 3:
                    return True        # some group is off the eigenvalue-1 eigenvector
            self.ampdep = True         # every active group is a full triple: amplitudes decide
        dev = self._local_deviation(r0, r1, r2, M, kind_)
        if dev > DEV_CHANGED:
            return True
        if dev < DEV_SAME:
            return False
        return _exact_changes_state(self.board, m, mt)

    def slider_changes(self, m: Move, mt: MoveType) -> bool:
        """Change test for a Split / Merge Slide (module docstring, item 5)."""
        board = self.board
        if not self.regular:
            return _exact_changes_state(board, m, mt)
        c1, c2 = slider_arms(board, m, mt)
        c1m = c2m = 0
        for q in c1:
            c1m |= 1 << q
        for q in c2:
            c2m |= 1 << q
        if mt is SliderMoveType.SPLIT_SLIDE:
            r0, r1, r2 = 1 << m.source, 1 << m.target, 1 << m.target2
            pa, pb = (r0, r1), (r0, r2)          # arm 1 clear only: iSWAP(s, t1); arm 2 only: (s, t2)
        else:
            r0, r1, r2 = 1 << m.source, 1 << m.source2, 1 << m.target
            pa, pb = (r0, r2), (r1, r2)          # arm 1 clear only: iSWAP^dag(s1, t); arm 2: (s2, t)
        M = r0 | r1 | r2
        groups: dict[int, set] = {}
        for b in self.amps:
            cl1 = not b & c1m
            cl2 = not b & c2m
            if cl1 and cl2:
                x = b & M
                if x and x != M:
                    g = groups.get(b & ~M)
                    if g is None:
                        groups[b & ~M] = {x}
                    else:
                        g.add(x)
            elif cl1:
                if (not b & pa[0]) != (not b & pa[1]):
                    return True
            elif cl2:
                if (not b & pb[0]) != (not b & pb[1]):
                    return True
        if not groups:
            return False
        for g in groups.values():
            n1 = sum(1 for x in g if x == r0 or x == r1 or x == r2)
            n2 = len(g) - n1
            if 0 < n1 < 3 or 0 < n2 < 3:
                return True
        self.ampdep = True                   # only full triples: the amplitudes decide
        return _exact_changes_state(board, m, mt)

    def _local_deviation(self, r0, r1, r2, M, kind_) -> float:
        """max |(U v - v)_j| over the groups' local 8-vectors v (exact up to rounding)."""
        cols = _local_cols(kind_)
        groups: dict[int, dict] = {}
        for b, a in self.amps.items():
            x = b & M
            if not x or x == M:
                continue
            g = groups.get(b & ~M)
            if g is None:
                g = groups[b & ~M] = {}
            g[(1 if b & r0 else 0) | (2 if b & r1 else 0) | (4 if b & r2 else 0)] = a
        dev = 0.0
        for g in groups.values():
            out: dict[int, complex] = {}
            for i, a in g.items():
                for j, u in cols[i]:
                    out[j] = out.get(j, 0j) + u * a
            for j in out.keys() | g.keys():
                d = abs(out.get(j, 0j) - g.get(j, 0j))
                if d > dev:
                    dev = d
        return dev


# =================================================================================================
# legality of one classified move (QuantumBoard._legal_given_type)
# =================================================================================================
def _ctrl_mask(board: QuantumBoard, tab: _Tables, s: int, t: int) -> int:
    """Bit mask of ``board.path_controls(s, t)`` (the typed path squares)."""
    types = board.types
    mask = 0
    for q in tab.path[s][t]:
        if types[q] is not None:
            mask |= 1 << q
    return mask


def _legal(ctx: _Ctx, tab: _Tables, m: Move, mt: MoveType) -> bool:
    s, t = m.source, m.target
    if mt in _SLIDES and ctx.path_blocked(s, t, tab):
        return False
    S, T = 1 << s, 1 << t
    if mt is _MT.JUMP or mt is _MT.PAWN_STEP:
        return ctx.perm_changes(0, S | T, (S, T), m, mt)
    if mt is _MT.SPLIT_JUMP:
        return ctx.mix_changes(S, T, 1 << m.target2, m, mt, "split")
    if mt is _MT.MERGE_JUMP:
        return ctx.mix_changes(S, 1 << m.source2, T, m, mt, "merge")
    if mt is SliderMoveType.SPLIT_SLIDE or mt is SliderMoveType.MERGE_SLIDE:
        return ctx.slider_changes(m, mt)
    if mt is _MT.SLIDE:
        return ctx.perm_changes(_ctrl_mask(ctx.board, tab, s, t), S | T, (S, T), m, mt)
    # measuring moves: illegal if p1 <= EPS; if p1 >= 1 - EPS the unitary must change the state
    if mt is _MT.BLOCKED_JUMP or mt is _MT.BLOCKED_PAWN_STEP:
        c = ctx.measuring(0, s, t, 0)
        return c == 1 or (c == 2 and ctx.perm_changes(0, S | T, (S, T), m, mt))
    if mt is _MT.BLOCKED_SLIDE:
        c = ctx.measuring(0, s, t, 0)
        return c == 1 or (c == 2 and ctx.perm_changes(_ctrl_mask(ctx.board, tab, s, t), S | T, (S, T), m, mt))
    if mt is _MT.CAPTURE_JUMP:
        c = ctx.measuring(1, s, t, 0)
        return c == 1 or (c == 2 and ctx.perm_changes(0, S | T, (S, T, S | T), m, mt))
    if mt is _MT.PAWN_CAPTURE:
        c = ctx.measuring(1, s, t, 0)
        return c == 1 or (c == 2 and ctx.perm_changes(0, S | T, (S | T,), m, mt))
    if mt is _MT.CAPTURE_SLIDE:
        pm = _ctrl_mask(ctx.board, tab, s, t)
        c = ctx.measuring(2, s, t, pm)
        return c == 1 or (c == 2 and ctx.perm_changes(pm, S | T, (S, T, S | T), m, mt))
    raise ValueError(mt)  # pragma: no cover


def fast_classify(board: QuantumBoard, m: Move) -> MoveType:
    """``board.classify(m)``.  Every case this does not settle (all illegal ones) is handed to
    ``board.classify``, which raises the same ``IllegalMove``."""
    tab = _tables(board.n)
    v = board.types
    s, t = m.source, m.target
    vs = v[s]
    white = board.turn == WHITE
    if vs is not None and vs.isupper() == white:
        k = vs.upper()
        if m.target2 is not None:
            t2 = m.target2
            if k in JUMPERS and t != t2 and t in tab.jump_set[k][s] and t2 in tab.jump_set[k][s]:
                v1, v2 = v[t], v[t2]
                if (v1 is None or v1 == vs) and (v2 is None or v2 == vs):
                    return _MT.SPLIT_JUMP
            elif (k in SLIDERS and board.slider_splits and t != t2 and t in tab.ray_set[k][s]
                  and t2 in tab.ray_set[k][s]):
                v1, v2 = v[t], v[t2]
                if (v1 is None or v1 == vs) and (v2 is None or v2 == vs):
                    return SliderMoveType.SPLIT_SLIDE
        elif m.source2 is not None:
            s2 = m.source2
            if (k in JUMPERS and s != s2 and v[s2] == vs and t in tab.jump_set[k][s]
                    and t in tab.jump_set[k][s2]):
                vt = v[t]
                if vt is None or vt == vs:
                    return _MT.MERGE_JUMP
            elif (k in SLIDERS and board.slider_splits and s != s2 and v[s2] == vs and t in tab.ray_set[k][s]
                  and t in tab.ray_set[k][s2]):
                vt = v[t]
                if vt is None or vt == vs:
                    return SliderMoveType.MERGE_SLIDE
        else:
            vt = v[t]
            same = vt is None or vt == vs
            if k == "P":
                if tab.pawn_step[board.turn][s] is not None and t == tab.pawn_step[board.turn][s][0]:
                    return _MT.PAWN_STEP if same else _MT.BLOCKED_PAWN_STEP
                if t in tab.pawn_cap_set[board.turn][s] and vt is not None and vt.isupper() != white:
                    return _MT.PAWN_CAPTURE
            elif k in JUMPERS:
                if t in tab.jump_set[k][s]:
                    return _MT.JUMP if same else (_MT.BLOCKED_JUMP if vt.isupper() == white else _MT.CAPTURE_JUMP)
            elif k in SLIDERS:
                if t in tab.ray_set[k][s]:
                    return _MT.SLIDE if same else (_MT.BLOCKED_SLIDE if vt.isupper() == white else _MT.CAPTURE_SLIDE)
    return board.classify(m)


# =================================================================================================
# move generation
# =================================================================================================
# key: (n, turn, quantum, *types, *sorted basis states), flat (len(types) == n * n);
# value: a tuple of move indices into _tables(n).moves, or, if some split/merge must be re-decided
# from the amplitudes on every hit, _RECHECK followed by indices, ~index marking those moves.
_CACHE: dict = {}
_CACHE_ON = True
_RECHECK = None
_STATS = {"calls": 0, "regular": 0, "hits": 0, "stored": 0, "rechecks": 0}


def set_cache(on: bool) -> None:
    global _CACHE_ON
    _CACHE_ON = bool(on)


def clear_cache() -> None:
    _CACHE.clear()
    for k in _STATS:
        _STATS[k] = 0


def cache_info() -> dict:
    return dict(_STATS, size=len(_CACHE), enabled=_CACHE_ON)


def _trim(limit: int) -> None:
    """Drop the oldest entries so that ``len(_CACHE) <= limit``, in chunks of at least
    ``CACHE_MAX // EVICT_DIV`` when something must go."""
    n = len(_CACHE)
    if n <= limit:
        return
    if limit <= 0:
        _CACHE.clear()
        return
    drop = min(n, max(n - limit, CACHE_MAX // max(1, EVICT_DIV)))
    for k in list(islice(iter(_CACHE), drop)):
        del _CACHE[k]


def _store(key: tuple, rec: list) -> None:
    """Cache ``rec`` (indices, ``~index`` for re-decided moves) under ``key``, within ``CACHE_MAX``."""
    _trim(CACHE_MAX - 1)
    _CACHE[key] = (_RECHECK, *rec) if rec and min(rec) < 0 else tuple(rec)
    _STATS["stored"] += 1


def fast_legal_moves(board: QuantumBoard, quantum: bool = True, use_cache: bool = True) -> list[Move]:
    """``board.legal_moves()``: the same moves in the same order.  ``quantum=False`` leaves out
    splits and merges: ``[m for m in board.legal_moves() if not (m.is_split or m.is_merge)]``."""
    _STATS["calls"] += 1
    regular, ws = _weights(board.state.amps)
    key = None
    tab = _tables(board.n)
    if regular:
        _STATS["regular"] += 1
        if use_cache and _CACHE_ON:
            if CACHE_MAX > 0:
                # boards with slider splits get their own keys (the default keys are unchanged)
                key = (board.n, board.turn, (quantum, "ss") if board.slider_splits else quantum,
                       *board.types, *sorted(board.state.amps))
                hit = _CACHE.get(key)
                if hit is not None:
                    _STATS["hits"] += 1
                    moves = tab.moves
                    if not hit or hit[0] is not _RECHECK:
                        return [moves[i] for i in hit]
                    _STATS["rechecks"] += 1
                    ctx = _Ctx(board, regular, ws)
                    out = []
                    for i in islice(hit, 1, None):
                        if i >= 0:
                            out.append(moves[i])
                        else:
                            mv = moves[~i]
                            if board.types[mv.source].upper() in JUMPERS:
                                mt = _MT.SPLIT_JUMP if mv.target2 is not None else _MT.MERGE_JUMP
                            else:
                                mt = SliderMoveType.SPLIT_SLIDE if mv.target2 is not None else SliderMoveType.MERGE_SLIDE
                            if _legal(ctx, tab, mv, mt):
                                out.append(mv)
                    return out
            elif _CACHE:
                _CACHE.clear()          # CACHE_MAX lowered to <= 0 at run time: caching is off
    ctx = _Ctx(board, regular, ws)
    rec: list | None = [] if key is not None else None
    out = _generate(board, ctx, tab, quantum, rec)
    if key is not None:
        _store(key, rec)
    return out


def _generate(board: QuantumBoard, ctx: _Ctx, tab: _Tables, quantum: bool, rec: list | None) -> list[Move]:
    """The loop of ``QuantumBoard.legal_moves`` with classification inlined.  ``rec`` (if given)
    receives the table index of every legal move and, in place, ``~index`` for every
    amplitude-dependent candidate (a full-triple split/merge, legal or not)."""
    v = board.types
    me = board.turn
    white = me == WHITE
    out: list[Move] = []
    add = out.append
    note = None if rec is None else rec.append
    nsq = tab.nsq
    for s in range(nsq):
        vs = v[s]
        if vs is None or vs.isupper() != white:
            continue
        k = vs.upper()
        if k in JUMPERS:
            for t, mv, i in tab.jumps[k][s]:
                vt = v[t]
                if vt is None or vt == vs:
                    mt = _MT.JUMP
                elif vt.isupper() == white:
                    mt = _MT.BLOCKED_JUMP
                else:
                    mt = _MT.CAPTURE_JUMP
                if _legal(ctx, tab, mv, mt):
                    add(mv)
                    if note is not None:
                        note(i)
            if not quantum:
                continue
            for t1, t2, mv, i in tab.splits[k][s]:
                v1, v2 = v[t1], v[t2]
                if (v1 is None or v1 == vs) and (v2 is None or v2 == vs):
                    ctx.ampdep = False
                    ok = _legal(ctx, tab, mv, _MT.SPLIT_JUMP)
                    if ok:
                        add(mv)
                    if note is not None and (ok or ctx.ampdep):
                        note(~i if ctx.ampdep else i)
            for s2 in range(nsq):
                if s2 != s and v[s2] == vs:
                    for t, mv, i in tab.merges[k][(s, s2)]:
                        vt = v[t]
                        if vt is None or vt == vs:
                            ctx.ampdep = False
                            ok = _legal(ctx, tab, mv, _MT.MERGE_JUMP)
                            if ok:
                                add(mv)
                            if note is not None and (ok or ctx.ampdep):
                                note(~i if ctx.ampdep else i)
        elif k in SLIDERS:
            for t, mv, i in tab.rays[k][s]:
                vt = v[t]
                if vt is None or vt == vs:
                    mt = _MT.SLIDE
                elif vt.isupper() == white:
                    mt = _MT.BLOCKED_SLIDE
                else:
                    mt = _MT.CAPTURE_SLIDE
                if _legal(ctx, tab, mv, mt):
                    add(mv)
                    if note is not None:
                        note(i)
            if not (quantum and board.slider_splits):
                continue
            ssplits, smerges = tab.slider_q()
            for t1, t2, mv, i in ssplits[k][s]:
                v1, v2 = v[t1], v[t2]
                if (v1 is None or v1 == vs) and (v2 is None or v2 == vs):
                    ctx.ampdep = False
                    ok = _legal(ctx, tab, mv, SliderMoveType.SPLIT_SLIDE)
                    if ok:
                        add(mv)
                    if note is not None and (ok or ctx.ampdep):
                        note(~i if ctx.ampdep else i)
            for s2 in range(nsq):
                if s2 != s and v[s2] == vs:
                    for t, mv, i in smerges[k][(s, s2)]:
                        vt = v[t]
                        if vt is None or vt == vs:
                            ctx.ampdep = False
                            ok = _legal(ctx, tab, mv, SliderMoveType.MERGE_SLIDE)
                            if ok:
                                add(mv)
                            if note is not None and (ok or ctx.ampdep):
                                note(~i if ctx.ampdep else i)
        else:  # pawn
            st = tab.pawn_step[me][s]
            if st is not None:
                t, mv, i = st
                vt = v[t]
                mt = _MT.PAWN_STEP if (vt is None or vt == vs) else _MT.BLOCKED_PAWN_STEP
                if _legal(ctx, tab, mv, mt):
                    add(mv)
                    if note is not None:
                        note(i)
            for t, mv, i in tab.pawn_caps[me][s]:
                vt = v[t]
                if vt is not None and vt.isupper() != white and _legal(ctx, tab, mv, _MT.PAWN_CAPTURE):
                    add(mv)
                    if note is not None:
                        note(i)
    return out


def fast_is_legal(board: QuantumBoard, m: Move) -> bool:
    """``board.is_legal(m)``."""
    try:
        mt = fast_classify(board, m)
    except IllegalMove:
        return False
    return _legal(_Ctx(board), _tables(board.n), m, mt)


# =================================================================================================
# playing moves
# =================================================================================================
class _Forced:
    """``base.ForcedRNG``: the next measurement gives ``outcome`` (``SparseState.measure`` returns
    1 iff ``rng.random() < p1``)."""
    __slots__ = ("outcome",)

    def __init__(self, outcome: int):
        self.outcome = outcome

    def random(self):
        return 0.0 if self.outcome == 1 else 1.0


def fast_push(board: QuantumBoard, m: Move | str, rng=None) -> MoveResult:
    """``board.push(m)`` in place; ``rng`` (if given) stands in for ``board.rng`` in the
    measurement, as ``ForcedRNG`` does in ``base.apply``."""
    if isinstance(m, str):
        m = Move.parse(m, board.geo)
    mt = fast_classify(board, m)
    tab = _tables(board.n)
    if not _legal(_Ctx(board), tab, m, mt):
        raise IllegalMove(f"{m.text(board.geo)} ({mt.value}) has no effect")
    state = board.state
    s, t = m.source, m.target
    vs = board.types[s]
    outcome = None
    p1 = None
    c = p = None
    if mt in MEASURING:
        pred = board.m1_predicate(m, mt)
        p1 = state.prob(pred)
        # SparseState.measure(pred, rng), whose own prob(pred) is this same p1
        outcome = 1 if (board.rng if rng is None else rng).random() < p1 else 0
        state.project(pred, outcome)
    if outcome != 0:
        need = _NEED.get(mt, 0)
        if need:
            c = state.add_qubit()
            board.captured.append(c)
            board.ancilla_notes[c] = f"captured on {board.geo.name(t)} by {tab.move_text(m)}"
        if need == 2:
            p = state.add_qubit()
            board.ancilla_notes[p] = f"pawn-capture control for {tab.move_text(m)}"
        if mt in SLIDER_QUANTUM:
            _apply_slider_plan(state, slider_plan(board, m, mt))
        else:
            for key, qs in unitary_op_keys(board, m, mt, c, p):
                _apply(state, key, qs)
        board.types[t] = vs
        if mt is _MT.SPLIT_JUMP or mt is SliderMoveType.SPLIT_SLIDE:
            board.types[m.target2] = vs
        if vs.upper() == "P" and t // board.n == tab.last_rank[board.turn]:
            board.types[t] = "Q" if board.turn == WHITE else "q"
    _clear_empty_types(board)
    res = MoveResult(m, mt, outcome, p1, tab.move_text(m))
    board.history.append(res)
    board.turn = "b" if board.turn == WHITE else "w"
    return res


def _clear_empty_types(board: QuantumBoard) -> None:
    """``board._clear_empty_types()``.  A square holding a branch of weight > EPS keeps its type
    without summing: a float sum of non-negative terms is at least each of its terms."""
    types = board.types
    amps = board.state.amps
    heavy = 0
    for b, a in amps.items():
        if abs(a) ** 2 > EPS:
            heavy |= b
    for q in range(board.nsq):
        if types[q] is not None and not (heavy >> q) & 1:
            m = 1 << q
            if float(sum(abs(a) ** 2 for b, a in amps.items() if b & m)) <= EPS:
                types[q] = None


def fast_apply(board: QuantumBoard, m: Move, outcome: int | None = None) -> QuantumBoard:
    """``base.apply``: a copy of ``board`` with ``m`` played and, for a measuring move, outcome
    ``outcome`` forced (``None`` draws it from the child's generator, as before)."""
    child = board.copy()
    fast_push(child, m, None if outcome is None else _Forced(outcome))
    return child


def fast_chance_children(board: QuantumBoard, m: Move) -> list[tuple[float, QuantumBoard]]:
    """``base.chance_children``: ``[(Born probability, child)]`` for every possible outcome."""
    mt = fast_classify(board, m)
    if mt not in MEASURING:
        return [(1.0, fast_apply(board, m, None))]
    p1 = board.state.prob(board.m1_predicate(m, mt))
    return [(p, fast_apply(board, m, o)) for o, p in ((1, p1), (0, 1.0 - p1)) if p > EPS]


__all__ = ["fast_legal_moves", "fast_is_legal", "fast_classify", "fast_push", "fast_apply",
           "fast_chance_children", "unitary_ops", "unitary_op_keys", "local_unitary", "set_cache",
           "slider_arms", "slider_plan",
           "clear_cache", "cache_info"]
