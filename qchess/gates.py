"""Qiskit gates for the Cantwell move unitaries (arXiv:1906.05836, Section "Movement Unitary Design").

Every move is expressed as a list of ``(gate, roles)`` pairs where ``gate`` is a Qiskit gate and
``roles`` names the qubits it acts on, in Qiskit order (``roles[0]`` is the gate's qubit 0).  The
same op lists drive the sparse simulator (``qchess.board``) and the hardware circuits
(``qchess.circuits``), so the two cannot drift apart.

Conventions (all checked by ``tests/test_equations.py``):

* ``ISWAP`` is eq. (iSwap); ``SQRT_ISWAP`` is eq. (iSwapRoot) — the principal square root,
  identical to ``cirq.ISWAP ** 0.5`` used by quantumlib/unitary.
* A ket written |t, s> in the paper has ``s`` as the least significant qubit, so with
  ``roles = [s, t]`` the ``Operator`` of the op list reproduces the paper's matrix verbatim.
* Path controls are *open* (zero) controls: the paper's path ancilla ``p`` is |1> when a square
  in the path is occupied and the slide acts only when ``p = 0`` (eq. SlideUnitary).  Using
  open controls on the path squares directly is the simplification in the paper's Fig.
  "SimpleStandardSlideCircuit".
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, iSwapGate
from qiskit.quantum_info import Operator

ISWAP: Gate = iSwapGate()
ISWAP_DG: Gate = iSwapGate().inverse()
SQRT_ISWAP: Gate = iSwapGate().power(0.5)
SQRT_ISWAP_DG: Gate = iSwapGate().power(-0.5)
X: Gate = XGate()

SQRT_ISWAP.name = "sqrt_iswap"
SQRT_ISWAP_DG.name = "sqrt_iswap_dg"

Op = tuple[Gate, list]


@lru_cache(maxsize=None)
def _matrix_cached(key) -> np.ndarray:
    gate = key
    return Operator(gate).data


def matrix(gate: Gate) -> np.ndarray:
    """Dense matrix of a gate (cached by gate identity)."""
    try:
        return _matrix_cached(gate)
    except TypeError:  # unhashable gate
        return Operator(gate).data


def open_controlled(gate: Gate, n_ctrl: int) -> Gate:
    """``gate`` controlled on ``n_ctrl`` qubits all being |0> (controls come first)."""
    if n_ctrl == 0:
        return gate
    return gate.control(n_ctrl, ctrl_state="0" * n_ctrl)


def closed_controlled(gate: Gate, n_ctrl: int) -> Gate:
    if n_ctrl == 0:
        return gate
    return gate.control(n_ctrl, ctrl_state="1" * n_ctrl)


# ---- move unitaries -------------------------------------------------------------------------
def jump_ops(s, t) -> list[Op]:
    """U_jump (eq. JumpUnitary) on |t, s>."""
    return [(ISWAP, [s, t])]


def split_jump_ops(s, t1, t2) -> list[Op]:
    """U_split (eq. SplitJumpUnitary) on |t2, t1, s>: sqrt-iSWAP(s,t1) then iSWAP(s,t2)."""
    return [(SQRT_ISWAP, [s, t1]), (ISWAP, [s, t2])]


def merge_jump_ops(s1, s2, t) -> list[Op]:
    """U_merge (eq. MergeJumpUnitary) on |s1, s2, t>: the Hermitian conjugate of U_split."""
    return [(ISWAP_DG, [s1, t]), (SQRT_ISWAP_DG, [s2, t])]


def slide_ops(s, t, path: list) -> list[Op]:
    """U_slide (eq. SlideUnitary): iSWAP(s,t) open-controlled on every path square."""
    if not path:
        return jump_ops(s, t)
    return [(open_controlled(ISWAP, len(path)), [*path, s, t])]


def capture_jump_ops(s, t, c) -> list[Op]:
    """U_m = U_jump(s,t) U_jump(t,c) for the Capture Jump (Section "Capture Jump")."""
    return [(ISWAP, [t, c]), (ISWAP, [s, t])]


def capture_slide_ops(s, t, c, path: list) -> list[Op]:
    """U_m = U_slide(s,t) U_slide(t,c) for the Capture Slide, both path-controlled."""
    if not path:
        return capture_jump_ops(s, t, c)
    g = open_controlled(ISWAP, len(path))
    return [(g, [*path, t, c]), (g, [*path, s, t])]


def pawn_capture_ops(s, t, c, p) -> list[Op]:
    """Pawn Capture: ``p`` <- (s AND t) via a Toffoli, then jump(t,c) and jump(s,t) controlled on p.

    The paper sets p = |0> when both are occupied and uses zero-controls; here p = |1> when both
    are occupied and the controls are closed.  The action on the board qubits is identical.
    """
    return [
        (closed_controlled(X, 2), [s, t, p]),
        (closed_controlled(ISWAP, 1), [p, t, c]),
        (closed_controlled(ISWAP, 1), [p, s, t]),
    ]


# ---- Cantwell's Split Slide / Merge Slide (B, R, Q; board option ``slider_splits``) ---------------
# A *plan* is a list of steps ``(gate, zeros, ones, [a, b])``: the two-qubit ``gate`` acts on
# ``[a, b]`` (Qiskit order) in the branches where every qubit of ``zeros`` is |0> and, if ``ones`` is
# non-empty, at least one qubit of ``ones`` is |1>; in every other branch the step is the identity.
# The conditions of the steps of one plan select disjoint path blocks except the both-clear block,
# whose two steps are applied in order, so a plan is exactly the block-diagonal unitary of the paper.
_INVERSE = {}


def _inverse(gate: Gate) -> Gate:
    if not _INVERSE:
        _INVERSE.update({id(ISWAP): ISWAP_DG, id(ISWAP_DG): ISWAP,
                         id(SQRT_ISWAP): SQRT_ISWAP_DG, id(SQRT_ISWAP_DG): SQRT_ISWAP})
    return _INVERSE[id(gate)]


def split_slide_plan(s, t1, t2, path1: list, path2: list) -> list:
    """U_split_slide (eq. SplitSlideUnitary of arXiv:1906.05836, basis |p2, p1, t2, t1, s>).

    ``path1`` / ``path2``: the control squares of the arms s -> t1 and s -> t2 (path squares that can
    be occupied; as in quantumlib/unitary ``quantum_board.py``, t2 is left out of arm 1 and t1 of arm
    2).  Arm i is *blocked* when one of its control squares is occupied.  Blocks:
    both arms clear: U_split = iSWAP(s, t2) sqrt-iSWAP(s, t1) (the Split Jump unitary);
    arm 1 clear, arm 2 blocked: iSWAP(s, t1); arm 2 clear, arm 1 blocked: iSWAP(s, t2);
    both blocked: identity ("full iSwaps in the event that a single path is blocked")."""
    c1, c2 = list(path1), list(path2)
    both = sorted(set(c1) | set(c2))
    steps = [(SQRT_ISWAP, both, [], [s, t1]), (ISWAP, both, [], [s, t2])]
    only2 = [q for q in c2 if q not in c1]
    only1 = [q for q in c1 if q not in c2]
    if only2:      # arm 1 clear and arm 2 blocked is possible
        steps.append((ISWAP, c1, only2, [s, t1]))
    if only1:
        steps.append((ISWAP, c2, only1, [s, t2]))
    return steps


def merge_slide_plan(s1, s2, t, path1: list, path2: list) -> list:
    """U_merge_slide (eq. MergeSlideUnitary, basis |p2, p1, s1, s2, t>).

    ``path1`` / ``path2``: control squares of s1 -> t and s2 -> t (s2 left out of arm 1, s1 of arm 2).
    Blocks: both clear: U_merge = sqrt-iSWAP^dag(s2, t) iSWAP^dag(s1, t) (the Merge Jump unitary,
    the Hermitian conjugate of U_split); arm 1 blocked, arm 2 clear: iSWAP^dag(s2, t); arm 2 blocked,
    arm 1 clear: iSWAP^dag(s1, t); both blocked: identity."""
    c1, c2 = list(path1), list(path2)
    both = sorted(set(c1) | set(c2))
    steps = [(ISWAP_DG, both, [], [s1, t]), (SQRT_ISWAP_DG, both, [], [s2, t])]
    only2 = [q for q in c2 if q not in c1]
    only1 = [q for q in c1 if q not in c2]
    if only1:      # arm 2 clear and arm 1 blocked is possible
        steps.append((ISWAP_DG, c2, only1, [s2, t]))
    if only2:
        steps.append((ISWAP_DG, c1, only2, [s1, t]))
    return steps


def plan_ops(steps: list) -> list[Op]:
    """A plan as a plain op list of open-controlled gates (for circuits and matrix checks): a step
    with a non-empty ``ones`` set becomes ``gate`` open-controlled on ``zeros`` followed by its
    inverse open-controlled on ``zeros + ones`` (the two cancel where every square is clear)."""
    ops: list[Op] = []
    for gate, zeros, ones, ab in steps:
        ops.append((open_controlled(gate, len(zeros)), [*zeros, *ab]))
        if ones:
            ops.append((open_controlled(_inverse(gate), len(zeros) + len(ones)), [*zeros, *ones, *ab]))
    return ops


def split_slide_ops(s, t1, t2, path1: list, path2: list) -> list[Op]:
    return plan_ops(split_slide_plan(s, t1, t2, path1, path2))


def merge_slide_ops(s1, s2, t, path1: list, path2: list) -> list[Op]:
    return plan_ops(merge_slide_plan(s1, s2, t, path1, path2))


def apply_plan(state, steps: list) -> None:
    """Apply a plan to a ``SparseState`` (the 4x4 matrices of ``matrix(gate)``, restricted per step)."""
    for gate, zeros, ones, ab in steps:
        zm = 0
        for q in zeros:
            zm |= 1 << q
        om = 0
        for q in ones:
            om |= 1 << q
        state.apply_restricted(matrix(gate), ab, zm, om)


def ops_unitary(ops: list[Op], roles: list) -> np.ndarray:
    """Dense unitary of an op list on the given role ordering (``roles[0]`` least significant)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(len(roles))
    pos = {r: i for i, r in enumerate(roles)}
    for gate, qs in ops:
        qc.append(gate, [pos[q] for q in qs])
    return Operator(qc).data
