"""A sparse pure-state simulator that applies Qiskit operator matrices to subsets of qubits.

The occupancy superposition of a quantum chess board is tiny compared with 2^(n*n) (Cantwell,
arXiv:1906.05836, Section "Classical Simulation"), so the state is kept as a dict
``{basis_int: amplitude}``.  Basis integers use Qiskit's little-endian convention: bit ``q`` of
the integer is qubit ``q``.  ``apply(mat, qubits)`` interprets ``mat`` exactly like
``qiskit.quantum_info.Operator(gate).data`` with ``qubits[0]`` as the least significant
sub-index bit, so any Qiskit gate (including ``gate.control(k, ctrl_state=...)``) can be applied
directly and cross-checked against ``Statevector.evolve`` (see ``tests/test_sparse_vs_qiskit.py``).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

AMP_EPS = 1e-12
Pred = Callable[[int], bool]


class SparseState:
    __slots__ = ("nq", "amps")

    def __init__(self, nq: int, amps: dict[int, complex] | None = None):
        self.nq = nq
        self.amps: dict[int, complex] = dict(amps) if amps else {0: 1.0 + 0j}

    # ---- construction / bookkeeping --------------------------------------------------
    @classmethod
    def from_bits(cls, bits: int, nq: int) -> "SparseState":
        return cls(nq, {bits: 1.0 + 0j})

    def copy(self) -> "SparseState":
        return SparseState(self.nq, self.amps)

    def add_qubit(self) -> int:
        """Append a fresh ancilla in |0> and return its index."""
        q = self.nq
        self.nq += 1
        return q

    def drop_qubit_if_zero(self, q: int) -> bool:
        """Remove the (highest-index) ancilla ``q`` if it is |0> in every branch."""
        if q != self.nq - 1:
            raise ValueError("only the most recently added qubit can be dropped")
        if any((b >> q) & 1 for b in self.amps):
            return False
        self.nq -= 1
        return True

    def support_size(self) -> int:
        return len(self.amps)

    # ---- linear algebra ---------------------------------------------------------------
    def apply(self, mat: np.ndarray, qubits: list[int]) -> None:
        """Apply a 2^k x 2^k matrix to ``qubits`` (``qubits[0]`` = least significant)."""
        k = len(qubits)
        if mat.shape != (1 << k, 1 << k):
            raise ValueError(f"matrix shape {mat.shape} does not match {k} qubits")
        masks = [1 << q for q in qubits]
        all_mask = 0
        for m in masks:
            all_mask |= m
        nz_cols = {}
        new: dict[int, complex] = {}
        for b, a in self.amps.items():
            i = 0
            for j, m in enumerate(masks):
                if b & m:
                    i |= 1 << j
            base = b & ~all_mask
            if i not in nz_cols:
                col = mat[:, i]
                nz_cols[i] = [(jj, col[jj]) for jj in np.nonzero(np.abs(col) > AMP_EPS)[0]]
            for jj, v in nz_cols[i]:
                nb = base
                for j, m in enumerate(masks):
                    if (jj >> j) & 1:
                        nb |= m
                new[nb] = new.get(nb, 0j) + v * a
        self.amps = {b: a for b, a in new.items() if abs(a) > AMP_EPS}

    def apply_restricted(self, mat: np.ndarray, qubits: list[int], zeros: int, ones: int) -> None:
        """``apply(mat, qubits)`` on the branches with ``b & zeros == 0`` and (if ``ones``)
        ``b & ones != 0``; every other branch is left as it is (the matrix of ``mat`` controlled on
        those conditions, which never touch ``qubits``).  Used by the Split / Merge Slide."""
        k = len(qubits)
        if mat.shape != (1 << k, 1 << k):
            raise ValueError(f"matrix shape {mat.shape} does not match {k} qubits")
        masks = [1 << q for q in qubits]
        all_mask = 0
        for m in masks:
            all_mask |= m
        if all_mask & (zeros | ones):
            raise ValueError("a condition qubit is also a target")
        nz_cols = {}
        new: dict[int, complex] = {}
        for b, a in self.amps.items():
            if (b & zeros) or (ones and not (b & ones)):
                new[b] = new.get(b, 0j) + a
                continue
            i = 0
            for j, m in enumerate(masks):
                if b & m:
                    i |= 1 << j
            base = b & ~all_mask
            if i not in nz_cols:
                col = mat[:, i]
                nz_cols[i] = [(jj, col[jj]) for jj in np.nonzero(np.abs(col) > AMP_EPS)[0]]
            for jj, v in nz_cols[i]:
                nb = base
                for j, m in enumerate(masks):
                    if (jj >> j) & 1:
                        nb |= m
                new[nb] = new.get(nb, 0j) + v * a
        self.amps = {b: a for b, a in new.items() if abs(a) > AMP_EPS}

    def prob(self, pred: Pred) -> float:
        return float(sum(abs(a) ** 2 for b, a in self.amps.items() if pred(b)))

    def bit_prob(self, q: int) -> float:
        m = 1 << q
        return float(sum(abs(a) ** 2 for b, a in self.amps.items() if b & m))

    def project(self, pred: Pred, outcome: int) -> float:
        """Project onto ``pred == outcome`` and renormalise.  Returns the outcome probability."""
        keep = {b: a for b, a in self.amps.items() if bool(pred(b)) == bool(outcome)}
        p = sum(abs(a) ** 2 for a in keep.values())
        if p <= AMP_EPS:
            raise ValueError("projecting onto a zero-probability outcome")
        norm = np.sqrt(p)
        self.amps = {b: a / norm for b, a in keep.items()}
        return float(p)

    def measure(self, pred: Pred, rng: np.random.Generator) -> int:
        """Two-outcome projective measurement M1 = sum_{pred(b)} |b><b|; collapses the state."""
        p1 = self.prob(pred)
        outcome = 1 if rng.random() < p1 else 0
        self.project(pred, outcome)
        return outcome

    def sample(self, rng: np.random.Generator, k: int) -> list[int]:
        basis = list(self.amps)
        probs = np.array([abs(self.amps[b]) ** 2 for b in basis])
        probs /= probs.sum()
        picks = rng.choice(len(basis), size=k, p=probs)
        return [basis[i] for i in picks]

    def norm(self) -> float:
        return float(np.sqrt(sum(abs(a) ** 2 for a in self.amps.values())))

    def to_vector(self) -> np.ndarray:
        vec = np.zeros(1 << self.nq, dtype=complex)
        for b, a in self.amps.items():
            vec[b] = a
        return vec

    def close_to(self, other: "SparseState", tol: float = 1e-9) -> bool:
        if self.nq != other.nq:
            return False
        keys = set(self.amps) | set(other.amps)
        return all(abs(self.amps.get(b, 0) - other.amps.get(b, 0)) <= tol for b in keys)

    def __repr__(self) -> str:
        terms = []
        for b in sorted(self.amps):
            a = self.amps[b]
            terms.append(f"({a.real:+.4f}{a.imag:+.4f}j)|{b:0{self.nq}b}>")
        return " + ".join(terms)
