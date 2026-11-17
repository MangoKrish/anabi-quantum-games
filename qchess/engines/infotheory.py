"""Information-theoretic quantities of a Cantwell quantum-chess position (added 20 October 2026).

Everything here is read off the exact sparse state (``board.state.amps``: one qubit per square plus
the capture ancillas, a pure state) and the classical type vector; nothing is sampled.  Entropies are
in bits.  Used by the evaluator's ``qfeat`` terms ``king_entropy``, ``king_variance``,
``entanglement`` and ``impurity`` (``qchess.engines.evaluators``), by the MCTS ``prior_info`` option
and by the MCTS ``time_manager`` (``qchess.engines.mcts``).

* ``king_distribution(board, colour)`` -> ``(mu, absent)``: ``mu[q] = P(colour's king on q)`` over
  the squares typed as that king, ``absent = 1 - sum(mu)`` (the king's captured mass).
* ``king_entropy``: Shannon entropy of that location variable, squares plus "absent":
  ``H = -sum_q mu_q log2 mu_q - a log2 a``.  0 for a classical king; 1 bit for an even split (or for a
  king captured with probability 1/2).
* ``king_variance``: ``sum_q mu_q (1 - mu_q)``, the summed Bernoulli variance of the king's square
  indicators (0 classical, 1/2 for an even split): the linear-entropy cousin of ``king_entropy``.
* ``branch_entropy``: Shannon entropy of the Born weights of the distinct classical boards.
* ``schmidt_spectrum(amps, mask)``: eigenvalues of ``rho_A = Tr_B |psi><psi|`` for the qubits in
  ``mask`` (A) against all others (B, ancillas included), from the singular values of the amplitude
  matrix ``M[a, b]``.  Since the global state is pure, ``S(rho_A) = S(rho_B)``.
* ``colour_cut_entropy(board)``: von Neumann entropy of the reduced state of one colour's occupancy
  qubits; the average of the white cut (white squares | everything else) and the black cut, which
  differ only through entanglement with the ancillas.  Colour-symmetric.  0 for any product state,
  in particular whenever only one side has pieces in superposition (a split king alone is not
  entangled with anything).
* ``colour_cut_purity(board)``: ``Tr rho_A^2`` for the same cuts (1 = product state).
* ``mutual_information(board)``: ``I(W:B) = S(W) + S(B) - S(W u B)`` of the board squares with the
  ancillas traced out (``S(W u B) = S(ancillas)`` by purity): all correlations, quantum and
  classical, between the two colours' occupancies.  Diagnostic only.
* ``binary_entropy(p)``: ``h(p)`` in bits, the Shannon information a two-outcome measurement with
  ``P(1) = p`` reveals.

Cost: the cut entropies diagonalise a matrix whose side is at most the number of non-zero amplitudes;
above ``cap`` amplitudes (default 256) they are not computed and ``None`` is returned (callers then use
0 and count it).
"""
from __future__ import annotations

import math

import numpy as np

from ..geometry import WHITE

LOG2 = math.log(2.0)
EIG_EPS = 1e-14
DEFAULT_CAP = 256


def _h_terms(ps) -> float:
    tot = 0.0
    for p in ps:
        if p > EIG_EPS:
            tot -= p * math.log(p)
    return tot / LOG2


def binary_entropy(p: float) -> float:
    """``h(p) = -p log2 p - (1 - p) log2 (1 - p)`` (0 at p = 0 or 1)."""
    p = min(max(float(p), 0.0), 1.0)
    return _h_terms((p, 1.0 - p))


def square_probs(board) -> list[float]:
    """``P(square occupied)`` for every board square (ancillas ignored)."""
    nsq = board.nsq
    pr = [0.0] * nsq
    for b, a in board.state.amps.items():
        w = float(a.real * a.real + a.imag * a.imag)
        q = 0
        while b and q < nsq:
            if b & 1:
                pr[q] += w
            b >>= 1
            q += 1
    return pr


def king_distribution(board, colour: str, probs: list[float] | None = None) -> tuple[list[float], float]:
    """``([P(king on q) for q typed as colour's king], P(king absent))``."""
    letter = "K" if colour == WHITE else "k"
    pr = probs if probs is not None else square_probs(board)
    mu = [pr[q] for q, t in enumerate(board.types) if t == letter and pr[q] > 0.0]
    absent = max(0.0, 1.0 - sum(mu))
    return mu, absent


def king_entropy(board, colour: str, probs: list[float] | None = None) -> float:
    """Shannon entropy (bits) of the king's location, squares plus "absent"."""
    mu, absent = king_distribution(board, colour, probs)
    return _h_terms(mu + [absent])


def king_variance(board, colour: str, probs: list[float] | None = None) -> float:
    """``sum_q mu_q (1 - mu_q)`` over the king's squares."""
    mu, _ = king_distribution(board, colour, probs)
    return float(sum(p * (1.0 - p) for p in mu))


def branch_weights(board) -> list[float]:
    """Born weights of the distinct classical boards (ancilla bits stripped, equal boards merged)."""
    mask = (1 << board.nsq) - 1
    acc: dict[int, float] = {}
    for b, a in board.state.amps.items():
        k = b & mask
        acc[k] = acc.get(k, 0.0) + float(a.real * a.real + a.imag * a.imag)
    return list(acc.values())


def branch_entropy(board) -> float:
    """Shannon entropy (bits) of the distribution over classical boards."""
    ws = branch_weights(board)
    tot = sum(ws)
    if tot <= 0 or len(ws) < 2:
        return 0.0
    return _h_terms([w / tot for w in ws])


def schmidt_spectrum(amps: dict, mask: int, cap: int | None = DEFAULT_CAP) -> np.ndarray | None:
    """Eigenvalues (summing to 1) of the reduced state of the qubits in ``mask``; ``None`` if the
    state has more than ``cap`` non-zero amplitudes."""
    if cap is not None and len(amps) > cap:
        return None
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    ent = []
    for b, a in amps.items():
        if a == 0:
            continue
        ra, cb = b & mask, b & ~mask
        i = rows.setdefault(ra, len(rows))
        j = cols.setdefault(cb, len(cols))
        ent.append((i, j, a))
    if len(rows) <= 1 or len(cols) <= 1:
        return np.ones(1)
    M = np.zeros((len(rows), len(cols)), dtype=complex)
    for i, j, a in ent:
        M[i, j] = a
    if M.shape[0] <= M.shape[1]:
        rho = M @ M.conj().T
    else:
        rho = M.conj().T @ M
    lam = np.linalg.eigvalsh(rho).real
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    if s <= 0:
        return np.ones(1)
    return lam / s


def vn_entropy(lam: np.ndarray) -> float:
    """``-sum lam log2 lam`` (bits)."""
    return _h_terms(float(x) for x in lam)


def purity(lam: np.ndarray) -> float:
    return float(np.sum(lam * lam))


def colour_mask(board, colour: str) -> int:
    """Bit mask of the squares typed with ``colour``'s pieces."""
    up = colour == WHITE
    m = 0
    for q, t in enumerate(board.types):
        if t is not None and t.isupper() == up:
            m |= 1 << q
    return m


def colour_cut_spectra(board, cap: int | None = DEFAULT_CAP):
    """``(spectrum of the white cut, spectrum of the black cut)``, or ``None`` above ``cap``."""
    amps = board.state.amps
    if len(amps) <= 1:
        one = np.ones(1)
        return one, one
    sw = schmidt_spectrum(amps, colour_mask(board, WHITE), cap)
    if sw is None:
        return None
    sb = schmidt_spectrum(amps, colour_mask(board, "b"), cap)
    return sw, sb


def colour_cut_entropy(board, cap: int | None = DEFAULT_CAP) -> float | None:
    """Average von Neumann entropy (bits) of the white and the black cut (``None`` above ``cap``)."""
    sp = colour_cut_spectra(board, cap)
    if sp is None:
        return None
    return 0.5 * (vn_entropy(sp[0]) + vn_entropy(sp[1]))


def colour_cut_purity(board, cap: int | None = DEFAULT_CAP) -> float | None:
    """Average ``Tr rho^2`` of the white and the black cut (``None`` above ``cap``)."""
    sp = colour_cut_spectra(board, cap)
    if sp is None:
        return None
    return 0.5 * (purity(sp[0]) + purity(sp[1]))


def mutual_information(board, cap: int | None = DEFAULT_CAP) -> float | None:
    """``I(W:B)`` of the board squares (ancillas traced out), bits; ``None`` above ``cap``."""
    amps = board.state.amps
    if len(amps) <= 1:
        return 0.0
    if cap is not None and len(amps) > cap:
        return None
    mw, mb = colour_mask(board, WHITE), colour_mask(board, "b")
    sw = vn_entropy(schmidt_spectrum(amps, mw, None))
    sb = vn_entropy(schmidt_spectrum(amps, mb, None))
    swb = vn_entropy(schmidt_spectrum(amps, mw | mb, None))
    return max(0.0, sw + sb - swb)


def features(board, cap: int | None = DEFAULT_CAP) -> dict:
    """Every quantity above for one position (a diagnostic record)."""
    pr = square_probs(board)
    S = colour_cut_entropy(board, cap)
    P = colour_cut_purity(board, cap)
    return {
        "amps": len(board.state.amps),
        "branches": len(branch_weights(board)),
        "H_branches": branch_entropy(board),
        "H_king_w": king_entropy(board, WHITE, pr), "H_king_b": king_entropy(board, "b", pr),
        "V_king_w": king_variance(board, WHITE, pr), "V_king_b": king_variance(board, "b", pr),
        "S_cut": S, "purity_cut": P, "I_WB": mutual_information(board, cap),
    }


__all__ = ["binary_entropy", "square_probs", "king_distribution", "king_entropy", "king_variance",
           "branch_weights", "branch_entropy", "schmidt_spectrum", "vn_entropy", "purity", "colour_mask",
           "colour_cut_spectra", "colour_cut_entropy", "colour_cut_purity", "mutual_information", "features",
           "DEFAULT_CAP"]
