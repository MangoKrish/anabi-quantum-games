"""Anabi: the champion hybrid engine of the study, under its own name.

Anabi is exactly the configuration that played as ``F-MC-signbest`` in the finals, the ablations and the
Deep Dirac match (``reports/engines/tournament/final_vsB1tm_configs.json``): Monte Carlo tree search with
PUCT, exact Born averaging at measurement nodes, a one-ply minimax step at the leaves, Fairy-Stockfish
depth-8 evaluation of every branch, quantum moves on, and the adversarial (minimax) opponent model, at
0.5 s per move.  Named on 24 October 2026; only the display name differs from the tournament config.

    from qchess.engines.anabi import make_anabi
    eng = make_anabi()
"""
from __future__ import annotations

from . import make_engine
from .base import Budget, Engine

ANABI: dict = {
    "family": "mcts",
    "select": "puct",
    "c": 1.0,
    "chance": "expect",
    "leaf": "minimax1",
    "eval": {"kind": "fsf", "depth": 8, "exact_max": 16, "samples": 8},
    "name": "Anabi",
}

#: thinking time per move used in every reported match
ANABI_SECONDS = 0.5


def make_anabi(**overrides) -> Engine:
    """A fresh Anabi engine; ``overrides`` replace top-level config keys (e.g. ``name``)."""
    return make_engine({**ANABI, **overrides})


def anabi_budget(seconds: float = ANABI_SECONDS) -> Budget:
    return Budget(seconds=seconds)


__all__ = ["ANABI", "ANABI_SECONDS", "make_anabi", "anabi_budget"]
