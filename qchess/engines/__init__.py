"""Hybrid quantum-chess engines: search x evaluation x game theory x Bayesian layers.

Build any engine from a JSON-able config::

    from qchess.engines import make_engine
    eng = make_engine({"family": "expectimax", "eval": {"kind": "fsf", "samples": 8},
                       "decide": {"rule": "argmax"}, "opponent": {"model": "minimax"}})

Config keys shared by the search families (each module documents its own extras):

* ``family``    - "random" | "greedy" | "expectimax" | "mcts" | "bayes" (wrapper) | "fsf" (baseline)
                  | "oracle" (endgame-only tablebase players)
                  | "deepdirac" (port of iamthoufic/quantum-chess's Deep Dirac bot, MIT; see its module)
* ``name``      - display name (defaults to a canonical string of the config)
* ``eval``      - evaluator config, see ``qchess.engines.evaluators``
* ``decide``    - root decision rule, see ``qchess.engines.gametheory``
* ``opponent``  - opponent model at MIN nodes, see ``qchess.engines.gametheory``
* ``quantum``   - bool, allow split/merge moves (default True)
* ``bayes``     - Bayesian layer config, see ``qchess.engines.bayes``
"""
from __future__ import annotations

import importlib
import json

from .base import Budget, Candidate, Engine

_FAMILIES = {
    "random": ("qchess.engines.simple", "RandomEngine"),
    "greedy": ("qchess.engines.simple", "GreedyEngine"),
    "expectimax": ("qchess.engines.expectimax", "ExpectimaxEngine"),
    "mcts": ("qchess.engines.mcts", "MCTSEngine"),
    "bayes": ("qchess.engines.bayes", "BayesAdaptiveEngine"),
    "trap": ("qchess.engines.traps", "TrapEngine"),
    "fsf": ("qchess.engines.baselines", "FSFSamplingEngine"),
    "oracle": ("qchess.engines.baselines", "OracleEngine"),
    "deepdirac": ("qchess.engines.deepdirac", "DeepDiracEngine"),
}


def canonical_name(cfg: dict) -> str:
    if cfg.get("name"):
        return cfg["name"]
    return json.dumps({k: v for k, v in cfg.items() if k != "name"}, sort_keys=True, separators=(",", ":"))


def make_engine(cfg: dict) -> Engine:
    fam = cfg.get("family")
    if fam not in _FAMILIES:
        raise ValueError(f"unknown engine family {fam!r}")
    mod, cls = _FAMILIES[fam]
    eng = getattr(importlib.import_module(mod), cls)(cfg)
    eng.name = canonical_name(cfg)
    return eng


__all__ = ["Budget", "Candidate", "Engine", "make_engine", "canonical_name"]
