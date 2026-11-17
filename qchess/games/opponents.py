"""Baseline opponents for the game ports (any board following ``core.GameBoard``) and the game factories.

* ``RandomAgent``   - a uniformly random legal move (its own seeded stream).
* ``GreedyAgent``   - one ply: the move with the best exact Born-averaged evaluator value (Anabi's prior argmax,
                      no search).  Takes an immediate win when there is one.
* ``classical_only`` - Anabi with ``quantum: False`` (it never splits / merges / entangles itself; it still
                      models the opponent's quantum moves): the "classical player" of the study.
* ``Perfect``       - exact optimum, TiqTaqToe bounded rule sets only (``tqt_solver``).

``make_game(spec)`` builds a fresh start board and the matching Anabi evaluator from a spec string:
``tqt:q1``, ``tqt:q2``, ``tqt:ql3``, ``tqt:t2``..., ``chq:5x5r1:L3`` (paper rules), ``cqw:6x6r2:L2`` (tournament rules).
"""
from __future__ import annotations

import numpy as np

from .core import WHITE, BranchAveragedEvaluator, make_anabi_for


class RandomAgent:
    def __init__(self, name: str = "Random"):
        self.name = name
        self.rng = np.random.default_rng(0)

    def new_game(self, color, seed):
        self.rng = np.random.default_rng(seed)

    def choose(self, board, budget):
        ms = board.legal_moves()
        return ms[int(self.rng.integers(len(ms)))]

    def observe(self, *a):
        pass


class GreedyAgent:
    def __init__(self, evaluator, name: str = "Greedy"):
        self.ev = evaluator
        self.name = name
        self.rng = np.random.default_rng(0)

    def new_game(self, color, seed):
        self.rng = np.random.default_rng(seed)

    def choose(self, board, budget):
        me = board.turn
        best, bv = [], -9.0
        for m in board.legal_moves():
            v = 0.0
            for o, p in board.outcome_probs(m):
                c = board.apply(m, o)
                r = c.result()
                if r is not None:
                    x = 0.0 if r == "draw" else (1.0 if (r == "white") == (me == WHITE) else -1.0)
                else:
                    x = self.ev(c, me)
                v += p * x
            if v > bv + 1e-12:
                best, bv = [m], v
            elif abs(v - bv) <= 1e-12:
                best.append(m)
        return best[int(self.rng.integers(len(best)))]

    def observe(self, *a):
        pass


def parse_spec(spec: str) -> dict:
    game, rest = spec.split(":", 1)
    if game == "tqt":
        return {"game": "tqt", "rules": rest}
    if game in ("chq", "cqw"):
        size_rows, lvl = rest.split(":")
        size, rows = size_rows.split("r")
        return {"game": game, "size": int(size.split("x")[0]), "rows": int(rows), "level": int(lvl[1:])}
    raise ValueError(spec)


def make_board(spec: str):
    d = parse_spec(spec)
    if d["game"] == "tqt":
        from .tiqtaqtoe import TQTBoard
        return TQTBoard(d["rules"])
    if d["game"] == "cqw":
        from .cheqqers_web import CQWebBoard
        return CQWebBoard.new(d["size"], d["rows"], d["level"])
    from .cheqqers import CheqqersBoard
    return CheqqersBoard.new(d["size"], d["rows"], d["level"])


def make_evaluator(spec: str, **kw):
    d = parse_spec(spec)
    if d["game"] == "tqt":
        from .tiqtaqtoe import classical_branch_value
        return BranchAveragedEvaluator(classical_branch_value, name="tqt-perfect-classical", **kw)
    if d["game"] == "cqw":
        from .cheqqers_web import ClassicalCheckersWeb as CC
    else:
        from .cheqqers import ClassicalCheckers as CC
    cc = CC(d["size"], depth=kw.pop("depth", 2))
    ev = BranchAveragedEvaluator(cc.branch_value, name=f"checkers-ab{cc.depth}", **kw)
    ev.engine = cc
    return ev


def make_anabi(spec: str, name: str = "Anabi", **overrides):
    return make_anabi_for(make_evaluator(spec), name=name, **overrides)


def make_agent(spec: str, kind: str):
    """``kind``: anabi | anabi-classical | greedy | random | perfect."""
    if kind == "anabi":
        return make_anabi(spec)
    if kind == "anabi-classical":
        return make_anabi(spec, name="Anabi-classical", quantum=False)
    if kind == "greedy":
        return GreedyAgent(make_evaluator(spec))
    if kind == "random":
        return RandomAgent()
    if kind == "perfect":
        from .tqt_solver import Perfect
        return Perfect(parse_spec(spec)["rules"])
    raise ValueError(kind)


__all__ = ["RandomAgent", "GreedyAgent", "make_board", "make_evaluator", "make_anabi", "make_agent", "parse_spec"]
