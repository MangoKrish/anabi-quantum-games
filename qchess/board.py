"""Quantum minichess board: Cantwell's rules (arXiv:1906.05836) on an N x N board.

State = (occupancy superposition over one qubit per square plus ancillas, classical piece-type
vector ``types``, side to move).  Rules implemented:

* Gardner minichess pieces (K Q R B N P); no castling, no double pawn step, no en passant,
  promotion to queen on the last rank.
* Jump unitary for N and K (Standard / Blocked / Capture Jump), path-controlled slide unitary
  for B, R, Q (Standard / Blocked / Capture Slide), Split Jump and Merge Jump for N and K only,
  Pawn Step / Blocked Pawn Step / Pawn Capture.
* Blocked and capture moves are *measuring moves* (Algorithm "Execute Measuring Move"): the
  two-outcome measurement M1 of the paper is applied first and the unitary only on outcome 1.
  Captured pieces go to a fresh "captured" ancilla, exactly as in the paper, so the global
  state stays pure.
* Option ``slider_splits=True`` (default False): Cantwell's Split Slide and Merge Slide for B, R, Q
  as well (eqs. SplitSlideUnitary / MergeSlideUnitary: path-controlled blocks, full iSWAP when one
  arm is blocked), as in full Cantwell chess and quantumlib/unitary.  With the default every
  behaviour is exactly the K/N-only reduction above.
* No check / checkmate.  The game ends when one side's king has probability 0 of being on the
  board (that side loses); both at 0 is a draw.
* A move is legal iff its possibility equation holds and it changes the state (Definition
  "Legal Move"); for a measuring move this means the outcome-1 probability is positive and,
  if it is 1, that the unitary acts non-trivially.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gates
from .geometry import (BLACK, JUMPERS, SLIDERS, START_FEN, WHITE, Geometry, color_of, kind,
                       other, parse_fen, to_fen)
from .moves import MEASURING, Move, MoveType, SliderMoveType
from .sparse import SparseState

EPS = 1e-10


class IllegalMove(ValueError):
    pass


@dataclass
class MoveResult:
    move: Move
    move_type: MoveType
    outcome: int | None          # None for non-measuring moves
    p_outcome1: float | None     # probability the measurement gave 1 (before it was taken)
    text: str


@dataclass
class Sample:
    """One classical board drawn from the superposition."""
    bits: int                    # occupancy of board squares (ancilla bits stripped)
    types: list[str | None]      # types restricted to occupied squares
    turn: str
    fen: str


SLIDER_QUANTUM = (SliderMoveType.SPLIT_SLIDE, SliderMoveType.MERGE_SLIDE)


class QuantumBoard:
    slider_splits = False      # class default (boards built with __new__ inherit the K/N-only rules)

    def __init__(self, n: int = 5, fen: str | None = None, seed: int | None = 0, slider_splits: bool = False):
        if slider_splits:
            self.slider_splits = True
        self.n = n
        self.geo = Geometry(n)
        self.types, self.turn = parse_fen(fen or START_FEN[n], n)
        bits = 0
        for i, p in enumerate(self.types):
            if p is not None:
                bits |= 1 << i
        self.state = SparseState.from_bits(bits, n * n)
        self.rng = np.random.default_rng(seed)
        self.history: list[MoveResult] = []
        self.captured: list[int] = []          # captured-piece ancillas
        self.ancilla_notes: dict[int, str] = {}
        self.initial_pieces = bin(bits).count("1")

    # ---- inspection --------------------------------------------------------------------
    @property
    def nsq(self) -> int:
        return self.n * self.n

    def copy(self) -> "QuantumBoard":
        b = QuantumBoard.__new__(QuantumBoard)
        b.n, b.geo = self.n, self.geo
        b.types, b.turn = list(self.types), self.turn
        b.state = self.state.copy()
        b.rng = np.random.default_rng(self.rng.integers(0, 2**32))
        b.history = list(self.history)
        b.captured = list(self.captured)
        b.ancilla_notes = dict(self.ancilla_notes)
        b.initial_pieces = self.initial_pieces
        if self.slider_splits:
            b.slider_splits = True
        return b

    def prob(self, square: int) -> float:
        return self.state.bit_prob(square)

    def probabilities(self) -> list[float]:
        return [self.prob(q) for q in self.geo.squares()]

    def king_prob(self, color: str) -> float:
        k = "K" if color == WHITE else "k"
        squares = [q for q in self.geo.squares() if self.types[q] == k]
        if not squares:
            return 0.0
        mask = 0
        for q in squares:
            mask |= 1 << q
        return 1.0 - self.state.prob(lambda b: (b & mask) == 0)

    def result(self) -> str | None:
        wk, bk = self.king_prob(WHITE), self.king_prob(BLACK)
        if wk <= EPS and bk <= EPS:
            return "draw"
        if wk <= EPS:
            return "black"
        if bk <= EPS:
            return "white"
        return None

    def piece_counts_per_branch(self) -> set[int]:
        """Number of pieces (board + captured ancillas) in each branch; should be one value."""
        mask = (1 << self.nsq) - 1
        for c in self.captured:
            mask |= 1 << c
        return {bin(b & mask).count("1") for b in self.state.amps}

    def check_invariants(self) -> None:
        if abs(self.state.norm() - 1) > 1e-8:
            raise AssertionError(f"norm {self.state.norm()}")
        for q in self.geo.squares():
            p = self.prob(q)
            if self.types[q] is None and p > EPS:
                raise AssertionError(f"{self.geo.name(q)} occupied (p={p}) but has no type")
            if self.types[q] is not None and p <= EPS:
                raise AssertionError(f"{self.geo.name(q)} typed {self.types[q]} but empty")
        counts = self.piece_counts_per_branch()
        if len(counts) != 1 or counts != {self.initial_pieces}:
            raise AssertionError(f"piece number not conserved: {counts} vs {self.initial_pieces}")

    def __str__(self) -> str:
        lines = []
        for y in range(self.n - 1, -1, -1):
            cells = []
            for x in range(self.n):
                q = self.geo.idx(x, y)
                p = self.types[q]
                pr = self.prob(q)
                cells.append(f"{p if p else '.'}{int(round(pr * 100)):3d}" if p else "  . ")
            lines.append(f"{y + 1} " + " ".join(cells))
        lines.append("  " + "    ".join(f" {c}" for c in "abcdefghij"[: self.n]))
        lines.append(f"{'white' if self.turn == WHITE else 'black'} to move; "
                     f"support={self.state.support_size()} branches; "
                     f"K={self.king_prob(WHITE):.3f} k={self.king_prob(BLACK):.3f}")
        return "\n".join(lines)

    # ---- possibility equations (Section "Quantum Chess Moves") --------------------------
    def classify(self, m: Move) -> MoveType:
        v, geo, me = self.types, self.geo, self.turn
        s, t = m.source, m.target
        vs = v[s]
        if vs is None or color_of(vs) != me:
            raise IllegalMove("source is not the mover's piece")
        k = kind(vs)

        def same_or_empty(q: int) -> bool:
            return v[q] is None or v[q] == vs

        slider_q = self.slider_splits and k in SLIDERS
        if m.is_split:
            t2 = m.target2
            if k not in JUMPERS and not slider_q:
                raise IllegalMove("only knights and kings may split")
            if t == t2 or not geo.valid(t, s, vs) or not geo.valid(t2, s, vs):
                raise IllegalMove("invalid split targets")
            if not (same_or_empty(t) and same_or_empty(t2)):
                raise IllegalMove("split targets must be empty or hold the same piece")
            return SliderMoveType.SPLIT_SLIDE if slider_q else MoveType.SPLIT_JUMP   # eq. SplitSlidePossibility
        if m.is_merge:
            s2 = m.source2
            if k not in JUMPERS and not slider_q:
                raise IllegalMove("only knights and kings may merge")
            if s == s2 or v[s2] != vs:
                raise IllegalMove("merge sources must hold the same piece")
            if not geo.valid(t, s, vs) or not geo.valid(t, s2, vs):
                raise IllegalMove("invalid merge target")
            if not same_or_empty(t):
                raise IllegalMove("merge target must be empty or hold the same piece")
            return SliderMoveType.MERGE_SLIDE if slider_q else MoveType.MERGE_JUMP   # eq. MergeSlidePossibility

        vt = v[t]
        own = vt is not None and color_of(vt) == me
        opp = vt is not None and color_of(vt) != me
        if k == "P":
            if t == geo.pawn_step(s, me):
                return MoveType.PAWN_STEP if same_or_empty(t) else MoveType.BLOCKED_PAWN_STEP
            if t in geo.pawn_captures(s, me) and opp:
                return MoveType.PAWN_CAPTURE
            raise IllegalMove("invalid pawn move")
        if not geo.valid(t, s, vs):
            raise IllegalMove("invalid movement pattern")
        if k in JUMPERS:
            if same_or_empty(t):
                return MoveType.JUMP
            return MoveType.BLOCKED_JUMP if own else MoveType.CAPTURE_JUMP
        if k in SLIDERS:
            if same_or_empty(t):
                return MoveType.SLIDE
            return MoveType.BLOCKED_SLIDE if own else MoveType.CAPTURE_SLIDE
        raise IllegalMove(f"unknown piece {vs}")

    # ---- circuit pieces shared with qchess.circuits ----------------------------------------
    def path_controls(self, s: int, t: int) -> list[int]:
        """Path squares that can be occupied (classically empty squares need no control)."""
        return [q for q in self.geo.path(s, t) if self.types[q] is not None]

    def slider_arms(self, m: Move, mt: MoveType) -> tuple[list[int], list[int]]:
        """Control squares of the two arms of a Split / Merge Slide (as quantumlib/unitary's
        ``quantum_board.py``: the typed path squares, with the other target (split) or the other
        source (merge) left out)."""
        if mt is SliderMoveType.SPLIT_SLIDE:
            s, t1, t2 = m.source, m.target, m.target2
            return ([q for q in self.path_controls(s, t1) if q != t2],
                    [q for q in self.path_controls(s, t2) if q != t1])
        s1, s2, t = m.source, m.source2, m.target
        return ([q for q in self.path_controls(s1, t) if q != s2],
                [q for q in self.path_controls(s2, t) if q != s1])

    def slider_plan(self, m: Move, mt: MoveType) -> list:
        """The step plan (``gates.split_slide_plan`` / ``merge_slide_plan``) of a Split / Merge Slide."""
        c1, c2 = self.slider_arms(m, mt)
        if mt is SliderMoveType.SPLIT_SLIDE:
            return gates.split_slide_plan(m.source, m.target, m.target2, c1, c2)
        return gates.merge_slide_plan(m.source, m.source2, m.target, c1, c2)

    def path_classically_blocked(self, s: int, t: int) -> bool:
        return any(self.prob(q) >= 1 - EPS for q in self.geo.path(s, t))

    def m1_predicate(self, m: Move, mt: MoveType):
        """The paper's M1 projector for a measuring move, as a predicate on basis integers."""
        s, t = m.source, m.target
        if mt in (MoveType.BLOCKED_JUMP, MoveType.BLOCKED_SLIDE, MoveType.BLOCKED_PAWN_STEP):
            return lambda b: not (b >> t) & 1                      # eq. BlockedJumpMeasurement
        if mt in (MoveType.CAPTURE_JUMP, MoveType.PAWN_CAPTURE):
            return lambda b: bool((b >> s) & 1)                    # eq. CaptureJumpMeasurement
        if mt is MoveType.CAPTURE_SLIDE:                           # eq. CaptureSlideM1
            pmask = 0
            for q in self.path_controls(s, t):
                pmask |= 1 << q

            def pred(b: int) -> bool:
                blocked = bool(b & pmask)
                return (not blocked and bool((b >> s) & 1)) or (blocked and not (b >> t) & 1)
            return pred
        raise ValueError(f"{mt} is not a measuring move")

    def unitary_ops(self, m: Move, mt: MoveType, c: int | None = None, p: int | None = None):
        """The op list U_m for a move (ancilla indices supplied by the caller)."""
        s, t = m.source, m.target
        if mt in (MoveType.JUMP, MoveType.BLOCKED_JUMP, MoveType.PAWN_STEP,
                  MoveType.BLOCKED_PAWN_STEP):
            return gates.jump_ops(s, t)
        if mt is MoveType.SPLIT_JUMP:
            return gates.split_jump_ops(s, t, m.target2)
        if mt is MoveType.MERGE_JUMP:
            return gates.merge_jump_ops(s, m.source2, t)
        if mt in (MoveType.SLIDE, MoveType.BLOCKED_SLIDE):
            return gates.slide_ops(s, t, self.path_controls(s, t))
        if mt is MoveType.CAPTURE_JUMP:
            return gates.capture_jump_ops(s, t, c)
        if mt is MoveType.CAPTURE_SLIDE:
            return gates.capture_slide_ops(s, t, c, self.path_controls(s, t))
        if mt is MoveType.PAWN_CAPTURE:
            return gates.pawn_capture_ops(s, t, c, p)
        if mt in SLIDER_QUANTUM:
            return gates.plan_ops(self.slider_plan(m, mt))
        raise ValueError(mt)

    @staticmethod
    def _ancillas_needed(mt: MoveType) -> int:
        return {MoveType.CAPTURE_JUMP: 1, MoveType.CAPTURE_SLIDE: 1, MoveType.PAWN_CAPTURE: 2}.get(mt, 0)

    def _unitary_changes_state(self, m: Move, mt: MoveType, state: SparseState) -> bool:
        if mt in SLIDER_QUANTUM:
            tmp = state.copy()
            gates.apply_plan(tmp, self.slider_plan(m, mt))
            return not tmp.close_to(state)
        tmp = state.copy()
        extra = [tmp.add_qubit() for _ in range(self._ancillas_needed(mt))]
        ops = self.unitary_ops(m, mt, *extra)
        for g, qs in ops:
            tmp.apply(gates.matrix(g), qs)
        ref = SparseState(tmp.nq, state.amps)
        return not tmp.close_to(ref)

    # ---- legality -------------------------------------------------------------------------
    def is_legal(self, m: Move) -> bool:
        try:
            mt = self.classify(m)
        except IllegalMove:
            return False
        return self._legal_given_type(m, mt)

    def _legal_given_type(self, m: Move, mt: MoveType) -> bool:
        if mt in (MoveType.SLIDE, MoveType.BLOCKED_SLIDE, MoveType.CAPTURE_SLIDE):
            if self.path_classically_blocked(m.source, m.target):
                return False
        if mt in MEASURING:
            p1 = self.state.prob(self.m1_predicate(m, mt))
            if p1 <= EPS:
                return False
            if p1 >= 1 - EPS:
                return self._unitary_changes_state(m, mt, self.state)
            return True
        return self._unitary_changes_state(m, mt, self.state)

    def legal_moves(self) -> list[Move]:
        v, geo, me = self.types, self.geo, self.turn
        out: list[Move] = []

        def consider(m: Move) -> None:
            try:
                mt = self.classify(m)
            except IllegalMove:
                return
            if self._legal_given_type(m, mt):
                out.append(m)

        for s in geo.squares():
            vs = v[s]
            if vs is None or color_of(vs) != me:
                continue
            k = kind(vs)
            if k in JUMPERS:
                targets = geo.jump_targets(s, vs)
                for t in targets:
                    consider(Move(s, t))
                for t1 in targets:
                    for t2 in targets:
                        if t1 != t2:
                            consider(Move(s, t1, target2=t2))
                for s2 in geo.squares():
                    if s2 != s and v[s2] == vs:
                        for t in set(targets) & set(geo.jump_targets(s2, vs)):
                            consider(Move(s, t, source2=s2))
            elif k in SLIDERS:
                rays = [t for t, _path in geo.ray_targets(s, vs)]
                for t in rays:
                    consider(Move(s, t))
                if self.slider_splits:
                    for t1 in rays:
                        for t2 in rays:
                            if t1 != t2:
                                consider(Move(s, t1, target2=t2))
                    for s2 in geo.squares():
                        if s2 != s and v[s2] == vs:
                            r2 = {t for t, _path in geo.ray_targets(s2, vs)}
                            for t in rays:
                                if t in r2:
                                    consider(Move(s, t, source2=s2))
            else:  # pawn
                t = geo.pawn_step(s, me)
                if t is not None:
                    consider(Move(s, t))
                for t in geo.pawn_captures(s, me):
                    consider(Move(s, t))
        return out

    # ---- playing --------------------------------------------------------------------------
    def push(self, m: Move | str) -> MoveResult:
        if isinstance(m, str):
            m = Move.parse(m, self.geo)
        mt = self.classify(m)
        if not self._legal_given_type(m, mt):
            raise IllegalMove(f"{m.text(self.geo)} ({mt.value}) has no effect")
        s, t = m.source, m.target
        vs = self.types[s]
        outcome: int | None = None
        p1: float | None = None
        c = p = None
        if mt in MEASURING:
            pred = self.m1_predicate(m, mt)
            p1 = self.state.prob(pred)
            outcome = self.state.measure(pred, self.rng)
        if outcome != 0:
            need = self._ancillas_needed(mt)
            if need:
                c = self.state.add_qubit()
                self.captured.append(c)
                self.ancilla_notes[c] = f"captured on {self.geo.name(t)} by {m.text(self.geo)}"
            if need == 2:
                p = self.state.add_qubit()
                self.ancilla_notes[p] = f"pawn-capture control for {m.text(self.geo)}"
            if mt in SLIDER_QUANTUM:
                gates.apply_plan(self.state, self.slider_plan(m, mt))
            else:
                for g, qs in self.unitary_ops(m, mt, c, p):
                    self.state.apply(gates.matrix(g), qs)
            # classical type information (Algorithm "Execute Move", step "update v")
            self.types[t] = vs
            if mt is MoveType.SPLIT_JUMP or mt is SliderMoveType.SPLIT_SLIDE:
                self.types[m.target2] = vs
            if kind(vs) == "P" and self.geo.xy(t)[1] == self.geo.last_rank(self.turn):
                self.types[t] = "Q" if self.turn == WHITE else "q"
        self._clear_empty_types()
        res = MoveResult(m, mt, outcome, p1, m.text(self.geo))
        self.history.append(res)
        self.turn = other(self.turn)
        return res

    def _clear_empty_types(self) -> None:
        for q in self.geo.squares():
            if self.types[q] is not None and self.prob(q) <= EPS:
                self.types[q] = None

    # ---- sampling (for the classical baseline) ---------------------------------------------
    def sample_boards(self, k: int, rng: np.random.Generator | None = None) -> list[Sample]:
        rng = rng or self.rng
        mask = (1 << self.nsq) - 1
        out = []
        for b in self.state.sample(rng, k):
            bits = b & mask
            types = [self.types[q] if (bits >> q) & 1 else None for q in self.geo.squares()]
            out.append(Sample(bits, types, self.turn, to_fen(types, self.turn, self.n)))
        return out

    def classical_fen(self) -> str | None:
        """FEN if the board is classical (every square's probability is 0 or 1), else None."""
        if self.state.support_size() != 1:
            return None
        return to_fen(self.types, self.turn, self.n)
