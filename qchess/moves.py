"""Move notation.

* ``b1c3``        two-square move (jump, slide, pawn step, any capture — the board decides)
* ``b1^a3c3``     split: source ``b1`` to targets ``a3`` and ``c3`` (Cantwell writes b1∧a3c3)
* ``a3c3^b1``     merge: sources ``a3`` and ``c3`` to target ``b1``

A trailing promotion letter (``a4a5q``) is accepted and ignored: promotion is always to a queen.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from .geometry import Geometry


class MoveType(enum.Enum):
    JUMP = "jump"
    BLOCKED_JUMP = "blocked_jump"
    CAPTURE_JUMP = "capture_jump"
    SLIDE = "slide"
    BLOCKED_SLIDE = "blocked_slide"
    CAPTURE_SLIDE = "capture_slide"
    SPLIT_JUMP = "split_jump"
    MERGE_JUMP = "merge_jump"
    PAWN_STEP = "pawn_step"
    BLOCKED_PAWN_STEP = "blocked_pawn_step"
    PAWN_CAPTURE = "pawn_capture"


class SliderMoveType(enum.Enum):
    """Cantwell's Split Slide / Merge Slide for B, R, Q (arXiv:1906.05836, eqs. SplitSlideUnitary,
    MergeSlideUnitary); only produced by a board built with ``slider_splits=True``.  Kept out of
    ``MoveType`` so that the K/N-only move-type enumeration (and everything that iterates it, such as
    the circuit scenarios' coverage test) is exactly as before."""
    SPLIT_SLIDE = "split_slide"
    MERGE_SLIDE = "merge_slide"


MEASURING = {
    MoveType.BLOCKED_JUMP,
    MoveType.CAPTURE_JUMP,
    MoveType.BLOCKED_SLIDE,
    MoveType.CAPTURE_SLIDE,
    MoveType.BLOCKED_PAWN_STEP,
    MoveType.PAWN_CAPTURE,
}


@dataclass(frozen=True)
class Move:
    source: int
    target: int
    source2: int | None = None   # merge
    target2: int | None = None   # split

    @property
    def is_split(self) -> bool:
        return self.target2 is not None

    @property
    def is_merge(self) -> bool:
        return self.source2 is not None

    @classmethod
    def parse(cls, text: str, geo: Geometry) -> "Move":
        text = text.strip().replace("∧", "^")
        if "^" in text:
            left, right = text.split("^")
            if len(left) == 2:   # split  s ^ t1 t2
                return cls(geo.sq(left), geo.sq(right[:2]), target2=geo.sq(right[2:4]))
            return cls(geo.sq(left[:2]), geo.sq(right[:2]), source2=geo.sq(left[2:4]))
        return cls(geo.sq(text[:2]), geo.sq(text[2:4]))

    def text(self, geo: Geometry) -> str:
        if self.is_split:
            return f"{geo.name(self.source)}^{geo.name(self.target)}{geo.name(self.target2)}"
        if self.is_merge:
            return f"{geo.name(self.source)}{geo.name(self.source2)}^{geo.name(self.target)}"
        return f"{geo.name(self.source)}{geo.name(self.target)}"
