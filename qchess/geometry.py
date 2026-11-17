"""Board geometry, piece movement patterns and start positions for N x N quantum minichess.

Squares are indexed ``idx = rank_index * n + file_index`` (a1 = 0, b1 = 1, ...), and the
occupancy qubit of a square has the same index.  Piece types use FEN letters; white is
upper case.  Nothing in this module knows about superposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

WHITE = "w"
BLACK = "b"
FILES = "abcdefghij"

JUMPERS = "NK"      # pieces that use the jump unitary (and may split / merge)
SLIDERS = "BRQ"     # pieces that use the path-controlled slide unitary

# Start positions.  5x5 is Gardner minichess exactly as Fairy-Stockfish's built-in ``gardner``
# variant spells it.  There is no standard 4x4 Gardner, so the 4x4 setup is a custom reduction
# (R N Q K with pawns staggered so that both pawn steps and pawn captures exist from move one);
# the same FEN is registered with Fairy-Stockfish as the custom variant ``gardner4`` by
# ``qchess.baseline``.
START_FEN = {
    5: "rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1",
    4: "rnqk/1p1p/P1P1/RNQK w - - 0 1",
}

KNIGHT_DELTAS = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
KING_DELTAS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def color_of(piece: str) -> str:
    return WHITE if piece.isupper() else BLACK


def other(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def kind(piece: str) -> str:
    """Upper-case piece letter regardless of colour."""
    return piece.upper()


@dataclass(frozen=True)
class Geometry:
    n: int

    # ---- square naming -------------------------------------------------------------
    def idx(self, x: int, y: int) -> int:
        return y * self.n + x

    def xy(self, idx: int) -> tuple[int, int]:
        return idx % self.n, idx // self.n

    def in_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.n and 0 <= y < self.n

    def sq(self, name: str) -> int:
        name = name.strip().lower()
        x = FILES.index(name[0])
        y = int(name[1:]) - 1
        if not self.in_board(x, y):
            raise ValueError(f"square {name!r} is off a {self.n}x{self.n} board")
        return self.idx(x, y)

    def name(self, idx: int) -> str:
        x, y = self.xy(idx)
        return f"{FILES[x]}{y + 1}"

    def squares(self) -> range:
        return range(self.n * self.n)

    # ---- movement patterns ---------------------------------------------------------
    def jump_targets(self, s: int, piece: str) -> list[int]:
        """Targets of a knight or king jump from ``s`` (board edges only; occupancy ignored)."""
        x, y = self.xy(s)
        deltas = KNIGHT_DELTAS if kind(piece) == "N" else KING_DELTAS
        return [self.idx(x + dx, y + dy) for dx, dy in deltas if self.in_board(x + dx, y + dy)]

    def ray_targets(self, s: int, piece: str) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(target, path)`` for every square a slider could reach from ``s``.

        ``path`` is the list of squares strictly between source and target.  Occupancy is
        deliberately ignored here: in quantum chess the path is handled by control qubits.
        """
        k = kind(piece)
        dirs = (ROOK_DIRS if k == "R" else BISHOP_DIRS if k == "B" else ROOK_DIRS + BISHOP_DIRS)
        x, y = self.xy(s)
        for dx, dy in dirs:
            path: list[int] = []
            cx, cy = x + dx, y + dy
            while self.in_board(cx, cy):
                t = self.idx(cx, cy)
                yield t, list(path)
                path.append(t)
                cx, cy = cx + dx, cy + dy

    def path(self, s: int, t: int) -> list[int]:
        """Squares strictly between two collinear squares."""
        x0, y0 = self.xy(s)
        x1, y1 = self.xy(t)
        dx, dy = x1 - x0, y1 - y0
        if dx and dy and abs(dx) != abs(dy):
            raise ValueError("squares are not collinear")
        steps = max(abs(dx), abs(dy))
        sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
        return [self.idx(x0 + sx * i, y0 + sy * i) for i in range(1, steps)]

    def pawn_dir(self, color: str) -> int:
        return 1 if color == WHITE else -1

    def pawn_step(self, s: int, color: str) -> int | None:
        x, y = self.xy(s)
        y2 = y + self.pawn_dir(color)
        return self.idx(x, y2) if self.in_board(x, y2) else None

    def pawn_captures(self, s: int, color: str) -> list[int]:
        x, y = self.xy(s)
        y2 = y + self.pawn_dir(color)
        return [self.idx(x + dx, y2) for dx in (-1, 1) if self.in_board(x + dx, y2)]

    def last_rank(self, color: str) -> int:
        return self.n - 1 if color == WHITE else 0

    def valid(self, t: int, s: int, piece: str) -> bool:
        """Cantwell's ``valid(t, s, v_s)``: is ``s -> t`` a legal movement pattern for the piece?

        Pawn patterns are *not* covered here (they depend on whether the move captures); use
        ``pawn_step`` / ``pawn_captures``.
        """
        k = kind(piece)
        if k in JUMPERS:
            return t in self.jump_targets(s, piece)
        if k in SLIDERS:
            return any(tt == t for tt, _ in self.ray_targets(s, piece))
        raise ValueError(f"valid() is not defined for pawns ({piece!r})")


# ---- FEN helpers ---------------------------------------------------------------------------
def parse_fen(fen: str, n: int) -> tuple[list[str | None], str]:
    """Return ``(types, turn)``; ``types[idx]`` is a FEN letter or ``None``."""
    parts = fen.split()
    rows = parts[0].split("/")
    if len(rows) != n:
        raise ValueError(f"FEN has {len(rows)} ranks, expected {n}")
    types: list[str | None] = [None] * (n * n)
    geo = Geometry(n)
    for r, row in enumerate(rows):
        y = n - 1 - r
        x = 0
        for ch in row:
            if ch.isdigit():
                x += int(ch)
            else:
                types[geo.idx(x, y)] = ch
                x += 1
        if x != n:
            raise ValueError(f"rank {y + 1} of FEN has width {x}, expected {n}")
    turn = parts[1] if len(parts) > 1 else WHITE
    return types, turn


def to_fen(types: list[str | None], turn: str, n: int, occupied: int | None = None) -> str:
    """Classical FEN of a type vector.  ``occupied`` (a bitmask) restricts to occupied squares."""
    geo = Geometry(n)
    rows = []
    for y in range(n - 1, -1, -1):
        row = ""
        empty = 0
        for x in range(n):
            i = geo.idx(x, y)
            p = types[i]
            if p is not None and (occupied is None or (occupied >> i) & 1):
                if empty:
                    row += str(empty)
                    empty = 0
                row += p
            else:
                empty += 1
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows) + f" {turn} - - 0 1"
