"""Anabi on boards of other rule sets (``board.rules_variant`` set), including 8x8 ones.

Two adapters, both outside Anabi's own module (``qchess.engines.anabi`` is unchanged):

* ``FSF8Evaluator``: the ``fsf`` evaluator with Anabi's settings (depth 8, ``exact_max`` 16,
  ``samples`` 8, default ``cp_scale``), extended to 8x8 boards.  On 4x4 / 5x5 it is the plain
  ``FSFEvaluator``.  On 8x8 every playable classical branch goes to Fairy-Stockfish in the king-capture
  variant ``chesskc`` (``KC8_INI``: standard chess with the king turned into a capturable commoner,
  losing all kings is a loss, castling off because branch FENs carry no castling rights), exactly as
  ``gardnerkc`` does on 5x5.  Without this class the stock evaluator silently scores 8x8 boards with
  its material fallback, which would not be Anabi.
* ``ForeignMCTS``: ``MCTSEngine`` unchanged except its no-iteration fallback.  When the budget ends
  before the root's one-ply expansion is complete, the stock fallback ranks moves with
  ``heuristic_scores``, which reads qchess's move classes; here it plays the best root move whose
  one-ply value is already known (else the first legal move), as the Deep Dirac adapter
  (``reports/engines/dd_original/engine_server.py``) did.

``make_anabi_foreign()`` builds Anabi's exact config (``qchess.engines.anabi.ANABI``) on these two
classes.  Added 24 October 2026 for ``reports/engines/anabi_vs_world``.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import weakref

from . import evaluators as _ev
from .anabi import ANABI
from .base import Candidate
from .evaluators import FSFEvaluator
from .mcts import MCTSEngine
from ..geometry import to_fen

KC8_VARIANT = "chesskc"
KC8_INI = """[chesskc:chess]
king = -
commoner = k
extinctionValue = loss
extinctionPieceTypes = k
castling = false
"""


class _KC8Engine:
    """Minimal UCI driver of Fairy-Stockfish playing ``chesskc`` (the interface ``FSFEvaluator`` uses)."""

    def __init__(self, path: str = "fairy-stockfish", hash_mb: int = 16):
        exe = shutil.which(path) or path
        if not os.path.exists(exe):
            raise FileNotFoundError(f"engine binary not found: {path}")
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        fd, ini = tempfile.mkstemp(suffix=".ini", prefix="qchess_kc8_")
        with os.fdopen(fd, "w") as fh:
            fh.write(KC8_INI)
        try:
            self._send("uci")
            self._read_until("uciok")
            self._send(f"load {ini}")
            self._send("setoption name Threads value 1")
            self._send(f"setoption name Hash value {hash_mb}")
            self._send(f"setoption name UCI_Variant value {KC8_VARIANT}")
            self._send("isready")
            self._read_until("readyok")
        finally:
            os.remove(ini)

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine terminated")
            line = line.strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def _pyffish8():
    import pyffish as sf
    if KC8_VARIANT not in sf.variants():
        sf.load_variant_config(KC8_INI)
    return sf


class FSF8Evaluator(FSFEvaluator):
    """``FSFEvaluator`` that also plays 8x8 boards (variant ``chesskc``); 4x4 / 5x5 unchanged."""

    def _engine(self, n: int):
        if n != 8:
            return super()._engine(n)
        if n in self._dead:
            return None
        eng = self._engines.get(n)
        if eng is not None and self._pids.get(n) != os.getpid():
            _ev._FOREIGN.append(self._engines.pop(n))       # inherited through fork: leave it alone
            self._pids.pop(n, None)
            eng = None
        if eng is None:
            for m in [m for m in self._engines if m != n and m not in self._used]:
                self._release(m)
            _ev._make_room(self)
            try:
                self._sf = _pyffish8()
                eng = self._engines[n] = _KC8Engine(self.path, self.hash_mb)
            except (FileNotFoundError, OSError, RuntimeError, ImportError):
                self._dead.add(n)
                self.counters["engine_unavailable"] += 1
                return None
            self._pids[n] = os.getpid()
            self._used.discard(n)
            _ev._LIVE[(id(self), n)] = weakref.ref(self)
            self.counters["engine_starts"] += 1
        elif (id(self), n) in _ev._LIVE:
            _ev._LIVE.move_to_end((id(self), n))
        return eng

    def _valid(self, n: int, fen: str) -> bool:
        if n != 8:
            return super()._valid(n, fen)
        if self._sf is None:
            return False
        try:
            return self._sf.validate_fen(fen, KC8_VARIANT) == 1
        except Exception:
            return False

    def _playable_values(self, board, scan, playable, me):
        n = board.n
        if n != 8:
            return super()._playable_values(board, scan, playable, me)
        # FSFEvaluator._playable_values with 8 among the supported sizes (same computation)
        off = self._offset(board, scan, me)
        stm = board.turn
        sgn = 1.0 if stm == me else -1.0
        out, fallback = [], []
        if self._engine(n) is None:
            fallback = list(playable)
        else:
            for w, bits in self._select(board, playable):
                fen = to_fen(board.types, stm, n, occupied=bits)
                if fen not in self._cache and not self._valid(n, fen):
                    self.counters["invalid_fen"] += 1
                    fallback.append((w, bits))
                    continue
                r = self._query(n, fen)
                if r is None:
                    fallback.append((w, bits))
                    continue
                kind, x = r
                if kind == "mate":
                    out.append((w, sgn * _ev._mate_value(x)))
                else:
                    out.append((w, _ev.HEUR_MAX * math.tanh((sgn * x / 100.0 + off) / self._cp_pawns)))
        if fallback:
            self.counters["fallback_branches"] += len(fallback)
            out.extend(self._fallback._heuristic_values(board, scan, fallback, me, off))
        return out

    def set_game_info(self, info: dict | None) -> None:
        super().set_game_info(info)
        if int(self.game_info.get("n") or 0) == 8 and shutil.which(self.path):
            self._engine(8)


class ForeignMCTS(MCTSEngine):
    """``MCTSEngine`` with a fallback that does not need qchess move classes (see the module docstring)."""

    def _candidates(self, root, board, iters):
        if root.moves is None:
            best, bv = None, -math.inf
            for m, e in root.edges.items():
                tot = 0.0
                for o, p in e.outcomes:
                    c = e.children.get(o)
                    v = None if c is None else (c.tv if c.tv is not None else c.ev)
                    if v is None:
                        tot = None
                        break
                    tot += p * v
                if tot is not None and tot > bv:
                    best, bv = m, tot
            if best is None:
                best = board.legal_moves()[0]
            self._stats["foreign_fallback"] = self._stats.get("foreign_fallback", 0) + 1
            return [Candidate(best, 0.0, 0.0, 0.0, 0, 0.0, {"iterations": iters, "fallback": True})]
        return super()._candidates(root, board, iters)

    def close(self) -> None:
        super().close()
        if not self._owns_eval and hasattr(self.evaluator, "close"):
            self.evaluator.close()


def make_anabi_foreign(**overrides) -> ForeignMCTS:
    """Anabi's config (``ANABI``, overrides as in ``make_anabi``) on ``ForeignMCTS`` + ``FSF8Evaluator``."""
    cfg = {**ANABI, **overrides}
    eng = ForeignMCTS(cfg, evaluator=FSF8Evaluator(cfg["eval"]))
    eng.name = cfg.get("name", "Anabi")
    return eng


__all__ = ["FSF8Evaluator", "ForeignMCTS", "make_anabi_foreign", "KC8_INI", "KC8_VARIANT"]
