"""Anabi on other quantum games: Quantum TiqTaqToe and Cheqqers (quantum checkers).

Added 26 October 2026. Anabi's own modules (``qchess.engines.anabi``, ``qchess.engines.mcts``) are imported unchanged.

* ``core``        - the shared layer: the board protocol that lets ``MCTSEngine`` search any rule set,
                    the branch-averaged evaluator, ``make_anabi_for`` and the match loop.
* ``tiqtaqtoe``   - Quantum TiqTaqToe rule sets (tiqtaqtoe.com public levels Q0-Q2, quantumlib V1-V3).
* ``tqt_solver``  - exact expectiminimax for the bounded TiqTaqToe rule sets (the optimum).
* ``cheqqers``    - port of Cheqqers (arXiv:2506.05962, raatmarien/cheqqers tag cog-paper-version).
* ``opponents``   - baselines (random, greedy, classical-only, perfect).
* ``external``    - the adapter slot for outside AIs (Evert van Nieuwenburg's external-player API).
"""
