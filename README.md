# Anabi: agents for Quantum Cheqqers and Quantum TiqTaqToe

Team **Ananya & Akshay** (Ananya Dutta, Akshay Krishna Sirigana), live arena of the quantum games project,
12, 13 and 16 November 2026.

This is the version that played on **16 November 2026** (arena day 3). `CHANGELOG.md` lists what changed between
the arena days.

Both agents are HTTP servers that speak the AI-tournament protocol of the local referees
(`local-cheqqers-server.js`, `local-tiqtaqtoe-server.js`). `POST /move` takes the referee's payload and answers
`{"move": [...]}`, always one of `legalMoves`. They bind to 127.0.0.1.

## Requirements
- Python 3.11 or newer: `pip install -r requirements.txt` (numpy, scipy, qiskit).
- Node 18 or newer, for Cheqqers only: `npm install` in this folder (installs `@quantumplayed/quantum-game-engine` 1.0.2,
  the engine version the arenas run).

## Run
```bash
./run_cheqqers.sh      # Quantum Cheqqers  on http://127.0.0.1:8080/move
./run_tiqtaqtoe.sh     # Quantum TiqTaqToe on http://127.0.0.1:8081/move
```
Use another port with `PORT=9000 ./run_cheqqers.sh`, and another interpreter with `PYTHON=/path/to/python`.
`GET /` (Cheqqers) and `GET /health` (TiqTaqToe) report counters.

With a local referee, for example:
```bash
node local-cheqqers-server.js --size 6 --quantumness 2 --timeout 3000 --p1 remote --p1-url http://127.0.0.1:8080/move --p2 local
node local-tiqtaqtoe-server.js --quantumness 2 --decoherence 0 --p1 remote --p1-url http://127.0.0.1:8081/move --p2 local
```

## How the agents work

### Quantum Cheqqers (`qchess/games/cheqqers_tournament_bot.py`)
- **Exact rules.** The game model is the referee's own code: `cheqqers/tournament_game_v4.mjs`, the 12 November
  2026 `local-cheqqers-server.js` (lines 1-918) with an export line. It runs in Node and is reached through a small
  bridge (`cheqqers/cqweb_bridge_kit.mjs`), so the agent searches exactly the rules the server plays.
- **State tracking.** The payload has the amplitudes but not the referee's hidden bookkeeping (piece ids, the
  multi-jump square, repetition and draw counters). The agent keeps the referee state it last answered from, and
  finds the descendant that matches the next request: its own move's outcome, then the opponent's moves. If nothing
  matches, it rebuilds the board from the payload.
- **Search.** Monte Carlo tree search (PUCT selection) with exact chance nodes for measurements, and a classical
  minimax evaluation at the leaves (`qchess/engines/mcts.py`, configuration `qchess/engines/anabi.py`).
  1.5 s per move at the 3000 ms limit.
- **Opening book** (`cheqqers/book_d3.json`). 338 positions, each searched for 60 s: every position where the agent
  was to move before ply 24 in the arena games of 12-13 November against the other teams. Where several moves are
  close, one is drawn by visit count.
- **Move variety.** After the search, the move is drawn among the near-best root moves: at least 25 % of the best
  move's visits, and a mean value within 0.02. The random generator is seeded per game.
- **Safety.** The answer is always sent before the deadline, and it is always a legal move.

### Quantum TiqTaqToe (`qchess/games/tqt_tournament_bot.py`)
- **Perfect play.** Moves come from an exact expectiminimax solution of the referee's rules, with measurement
  outcomes weighted by the referee's own probabilities. `tiqtaqtoe/tables/kit_q2_d0.npz` covers the arena setting
  (quantumness 2, decoherence off; 1,643,542 states). It is solved by `qchess/games/tqt_kit_solver.py`.
- **Tie-breaks against a random player** (`rand_q2_d0.npz`). Among the optimal moves, the agent picks the one with the
  highest expected score against a uniformly random opponent. It never leaves the optimal set.
- **Anti-random mode** (`arand_q2_d0.npz`, `--anti-random --anti-random-k 3`). After the opponent has played
  3 provably non-optimal moves in a round, the agent plays the best move against a random opponent. A perfect
  opponent never triggers it.
- **Other settings.** Without a table, the agent falls back to a capped search. Tables for other settings can be
  built with the solver modules.

## Files
- `qchess/`: the Python package (engines and game layers used by the two agents).
- `cheqqers/`: the rules copy, the Node bridge and the opening book.
- `tiqtaqtoe/tables/`: the exact, tie-break and anti-random tables for the arena setting.
- `run_cheqqers.sh`, `run_tiqtaqtoe.sh`, `requirements.txt`, `package.json`.

## Acknowledgements
`cheqqers/tournament_game_v4.mjs` is derived from the organizers' referee code, and the game engine is
`@quantumplayed/quantum-game-engine`.

Much of the code was written with the help of an AI coding assistant (Claude, by Anthropic), under the authors'
direction. The authors designed the approaches, ran and checked the experiments, and are responsible for the code.
