# Changes between arena days

## 16 November 2026 (day 3): this version
- **Cheqqers:**
  - Deep opening book (`cheqqers/book_d3.json`), loaded with `--book`.
  - Against the day-2 agent: 0.556 over 1,000 games, 1.5 s per move, colours swapped.
- **TiqTaqToe:**
  - Anti-random mode (`--anti-random --anti-random-k 3`).
  - Exact expected score against a uniformly random opponent: 0.9723 → 0.9773 per round.
  - Unchanged against perfect opponents.

## 13 November 2026 (day 2)
- **Cheqqers:**
  - Near-best move variety, seeded per game. On day 1, two deterministic agents replayed the same game.
  - Rules copy updated to the 12 November referee.
- **TiqTaqToe:** tie-breaks against a random player among the optimal moves (`rand_q2_d0.npz`).

## 12 November 2026 (day 1)
- **Cheqqers:** the search agent with state tracking, deterministic (no move variety), 1.5 s per move.
- **TiqTaqToe:** perfect play from the exact table; ties broken by a "spread" heuristic.
