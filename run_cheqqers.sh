#!/bin/bash
# Anabi for Quantum Cheqqers: HTTP agent on 127.0.0.1:${PORT:-8080}/move (AI-tournament payload).
# Needs: Python deps (requirements.txt) and Node 18+ with `npm install` run in this folder.
cd "$(dirname "$0")"
export AOG_CQWEB_BRIDGE="$PWD/cheqqers/cqweb_bridge_kit.mjs"
export AOG_CQWEB_GAME="$PWD/cheqqers/tournament_game_v4.mjs"
exec ${PYTHON:-python3} -m qchess.games.cheqqers_tournament_bot --port "${PORT:-8080}" --seconds 1.5 \
     --book cheqqers/book_d3.json "$@"
