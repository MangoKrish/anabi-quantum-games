#!/bin/bash
# Anabi for Quantum TiqTaqToe: HTTP agent on 127.0.0.1:${PORT:-8081}/move (AI-tournament payload).
cd "$(dirname "$0")"
exec ${PYTHON:-python3} -m qchess.games.tqt_tournament_bot --port "${PORT:-8081}" --tables tiqtaqtoe/tables \
     --anti-random --anti-random-k 3 "$@"
