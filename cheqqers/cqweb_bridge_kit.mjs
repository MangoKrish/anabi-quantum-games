// [kit copy, 7 Nov 2026: + cmd 'from_payload' (rebuild a game from an AI-tournament payload); rules untouched]
// [11 Nov 2026: outcome enumeration follows the engine's sampler exactly (samplerDist)]
// JSON-lines bridge to the 1st European Tournament in Quantum Games' own Cheqqers referee
// (github.com/VincenzoLipardi/ETQG_2026 Cheqqers/local-cheqqers-server.js, lines 1-906 unmodified + one export
// line = tournament_game.mjs) running on Evert van Nieuwenburg's @quantumplayed/quantum-game-engine 1.0.3.
// Krishna allowed running both offline (26 Oct 2026). This file adds no rules: it stores game states, deep-clones
// them, forces measurement outcomes through the game's own measureSquare(index, forced) and enumerates them.
//
// Usage: node cqweb_bridge.mjs /path/tournament_game.mjs      (stdin/stdout JSON lines; logs to stderr)
// Moves are encoded ["m", take, from, to] / ["s", from, to1, to2] / ["g", from1, from2, to] with the game's own
// square indices (index 0 = top-left dark square, Black's side).
import { pathToFileURL } from 'url';
import readline from 'readline';

const T = await import(pathToFileURL(process.argv[2]).href);
await T.initEngine();
const { CheqqersGame, serializeBoardState, moveToArray, GameState, PieceColor, JointQuantumSystem } = T;
const out = process.stdout;
console.log = () => {};                         // the game logs a lot; keep the protocol stream clean
console.error = (...a) => process.stderr.write(a.join(' ') + '\n');

// ---- seeded Math.random: only reached by an UNforced measurement (counted; must stay 0 in our use) ----
let unforcedRandom = 0, seed = 1;
Math.random = () => { unforcedRandom++; seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };

// ---- forcing / enumerating measurements -----------------------------------------------------------------
// [11 Nov 2026] The engine samples an unforced measurement as: rand = Math.random(); walk the probability map in
// key order and take the first key whose cumulative probability reaches rand; if none does (the map sums to
// less than 1, as unnormalised interference states can), take the LAST key, even one of probability 0.  So the
// exact outcome distribution is the clamped cumulative widths plus the residual on the last key - not the map
// renormalised.  (Forcing a zero-probability key reproduces the engine's own fallback branch.)
function samplerDist(probs) {
  const keys = Object.keys(probs), dist = {};
  let cum = 0;
  for (const k of keys) {
    const lo = Math.min(cum, 1); cum += probs[k]; const hi = Math.min(cum, 1);
    if (hi - lo > 0) dist[k] = (dist[k] || 0) + (hi - lo);
  }
  if (cum < 1 && keys.length) { const last = keys[keys.length - 1]; dist[last] = (dist[last] || 0) + (1 - cum); }
  return dist;
}
let FORCE = null;                                // {prefix:[..], k:0, need:null}
const origMeasure = CheqqersGame.prototype.measureSquare;
CheqqersGame.prototype.measureSquare = function (index, forced) {
  if (FORCE && forced === undefined) {
    if (FORCE.k < FORCE.prefix.length) {
      forced = FORCE.prefix[FORCE.k++];
    } else {
      const sq = this.squares[index];
      const probs = sq.entangledSystem ? sq.entangledSystem.getTargetProbabilities(sq) : sq.getProbabilities();
      if (FORCE.need === null) FORCE.need = samplerDist(probs);
      let best = 0, bp = -1;
      for (const [s, p] of Object.entries(probs)) if (p > bp) { bp = p; best = parseInt(s); }
      forced = best;                             // continue deterministically; this run is discarded
      FORCE.k++;
    }
  }
  return origMeasure.call(this, index, forced);
};

// ---- deep clone preserving prototypes, shared references and cycles --------------------------------------
function deepClone(x, memo = new Map()) {
  if (x === null || typeof x !== 'object') return x;
  if (memo.has(x)) return memo.get(x);
  if (Array.isArray(x)) { const a = []; memo.set(x, a); for (const v of x) a.push(deepClone(v, memo)); return a; }
  if (x instanceof Map) { const m = new Map(); memo.set(x, m); for (const [k, v] of x) m.set(deepClone(k, memo), deepClone(v, memo)); return m; }
  if (x instanceof Set) { const s = new Set(); memo.set(x, s); for (const v of x) s.add(deepClone(v, memo)); return s; }
  const o = Object.create(Object.getPrototypeOf(x)); memo.set(x, o);
  for (const k of Object.keys(x)) o[k] = deepClone(x[k], memo);
  return o;
}

function enc(m) {
  if ('to_index1' in m) return ['s', m.from_index, m.to_index1, m.to_index2];
  if ('from_index1' in m) return ['g', m.from_index1, m.from_index2, m.to_index];
  return ['m', !!m.is_take_move, m.from_index, m.to_index];
}
const RES = { [GameState.IN_PROGRESS]: null, [GameState.WHITE_WON]: 'white', [GameState.BLACK_WON]: 'black', [GameState.DRAW]: 'draw' };

const store = new Map();
let nextId = 1;
const put = (g) => { const id = nextId++; store.set(id, { g, moves: null }); return id; };
const get = (id) => { const e = store.get(id); if (!e) throw new Error(`unknown id ${id}`); return e; };
function moves(e) { if (e.moves === null) e.moves = e.g.state === GameState.IN_PROGRESS ? e.g.getPossibleMoves() : []; return e.moves; }
function info(id) {
  const e = get(id), g = e.g;
  return { id, turn: g.turn === PieceColor.WHITE ? 'w' : 'b', result: RES[g.state], moves: moves(e).map(enc),
           multi: g.multiJumpSquare, since_take: g.movesSinceTake, hash: g.getBoardStateHash() };
}

function children(id, mi) {
  const e = get(id), m = moves(e)[mi];
  const outs = [];
  const rec = (prefix, p) => {
    const g = deepClone(e.g);
    FORCE = { prefix, k: 0, need: null };
    g.applyMove(m);
    const need = FORCE.need; FORCE = null;
    if (need === null) { outs.push({ o: prefix, p, id: put(g) }); return; }
    for (const [s, q] of Object.entries(need)) if (q > 1e-12) rec(prefix.concat([parseInt(s)]), p * q);
  };
  rec([], 1.0);
  const tot = outs.reduce((a, b) => a + b.p, 0);
  return outs.map(c => ({ o: c.o, p: c.p / tot, ...info(c.id) }));
}

function groups(id) {
  // independent measurement groups: [[ [p, occupiedSquares[]], ... ], ...] for the evaluator's branches
  const g = get(id).g, seen = new Set(), res = [];
  for (let i = 0; i < g.squares.length; i++) {
    const sq = g.squares[i], sys = sq.entangledSystem;
    if (sys) {
      if (seen.has(sys)) continue; seen.add(sys);
      const idx = sys.entities.map(en => g.squares.indexOf(en)), acc = new Map();
      for (const [k, a] of Object.entries(sys.jointAmplitudes)) {
        const p = a.re * a.re + a.im * a.im; if (p <= 1e-12) continue;
        const st = sys.decomposeState(parseInt(k)), occ = idx.filter((_, j) => st[j] === 1);
        const key = occ.join(','); acc.set(key, (acc.get(key) || 0) + p);
      }
      res.push({ squares: idx, dist: [...acc].map(([k, p]) => [p, k === '' ? [] : k.split(',').map(Number)]) });
    } else {
      const pr = sq.getProbabilities(), p1 = pr[1] || 0;
      if (p1 > 1e-12 && p1 < 1 - 1e-12) res.push({ squares: [i], dist: [[1 - p1, []], [p1, [i]]] });
    }
  }
  const pieces = [];
  for (let i = 0; i < g.squares.length; i++) {
    const pp = g.pieceProperties[i], occ = g.squares[i].getProbabilities()[1] || 0;
    pieces.push(pp && occ > 1e-12 ? [pp.color === PieceColor.WHITE ? 'w' : 'b', !!pp.crowned, occ] : null);
  }
  return { groups: res, pieces };
}

function payload(id, timeout) {
  const g = get(id).g, st = g.toGameStateObject();
  return { gameId: 'aog', roundId: 'round-1', player: g.turn === PieceColor.WHITE ? 'white' : 'black',
           currentTurn: g.turn === PieceColor.WHITE ? 'white' : 'black', board: serializeBoardState(st),
           quantumStates: st.quantum_states, settings: { boardSize: g.boardSize, quantumness: g.gameType, timeout },
           legalMoves: st.possible_moves.map(moveToArray) };
}


// ---- rebuild a game from an AI-tournament payload (Evert's kit, 7 Nov 2026) -------------------------------
// The payload carries the full amplitudes (quantumStates) but not pieceIds, the multi-jump square, the pieces
// jumped this turn, the repetition history or the draw counters. pieceIds: one id per piece, shared by the
// same-colour squares of one entangled system (a split piece keeps its id in the referee). The multi-jump square
// is inferred: if the rebuilt move list differs from legalMoves and every legal move starts on one square, the
// game is continued from that square. Returns info + 'match' (rebuilt moves == legalMoves, same order).
const sameMoves = (g, legal) => {
  const mine = g.getPossibleMoves().map(moveToArray);
  return mine.length === legal.length && mine.every((m, i) => m.length === legal[i].length && m.every((v, j) => v === legal[i][j]));
};
function fromPayload(pl) {
  const size = pl.settings.boardSize, type = pl.settings.quantumness;
  const g = new CheqqersGame(size, 0, type);
  g.turn = pl.currentTurn === 'white' ? PieceColor.WHITE : PieceColor.BLACK;
  const code = (s) => (s === 'E' ? 0 : 1);
  const ids = new Array(g.squares.length).fill(0);
  let nid = 1;
  for (const qs of pl.quantumStates || []) {
    if (qs.type === 'entangled') {
      const byColour = {};
      for (const i of qs.squares) { const c = pl.board[i] && pl.board[i].piece; if (c) ids[i] = byColour[c] || (byColour[c] = nid++); }
    }
  }
  for (let i = 0; i < g.squares.length; i++) {
    const c = pl.board[i];
    if (c && c.piece) g.pieceProperties[i] = { color: c.piece === 'white' ? PieceColor.WHITE : PieceColor.BLACK,
                                               crowned: !!c.crowned, pieceIds: [ids[i] || nid++] };
  }
  g.nextPieceId = nid;
  for (const qs of pl.quantumStates || []) {
    if (qs.type === 'entangled') {
      const ents = qs.squares.map(i => g.squares[i]);
      const sys = JointQuantumSystem.create(ents);
      const amps = {};
      for (const a of qs.amplitudes) amps[sys.composeState(a.state.split('⊗').map(code))] = { re: a.re, im: a.im };
      sys.jointAmplitudes = amps;
    } else {
      const i = qs.squares[0], amps = {};
      for (const a of qs.amplitudes) amps[code(a.state)] = { re: a.re, im: a.im };
      g.squares[i].entangledSystem = undefined;
      g.squares[i].amplitudes = amps;
    }
  }
  const legal = pl.legalMoves || [];
  let match = sameMoves(g, legal);
  if (!match && legal.length && legal.every(m => m.length === 2 && m[0] === legal[0][0])) {
    g.multiJumpSquare = legal[0][0];
    match = sameMoves(g, legal);
    // [11 Nov 2026 review] keep it even without an exact match: every legal move continues from that square
  }
  const id = put(g);
  return { ...info(id), match };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let msg, r;
  try {
    msg = JSON.parse(line);
    switch (msg.cmd) {
      case 'new': r = info(put(new CheqqersGame(msg.size, msg.rows, msg.type))); break;
      case 'info': r = info(msg.id); break;
      case 'from_payload': r = fromPayload(msg.payload); break;
      case 'children': r = { children: children(msg.id, msg.mi) }; break;
      case 'groups': r = groups(msg.id); break;
      case 'payload': r = { payload: payload(msg.id, msg.timeout) }; break;
      case 'state': r = { state: get(msg.id).g.toGameStateObject(true), hist: [...get(msg.id).g.stateHistory],
                          q: get(msg.id).g.consecutiveQueenMoves }; break;
      case 'sample': {                           // validation: the engine's own unforced sampling of a move
        const e = get(msg.id), m = moves(e)[msg.mi], counts = {};
        seed = msg.seed >>> 0;
        for (let k = 0; k < msg.n; k++) { const g = deepClone(e.g); g.applyMove(m); const h = g.getBoardStateHash() + '|' + (g.turn === PieceColor.WHITE ? 'w' : 'b') + '|' + RES[g.state]; counts[h] = (counts[h] || 0) + 1; }
        r = { counts }; break;
      }
      case 'replay': {                           // validation: the game's own replay path, no clone, no hook
        const g = new CheqqersGame(msg.size, msg.rows, msg.type), states = [];
        for (const [mi, forced] of msg.moves) {
          const m = g.state === GameState.IN_PROGRESS ? g.getPossibleMoves()[mi] : null;
          if (!m) { states.push(null); break; }
          g.applyMove(m, forced);
          states.push({ hash: g.getBoardStateHash(), turn: g.turn === PieceColor.WHITE ? 'w' : 'b', result: RES[g.state],
                        st: g.toGameStateObject(true), hist: [...g.stateHistory], q: g.consecutiveQueenMoves });
        }
        r = { states }; break;
      }
      case 'release': for (const i of msg.ids) store.delete(i); r = { ok: true, live: store.size }; break;
      case 'stats': r = { live: store.size, unforced_random: unforcedRandom }; break;
      default: r = { error: `unknown cmd ${msg.cmd}` };
    }
  } catch (err) { r = { error: String(err && err.stack || err) }; }
  out.write(JSON.stringify(r) + '\n');
});
