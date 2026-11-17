#!/usr/bin/env node

/**
 * Quantum Cheqqers Local Game Server / Test Runner
 * A self-contained script for developers to test their AIs locally.
 * Includes complete checkers rules, local Random baseline AI,
 * and remote AI caller with timeout/forfeit logic.
 *
 * Usage:
 *   node local-cheqqers-server.js --url http://localhost:8080/move [options]
 */

import http from 'http';
import https from 'https';
import readline from 'readline';

let QuantumEntity, Operations, EntanglementEngine, JointQuantumSystem;

async function initEngine() {
    const engine = await import('@quantumplayed/quantum-game-engine');
    QuantumEntity = engine.QuantumEntity;
    Operations = engine.Operations;
    EntanglementEngine = engine.EntanglementEngine;
    JointQuantumSystem = engine.JointQuantumSystem;
}

class BoardGameHelpers {
    static move(source, target) {
        const trans = [{ from: 2, to: 1, amplitude: { re: 1, im: 0 } }];
        EntanglementEngine.interactMany([source, target], Operations.SparseTransition(trans));
    }

    static split(source, target1, target2) {
        target1.amplitudes = { 0: { re: 1, im: 0 } };
        target1.entangledSystem = undefined;
        target2.amplitudes = { 0: { re: 1, im: 0 } };
        target2.entangledSystem = undefined;

        const trans = [
            { from: 4, to: 2, amplitude: { re: 0.70710678, im: 0 } },
            { from: 4, to: 1, amplitude: { re: 0.70710678, im: 0 } }
        ];
        EntanglementEngine.interactMany([source, target1, target2], Operations.SparseTransition(trans));
    }

    static merge(source1, source2, target) {
        const trans = [
            { from: 4, to: 1, amplitude: { re: 0.70710678, im: 0 } },
            { from: 4, to: 6, amplitude: { re: 0.70710678, im: 0 } },
            { from: 2, to: 1, amplitude: { re: 0.70710678, im: 0 } },
            { from: 2, to: 6, amplitude: { re: -0.70710678, im: 0 } }
        ];
        EntanglementEngine.interactMany([source1, source2, target], Operations.SparseTransition(trans));
    }

    static capture(attacker, victim, target) {
        EntanglementEngine.conditionalInteract({
            controls: [
                { entity: attacker, state: 1 },
                { entity: victim, state: 1 }
            ],
            targets: [
                { entity: attacker, operation: Operations.Shift(1) },
                { entity: target, operation: Operations.Shift(1) },
                { entity: victim, operation: Operations.Shift(1) }
            ]
        });
    }

    static captureMany(attacker, victims, target) {
        for (const victim of victims) {
            this.capture(attacker, victim, target);
        }
    }
}

// ==========================================
// 1. Quantum Cheqqers Game Constants
// ==========================================
const PieceColor = { WHITE: 0, BLACK: 1 };
const GameState = { IN_PROGRESS: 0, WHITE_WON: 1, BLACK_WON: 2, DRAW: 3 };
const GameType = { CLASSIC: 0, SUPERPOSITION: 1, ENTANGLEMENT: 2, INTERFERENCE: 3 };
const SquareState = { EMPTY: 0, OCCUPIED: 1 };

// ==========================================
// 2. Quantum Cheqqers Game Engine
// ==========================================
class CheqqersGame {
    constructor(size = 6, startRows = 2, gameType = GameType.ENTANGLEMENT) {
        this.boardSize = size;
        this.gameType = gameType;
        this.startRows = startRows;
        this.turn = PieceColor.WHITE;

        this.initParams = { type: 'standard', size, startRows, gameType };

        this.squares = [];
        this.pieceProperties = {};
        const totalSquares = Math.floor((size * size) / 2);

        for (let i = 0; i < totalSquares; i++) {
            this.squares.push(new QuantumEntity({ dimension: 2, initialState: SquareState.EMPTY }));
            this.pieceProperties[i] = null;
        }

        this.nextPieceId = 1;
        this.setupStartingPositions(startRows);
        this.lastMove = null;
        this.jumpedPieceIdsThisTurn = [];
        this.movesSinceTake = 0;
        this.multiJumpSquare = null;
        this.justCrowned = false;
        this.stateHistory = new Map();
        this.consecutiveQueenMoves = 0;
        this._isCalculatingDepth = false;
        this.disableDepthCalculation = false;
        this.forceAllMoves = false;
        this.state = GameState.IN_PROGRESS;
        this.moveHistory = [];
        this.stats = {
            white: { classicalMoves: 0, quantumMoves: 0, measurements: 0, captures: 0 },
            black: { classicalMoves: 0, quantumMoves: 0, measurements: 0, captures: 0 }
        };
    }

    clone() {
        const game = new CheqqersGame(this.boardSize, 0, this.gameType);
        game.turn = this.turn;
        game.startRows = this.startRows;
        game.nextPieceId = this.nextPieceId;
        game.jumpedPieceIdsThisTurn = [...this.jumpedPieceIdsThisTurn];
        game.movesSinceTake = this.movesSinceTake;
        game.multiJumpSquare = this.multiJumpSquare;
        game.consecutiveQueenMoves = this.consecutiveQueenMoves;
        game.state = this.state;
        game.initParams = this.initParams;
        game.moveHistory = [...this.moveHistory];

        const squareMap = new Map();
        for (let i = 0; i < this.squares.length; i++) {
            squareMap.set(this.squares[i], game.squares[i]);
            const prop = this.pieceProperties[i];
            if (prop) {
                game.pieceProperties[i] = {
                    color: prop.color,
                    crowned: prop.crowned,
                    pieceIds: [...prop.pieceIds]
                };
            } else {
                game.pieceProperties[i] = null;
            }
        }

        const visitedSystems = new Set();
        for (let i = 0; i < this.squares.length; i++) {
            const sys = this.squares[i].entangledSystem;
            if (sys && !visitedSystems.has(sys)) {
                visitedSystems.add(sys);
                const clonedEntities = sys.entities.map(e => squareMap.get(e));
                const newSys = JointQuantumSystem.create(clonedEntities);
                newSys.jointAmplitudes = { ...sys.jointAmplitudes };
            }
        }

        for (let i = 0; i < this.squares.length; i++) {
            if (!this.squares[i].entangledSystem) {
                game.squares[i].amplitudes = { ...this.squares[i].amplitudes };
            }
        }

        return game;
    }

    setupStartingPositions(startRows) {
        const totalSquares = Math.floor((this.boardSize * this.boardSize) / 2);
        const squaresPerRow = Math.floor(this.boardSize / 2);

        // Black starts at the top
        for (let i = 0; i < startRows * squaresPerRow; i++) {
            this.squares[i].apply(Operations.Shift(SquareState.OCCUPIED));
            this.pieceProperties[i] = { color: PieceColor.BLACK, crowned: false, pieceIds: [this.nextPieceId++] };
        }

        // White starts at the bottom
        for (let i = totalSquares - (startRows * squaresPerRow); i < totalSquares; i++) {
            this.squares[i].apply(Operations.Shift(SquareState.OCCUPIED));
            this.pieceProperties[i] = { color: PieceColor.WHITE, crowned: false, pieceIds: [this.nextPieceId++] };
        }
    }

    getIndex(col, row) {
        const visualRow = (this.boardSize - 1) - row;
        return visualRow * Math.floor(this.boardSize / 2) + Math.floor(col / 2);
    }

    getRowColFromIndex(index) {
        const squaresPerRow = Math.floor(this.boardSize / 2);
        const visualRow = Math.floor(index / squaresPerRow);
        const row = (this.boardSize - 1) - visualRow;

        const isRowEven = row % 2 === 0;
        const colInRow = index % squaresPerRow;
        const col = isRowEven ? (colInRow * 2) : (colInRow * 2) + 1;
        return { col, row };
    }

    getPossibleMoves(options = false) {
        const skipDepthCalculation = typeof options === 'boolean' ? options : !!options.skipDepthCalculation;
        const ignoreTurn = typeof options === 'object' && !!options.ignoreTurn;
        const ignoreForcedCaptures = (typeof options === 'object' && !!options.ignoreForcedCaptures) || this.forceAllMoves;

        const takeMoves = (this.multiJumpSquare !== null)
            ? this.findMoves(true, this.multiJumpSquare, ignoreTurn)
            : this.getTakeMoves(ignoreTurn);

        const standardMoves = this.getStandardMoves(ignoreTurn);

        if (ignoreForcedCaptures) {
            let all = [...standardMoves, ...takeMoves];
            if (!ignoreTurn) {
                if (this.gameType === GameType.CLASSIC) {
                    return all.filter(m => !('to_index1' in m) && !('from_index1' in m));
                }
                if (this.gameType === GameType.SUPERPOSITION || this.gameType === GameType.ENTANGLEMENT) {
                    return all.filter(m => !('from_index1' in m));
                }
            }
            return all;
        }

        if (takeMoves.length > 0) {
            if (!skipDepthCalculation && !this.forceAllMoves && !this._isCalculatingDepth && !this.disableDepthCalculation && takeMoves.length > 1) {
                this._isCalculatingDepth = true;
                try {
                    const currentState = this.getSimulationState();
                    const moveDepths = takeMoves.map(m => this.calculateMaxCaptureDepth(currentState, m));
                    const maxDepth = Math.max(...moveDepths);
                    return takeMoves.filter((_, i) => moveDepths[i] === maxDepth);
                } finally {
                    this._isCalculatingDepth = false;
                }
            }
            return takeMoves;
        }

        if (this.gameType === GameType.CLASSIC) {
            return standardMoves.filter(m => !('to_index1' in m) && !('from_index1' in m));
        }
        if (this.gameType === GameType.SUPERPOSITION || this.gameType === GameType.ENTANGLEMENT) {
            return standardMoves.filter(m => !('from_index1' in m));
        }

        return standardMoves;
    }

    getTakeMoves(ignoreTurn) {
        return this.findMoves(true, undefined, ignoreTurn);
    }

    getStandardMoves(ignoreTurn) {
        const moves = this.findMoves(false, undefined, ignoreTurn);
        const splitsAndMerges = this.findSplitAndMergeMoves(moves);
        return [...moves, ...splitsAndMerges];
    }

    findMoves(isTake, restrictToSquareIndex, ignoreTurn) {
        const moves = [];
        const squareProbs = this.squares.map(sq => sq.getProbabilities());

        for (let i = 0; i < this.squares.length; i++) {
            if (restrictToSquareIndex !== undefined && i !== restrictToSquareIndex) continue;

            const probs = squareProbs[i];
            const isOccupiedProb = probs[SquareState.OCCUPIED] || 0;
            if (isOccupiedProb === 0) continue;

            const pieceProp = this.pieceProperties[i];
            if (!pieceProp) continue;
            if (!ignoreTurn && pieceProp.color !== this.turn) continue;

            const forwardDirection = pieceProp.color === PieceColor.WHITE ? 1 : -1;
            const canMoveBackwards = pieceProp.crowned;
            const { col: startX, row: startY } = this.getRowColFromIndex(i);

            const directions = [
                { dx: -1, dy: 1 }, { dx: 1, dy: 1 },
                { dx: -1, dy: -1 }, { dx: 1, dy: -1 }
            ];

            for (const { dx, dy } of directions) {
                if (!isTake && !canMoveBackwards && dy !== forwardDirection) continue;

                let curX = startX;
                let curY = startY;
                let foundEnemySquareIndex = -1;

                while (true) {
                    curX += dx;
                    curY += dy;

                    if (curX < 0 || curX >= this.boardSize || curY < 0 || curY >= this.boardSize) break;

                    const curIndex = this.getIndex(curX, curY);
                    const curProbs = squareProbs[curIndex];
                    const curOccupiedProb = curProbs[SquareState.OCCUPIED] || 0;
                    const isEmpty = curOccupiedProb < 1;
                    const isOccupied = curOccupiedProb > 0;

                    const curPieceProp = this.pieceProperties[curIndex];
                    const isEnemy = isOccupied && curPieceProp && curPieceProp.color !== pieceProp.color;
                    const isFriendly = isOccupied && curPieceProp && curPieceProp.color === pieceProp.color;
                    const hasEmptyChance = (curProbs[SquareState.EMPTY] || 0) > 0.001;

                    if (isTake) {
                        if (foundEnemySquareIndex === -1) {
                            if (isFriendly) break;
                            if (isEnemy) {
                                const sharesId = curPieceProp.pieceIds?.some(id => this.jumpedPieceIdsThisTurn.includes(id));
                                if (sharesId) break;
                                if (!this.canCoexist(i, curIndex)) break;
                                foundEnemySquareIndex = curIndex;
                            } else if (isEmpty) {
                                if (!canMoveBackwards) break;
                            }
                        } else {
                            if (hasEmptyChance) {
                                moves.push({ is_take_move: true, from_index: i, to_index: curIndex });
                                if (!canMoveBackwards) break;
                            } else {
                                break;
                            }
                        }
                    } else {
                        if (hasEmptyChance) {
                            moves.push({ is_take_move: false, from_index: i, to_index: curIndex });
                            if (!canMoveBackwards) break;
                            // A slide never passes an enemy piece: a move past an enemy must be a capture.
                            if (isEnemy) break;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        return moves;
    }

    canCoexist(sqIndex1, sqIndex2) {
        const sq1 = this.squares[sqIndex1];
        const sq2 = this.squares[sqIndex2];
        if (sq1.entangledSystem && sq1.entangledSystem === sq2.entangledSystem) {
            return !sq1.entangledSystem.areMutuallyExclusive([
                { entity: sq1, state: SquareState.OCCUPIED },
                { entity: sq2, state: SquareState.OCCUPIED }
            ]);
        }
        return true;
    }

    getSimulationState() {
        return {
            board_size: this.boardSize,
            game_type: this.gameType,
            turn: this.turn,
            next_piece_id: this.nextPieceId,
            jumped_piece_ids_this_turn: [...this.jumpedPieceIdsThisTurn],
            multi_jump_square: this.multiJumpSquare,
            piece_map: this.getMinimalPieceMap()
        };
    }

    calculateMaxCaptureDepth(state, initialMove) {
        const game = new CheqqersGame(state.board_size, 0, state.game_type);
        game.disableDepthCalculation = true;
        game.turn = state.turn;
        game.nextPieceId = state.next_piece_id;
        game.jumpedPieceIdsThisTurn = [...state.jumped_piece_ids_this_turn];
        game.multiJumpSquare = state.multi_jump_square;

        for (const [idxStr, prop] of Object.entries(state.piece_map)) {
            const idx = parseInt(idxStr);
            if (prop) {
                game.squares[idx].apply(Operations.Shift(SquareState.OCCUPIED));
                game.pieceProperties[idx] = { color: prop.color, crowned: prop.crowned, pieceIds: [...prop.pieceIds] };
            }
        }

        const moves = game.getPossibleMoves(true);
        const moveIdx = moves.findIndex(m =>
            'from_index' in m && !('to_index1' in m) && m.from_index === initialMove.from_index && m.to_index === initialMove.to_index
        );

        if (moveIdx === -1) return 0;
        game.applyMove(moveIdx);
        if (game.multiJumpSquare === null) return 1;

        const nextMoves = game.getPossibleMoves(true);
        if (nextMoves.length === 0) return 1;

        const nextState = game.getSimulationState();
        const nextDepths = nextMoves.map(nm => this.calculateMaxCaptureDepth(nextState, nm));
        return 1 + Math.max(0, ...nextDepths);
    }

    getMinimalPieceMap() {
        const map = new Array(this.squares.length).fill(null);
        for (let i = 0; i < this.squares.length; i++) {
            const probs = this.squares[i].getProbabilities();
            if ((probs[SquareState.OCCUPIED] || 0) > 0.001 && this.pieceProperties[i]) {
                map[i] = {
                    color: this.pieceProperties[i].color,
                    crowned: this.pieceProperties[i].crowned,
                    pieceIds: this.pieceProperties[i].pieceIds
                };
            }
        }
        return map;
    }

    findSplitAndMergeMoves(classicalMoves) {
        const splitMoves = [];
        const mergeMoves = [];
        const moves = classicalMoves;

        for (let i = 0; i < moves.length; i++) {
            for (let j = i + 1; j < moves.length; j++) {
                if (moves[i].from_index === moves[j].from_index && moves[i].to_index !== moves[j].to_index) {
                    if (!splitMoves.some(sm => sm.from_index === moves[i].from_index &&
                        ((sm.to_index1 === moves[i].to_index && sm.to_index2 === moves[j].to_index) ||
                            (sm.to_index1 === moves[j].to_index && sm.to_index2 === moves[i].to_index)))) {
                        splitMoves.push({ from_index: moves[i].from_index, to_index1: moves[i].to_index, to_index2: moves[j].to_index });
                    }
                }
            }
        }

        for (let i = 0; i < moves.length; i++) {
            for (let j = i + 1; j < moves.length; j++) {
                if (moves[i].to_index === moves[j].to_index && moves[i].from_index !== moves[j].from_index) {
                    const occ1 = this.squares[moves[i].from_index].getProbabilities()[SquareState.OCCUPIED] || 0;
                    const occ2 = this.squares[moves[j].from_index].getProbabilities()[SquareState.OCCUPIED] || 0;
                    if (occ1 > 0.99 && occ2 > 0.99) continue;

                    if (!mergeMoves.some(mm => mm.to_index === moves[i].to_index &&
                        ((mm.from_index1 === moves[i].from_index && mm.from_index2 === moves[j].from_index) ||
                            (mm.from_index1 === moves[j].from_index && mm.from_index2 === moves[i].from_index)))) {
                        mergeMoves.push({ from_index1: moves[i].from_index, from_index2: moves[j].from_index, to_index: moves[i].to_index });
                    }
                }
            }
        }
        return [...splitMoves, ...mergeMoves];
    }

    getInvolvedSquares(move) {
        const result = [];
        if ('to_index1' in move && 'to_index2' in move) {
            result.push({ index: move.to_index1, role: 'landing' }, { index: move.to_index2, role: 'landing' });
        } else if ('from_index1' in move && 'from_index2' in move) {
            result.push({ index: move.to_index, role: 'landing' });
        } else {
            const cm = move;
            result.push({ index: cm.from_index, role: 'source' });
            result.push({ index: cm.to_index, role: 'landing' });
            // Squares between start and landing that may hold a piece. On a capture, the first one
            // is the piece being jumped ('jumped'). Every other one, and every such square on a
            // slide, is passed over ('transit'): it is measured and must turn out empty.
            const { col: fC, row: fR } = this.getRowColFromIndex(cm.from_index);
            const { col: tC, row: tR } = this.getRowColFromIndex(cm.to_index);
            const dx = tC > fC ? 1 : -1;
            const dy = tR > fR ? 1 : -1;
            let jumpedFound = !cm.is_take_move;
            for (let step = 1; step < Math.abs(tC - fC); step++) {
                const checkIdx = this.getIndex(fC + (step * dx), fR + (step * dy));
                const emptyProb = this.squares[checkIdx].getProbabilities()[SquareState.EMPTY] || 0;
                if (emptyProb < 0.999) {
                    result.push({ index: checkIdx, role: jumpedFound ? 'transit' : 'jumped' });
                    jumpedFound = true;
                }
            }
        }
        return result;
    }

    static encodeMoveNotation(move) {
        const sq = (idx) => String(idx + 1);
        if ('is_take_move' in move && move.is_take_move) return `${sq(move.from_index)}x${sq(move.to_index)}`;
        if ('to_index1' in move) {
            const sorted = [move.to_index1, move.to_index2].sort((a, b) => a - b);
            return `${sq(move.from_index)}-${sq(sorted[0])}^${sq(sorted[1])}`;
        }
        if ('from_index1' in move) {
            const sorted = [move.from_index1, move.from_index2].sort((a, b) => a - b);
            return `${sq(sorted[0])}^${sq(sorted[1])}-${sq(move.to_index)}`;
        }
        return `${sq(move.from_index)}-${sq(move.to_index)}`;
    }

    getBoardStateHash() {
        const parts = [`T:${this.turn}`];
        for (let i = 0; i < this.squares.length; i++) {
            const occ = this.squares[i].getProbabilities()[SquareState.OCCUPIED] || 0;
            if (occ > 0.0001) {
                const props = this.pieceProperties[i];
                parts.push(`${i}${props?.color === PieceColor.WHITE ? 'W' : 'B'}${props?.crowned ? 'K' : 'M'}${occ > 0.999 ? '1' : occ.toFixed(3)}`);
            }
        }
        return parts.join('|');
    }

    applyMove(moveOrIndex, optionsOrReplay) {
        let move;
        let replayForcedMeasurements;

        if (typeof moveOrIndex === 'number') {
            const options = (optionsOrReplay && !Array.isArray(optionsOrReplay)) ? optionsOrReplay : {};
            const moves = this.getPossibleMoves(options);
            move = moves[moveOrIndex];
            if (!move) return 0;
            replayForcedMeasurements = Array.isArray(optionsOrReplay) ? optionsOrReplay : undefined;
        } else {
            move = moveOrIndex;
            replayForcedMeasurements = Array.isArray(optionsOrReplay) ? optionsOrReplay : undefined;
        }

        const wasInMultiJump = this.multiJumpSquare !== null;

        const fromIdxBefore = ('from_index' in move) ? move.from_index : (('from_index1' in move) ? move.from_index1 : -1);
        const movedPiece = fromIdxBefore !== -1 ? this.pieceProperties[fromIdxBefore] : null;
        const wasCrownedBefore = movedPiece ? movedPiece.crowned === true : false;
        const wasTakeMove = 'is_take_move' in move && move.is_take_move === true;

        const forcedMeasurementOutcomes = [];
        let replayMeasurementIndex = 0;

        try {
            this.justCrowned = false;
            const ignoreTurn = typeof optionsOrReplay === 'object' && !Array.isArray(optionsOrReplay) && !!optionsOrReplay.ignoreTurn;
            const movingColor = (movedPiece && ignoreTurn) ? movedPiece.color : this.turn;
            const activePlayer = movingColor === PieceColor.WHITE ? 'white' : 'black';
            const involvement = this.getInvolvedSquares(move);

            if ('to_index1' in move || 'from_index1' in move) {
                this.stats[activePlayer].quantumMoves++;
            } else {
                this.stats[activePlayer].classicalMoves++;
            }

            let moveFailed = false;

            for (const { index: sqIdx, role } of involvement) {
                const probs = this.squares[sqIdx].getProbabilities();
                const occ = probs[SquareState.OCCUPIED] || 0;
                const isSuperposition = occ > 0.001 && occ < 0.999;

                let shouldMeasure = this.gameType === GameType.SUPERPOSITION || role === 'landing' || role === 'transit';
                if (this.gameType === GameType.SUPERPOSITION && role === 'source') shouldMeasure = true;

                if (isSuperposition && shouldMeasure) {
                    const forced = replayForcedMeasurements?.[replayMeasurementIndex];
                    const result = (forced !== undefined) ? this.measureSquare(sqIdx, forced) : this.measureSquare(sqIdx);
                    if (forced !== undefined) replayMeasurementIndex++;
                    forcedMeasurementOutcomes.push(result);
                    this.stats[activePlayer].measurements++;
                    if ((role === 'landing' || role === 'transit') && result === SquareState.OCCUPIED) { moveFailed = true; break; }
                }
                // A landing or passed-over square that an earlier measurement in this move made certainly
                // occupied blocks the move too (e.g. the second target of a split). Without this check the
                // square would not be measured again, and the move would overwrite the piece standing there.
                if ((role === 'landing' || role === 'transit') && (this.squares[sqIdx].getProbabilities()[SquareState.OCCUPIED] || 0) >= 0.999) { moveFailed = true; break; }
                // The piece to be jumped is not there (measured away, or presence below 0.001): the jump fails.
                if (role === 'jumped' && (this.squares[sqIdx].getProbabilities()[SquareState.OCCUPIED] || 0) <= 0.001) { moveFailed = true; break; }
            }

            if (!moveFailed) {
                if (this.gameType === GameType.CLASSIC && (('to_index1' in move) || ('from_index1' in move))) {
                    this.gameType = GameType.SUPERPOSITION;
                    if (this.initParams) {
                        this.initParams.gameType = this.gameType;
                    }
                }

                if ('is_take_move' in move && move.is_take_move) this.applyTakeMove(move);
                else if ('to_index1' in move) this.applySplitMove(move);
                else if ('from_index1' in move) this.applyMergeMove(move);
                else this.applyClassicalMove(move);

                this.applyPhasePass();
                this.cleanupPieceProperties();
                this.disentangleCollapsedEntities();

                if (ignoreTurn) {
                    this.multiJumpSquare = null;
                    this.jumpedPieceIdsThisTurn = [];
                } else if (!wasTakeMove) {
                    if (wasCrownedBefore) this.consecutiveQueenMoves++; else this.consecutiveQueenMoves = 0;
                    this.turn = this.turn === PieceColor.WHITE ? PieceColor.BLACK : PieceColor.WHITE;
                    this.multiJumpSquare = null;
                    this.jumpedPieceIdsThisTurn = [];
                    this.movesSinceTake++;
                } else {
                    this.movesSinceTake = 0; this.consecutiveQueenMoves = 0; this.stateHistory.clear();
                    const landing = move.to_index;
                    if (!this.justCrowned && this.findMoves(true, landing).length > 0) this.multiJumpSquare = landing;
                    else { this.turn = this.turn === PieceColor.WHITE ? PieceColor.BLACK : PieceColor.WHITE; this.multiJumpSquare = null; this.jumpedPieceIdsThisTurn = []; }
                }
                this.checkWinStates();
                this.lastMove = move;
            } else {
                this.turn = this.turn === PieceColor.WHITE ? PieceColor.BLACK : PieceColor.WHITE;
                this.multiJumpSquare = null; this.jumpedPieceIdsThisTurn = [];
                this.cleanupPieceProperties(); this.disentangleCollapsedEntities(); this.checkWinStates();
            }
        } catch (e) { console.error("Move application error", e); }

        const notation = CheqqersGame.encodeMoveNotation(move);
        const mString = forcedMeasurementOutcomes.map(o => `m${o}`).join('');
        if (wasInMultiJump && this.moveHistory.length > 0) {
            const last = this.moveHistory[this.moveHistory.length - 1];
            const parts = notation.split('x');
            this.moveHistory[this.moveHistory.length - 1] = `${last}x${parts.length > 1 ? parts[1] : parts[0]}${mString}`;
        } else {
            this.moveHistory.push(`${notation}${mString}`);
        }
        return forcedMeasurementOutcomes.length;
    }

    measureSquare(index, forced) {
        const sq = this.squares[index];
        if (!sq) return SquareState.EMPTY;
        const res = sq.entangledSystem ? sq.entangledSystem.measureTarget(sq, forced) : sq.measure(forced);
        this.disentangleCollapsedEntities();
        this.cleanupPieceProperties();
        return res;
    }

    disentangleCollapsedEntities() {
        const processed = new Set();
        for (const sq of this.squares) {
            const sys = sq.entangledSystem;
            if (!sys || processed.has(sys)) continue;
            processed.add(sys);
            const definite = new Map();
            let allD = true;
            for (const ent of sys.entities) {
                const probs = sys.getTargetProbabilities(ent);
                let d = null;
                for (const [s, p] of Object.entries(probs)) { if (p > 0.999) { d = parseInt(s); break; } }
                if (d !== null) definite.set(ent, d); else allD = false;
            }
            if (allD) {
                for (const ent of sys.entities) { const s = definite.get(ent); ent.entangledSystem = undefined; ent.amplitudes = { [s]: { re: 1, im: 0 } }; }
            } else {
                for (const [ent, s] of definite) {
                    ent.entangledSystem = undefined; ent.amplitudes = { [s]: { re: 1, im: 0 } };
                    const idx = sys.entities.indexOf(ent);
                    if (idx !== -1) this.factorOutEntity(sys, idx, s);
                }
            }
        }
    }

    factorOutEntity(sys, idx, state) {
        const remaining = sys.entities.filter((_, i) => i !== idx);
        if (remaining.length === 0) return;
        if (remaining.length === 1) {
            const ent = remaining[0];
            const rIdx = idx === 0 ? 1 : 0;
            const amps = {};
            for (const [k, a] of Object.entries(sys.jointAmplitudes)) {
                const amplitude = a;
                const s = sys.decomposeState(parseInt(k));
                if (s[idx] === state) {
                    const rs = s[rIdx];
                    if (!amps[rs]) amps[rs] = { re: 0, im: 0 };
                    amps[rs].re += amplitude.re; amps[rs].im += amplitude.im;
                }
            }
            ent.entangledSystem = undefined; ent.amplitudes = amps;
            return;
        }
        const dims = remaining.map(e => e.dimension);
        const amps = {};
        for (const [k, a] of Object.entries(sys.jointAmplitudes)) {
            const amplitude = a;
            const s = sys.decomposeState(parseInt(k));
            if (s[idx] !== state) continue;
            const ns = s.filter((_, i) => i !== idx);
            let nk = 0; for (let i = 0; i < ns.length; i++) nk = (nk * dims[i]) + ns[i];
            if (!amps[nk]) amps[nk] = { re: 0, im: 0 };
            amps[nk].re += amplitude.re; amps[nk].im += amplitude.im;
        }
        sys.entities.splice(idx, 1); sys.dimensions.splice(idx, 1);
        sys.totalDimension = sys.dimensions.reduce((a, b) => a * b, 1);
        sys.jointAmplitudes = amps;
        for (const ent of sys.entities) ent.entangledSystem = sys;
    }

    cleanupPieceProperties() {
        for (let i = 0; i < this.squares.length; i++) { if ((this.squares[i].getProbabilities()[SquareState.OCCUPIED] || 0) <= 0.0001) this.pieceProperties[i] = null; }
    }

    movePieceProperties(f, t) {
        if (!this.pieceProperties[f]) return;
        const cur = this.pieceProperties[f];
        this.pieceProperties[t] = { ...cur };
        const { row } = this.getRowColFromIndex(t);
        if ((cur.color === PieceColor.WHITE && row === this.boardSize - 1) || (cur.color === PieceColor.BLACK && row === 0)) {
            if (!cur.crowned) { this.pieceProperties[t].crowned = true; this.justCrowned = true; }
        }
    }

    executeClassicalOverride(f, taken, t) {
        this.squares[f].amplitudes = { [SquareState.EMPTY]: { re: 1, im: 0 } };
        for (const idx of taken) { this.squares[idx].amplitudes = { [SquareState.EMPTY]: { re: 1, im: 0 } }; this.pieceProperties[idx] = null; }
        this.squares[t].amplitudes = { [SquareState.OCCUPIED]: { re: 1, im: 0 } };
        this.movePieceProperties(f, t);
        if (f !== t) this.pieceProperties[f] = null;
    }

    applyClassicalMove(m) {
        if (this.gameType === GameType.CLASSIC) { this.executeClassicalOverride(m.from_index, [], m.to_index); return; }
        const f = this.squares[m.from_index], t = this.squares[m.to_index];
        if (this.gameType === GameType.SUPERPOSITION && (f.getProbabilities()[SquareState.OCCUPIED] || 0) < 0.999) this.measureSquare(m.from_index);
        BoardGameHelpers.move(f, t);
        if ((t.getProbabilities()[SquareState.OCCUPIED] || 0) > 0.0001) this.movePieceProperties(m.from_index, m.to_index);
    }

    applyTakeMove(m) {
        const { col: fC, row: fR } = this.getRowColFromIndex(m.from_index);
        const { col: tC, row: tR } = this.getRowColFromIndex(m.to_index);
        const dx = tC > fC ? 1 : -1, dy = tR > fR ? 1 : -1;
        const taken = [];
        for (let s = 1; s < Math.abs(tC - fC); s++) {
            const idx = this.getIndex(fC + (s * dx), fR + (s * dy));
            if ((this.squares[idx].getProbabilities()[SquareState.OCCUPIED] || 0) > 0.001) taken.push(idx);
        }
        if (taken.length === 0) return; // nothing to capture; applyMove already treats this as a failed jump

        const active = this.turn === PieceColor.WHITE ? 'white' : 'black';
        for (const idx of taken) {
            const p = this.pieceProperties[idx];
            if (p) { this.jumpedPieceIdsThisTurn.push(...p.pieceIds); this.stats[active].captures++; }
        }

        if (this.gameType === GameType.CLASSIC) { this.executeClassicalOverride(m.from_index, taken, m.to_index); return; }
        if (this.gameType === GameType.SUPERPOSITION) this.measureSquare(m.from_index);

        const f = this.squares[m.from_index], t = this.squares[m.to_index];
        const victimSquares = taken.map(idx => this.squares[idx]);
        BoardGameHelpers.captureMany(f, victimSquares, t);
        if ((t.getProbabilities()[SquareState.OCCUPIED] || 0) > 0.0001) {
            this.movePieceProperties(m.from_index, m.to_index);
            if ((f.getProbabilities()[SquareState.OCCUPIED] || 0) < 0.001) this.pieceProperties[m.from_index] = null;
            for (const idx of taken) { if ((this.squares[idx].getProbabilities()[SquareState.OCCUPIED] || 0) < 0.001) this.pieceProperties[idx] = null; }
        }
    }

    applySplitMove(m) {
        if (this.gameType === GameType.CLASSIC) {
            this.gameType = GameType.SUPERPOSITION;
            if (this.initParams) this.initParams.gameType = this.gameType;
        }

        const sqFrom = this.squares[m.from_index];
        const sqTo1 = this.squares[m.to_index1];
        const sqTo2 = this.squares[m.to_index2];

        BoardGameHelpers.split(sqFrom, sqTo1, sqTo2);

        this.movePieceProperties(m.from_index, m.to_index1);
        this.movePieceProperties(m.from_index, m.to_index2);
        this.pieceProperties[m.from_index] = null;
    }

    applyMergeMove(m) {
        const f1 = this.squares[m.from_index1], f2 = this.squares[m.from_index2], t = this.squares[m.to_index];
        BoardGameHelpers.merge(f1, f2, t);
        this.movePieceProperties(m.from_index1, m.to_index);
        const p = this.pieceProperties[m.to_index];
        if (p) p.pieceIds = Array.from(new Set([...(this.pieceProperties[m.from_index1]?.pieceIds || []), ...(this.pieceProperties[m.from_index2]?.pieceIds || [])]));
        this.pieceProperties[m.from_index1] = null; this.pieceProperties[m.from_index2] = null;
    }

    applyPhasePass() {
        if (this.gameType === GameType.INTERFERENCE) {
            for (const sq of this.squares) EntanglementEngine.conditionalInteract({ controls: [{ entity: sq, state: 1 }], targets: [{ entity: sq, operation: Operations.Phase(Math.PI / 4) }] });
        }
    }

    checkWinStates() {
        let wProb = 0, bProb = 0, wQ = 0, bQ = 0;
        for (let i = 0; i < this.squares.length; i++) {
            const p = this.squares[i].getProbabilities()[SquareState.OCCUPIED] || 0, props = this.pieceProperties[i];
            if (props && p > 0.0001) {
                if (props.color === PieceColor.WHITE) { wProb += p; if (props.crowned) wQ++; }
                else { bProb += p; if (props.crowned) bQ++; }
            }
        }
        if (wProb < 0.0001 && bProb < 0.0001) { this.state = GameState.DRAW; return; }
        if (wProb < 0.0001) { this.state = GameState.BLACK_WON; return; }
        if (bProb < 0.0001) { this.state = GameState.WHITE_WON; return; }
        const hash = this.getBoardStateHash(), count = (this.stateHistory.get(hash) || 0) + 1; this.stateHistory.set(hash, count);
        if (this.stateHistory.size > 500) {
            const firstKey = this.stateHistory.keys().next().value;
            if (firstKey !== undefined) this.stateHistory.delete(firstKey);
        }
        if (count >= 3) { this.state = GameState.DRAW; return; }
        if (wQ > 0 && bQ > 0 && this.consecutiveQueenMoves >= 50) { this.state = GameState.DRAW; return; }
        if (this.movesSinceTake >= 100) { this.state = GameState.DRAW; return; }
        if (this.getPossibleMoves(true).length === 0) this.state = this.turn === PieceColor.WHITE ? GameState.BLACK_WON : GameState.WHITE_WON;
    }

    toGameStateObject(skipMoves = false) {
        const piece_map = {}, chances = {};
        for (let i = 0; i < this.squares.length; i++) {
            const occ = this.squares[i].getProbabilities()[SquareState.OCCUPIED] || 0;
            if (occ > 0.0001 && this.pieceProperties[i]) {
                const p = this.pieceProperties[i];
                piece_map[i] = { color: p.color, crowned: p.crowned, pieceIds: p.pieceIds };
                chances[i] = parseFloat((occ > 0.999 ? 1.0 : occ).toFixed(10));
            } else { chances[i] = 0; piece_map[i] = null; }
        }
        const quantum_states = [];
        const reported = new Set();
        const getS = (s, idx) => { if (s === 0) return "E"; const p = this.pieceProperties[idx]; if (!p) return "?"; return p.color === PieceColor.WHITE ? (p.crowned ? "WQ" : "W") : (p.crowned ? "BQ" : "B"); };
        for (let i = 0; i < this.squares.length; i++) {
            const sq = this.squares[i], sys = sq.entangledSystem;
            if (sys) {
                if (!reported.has(sys)) {
                    reported.add(sys);
                    const idxs = sys.entities.map(e => this.squares.indexOf(e)), amps = [];
                    for (const [k, a] of Object.entries(sys.jointAmplitudes)) {
                        const amplitude = a;
                        if (amplitude.re === 0 && amplitude.im === 0) continue;
                        const s = sys.decomposeState(parseInt(k));
                        amps.push({ state: s.map((state, j) => getS(state, idxs[j])).join("⊗"), re: amplitude.re, im: amplitude.im });
                    }
                    quantum_states.push({ type: 'entangled', squares: idxs, amplitudes: amps });
                }
            } else {
                const amps = []; let nt = false;
                for (const [k, a] of Object.entries(sq.amplitudes)) {
                    const amplitude = a;
                    if (amplitude.re === 0 && amplitude.im === 0) continue;
                    const s = parseInt(k);
                    if (s !== 0) nt = true;
                    amps.push({ state: getS(s, i), re: amplitude.re, im: amplitude.im });
                }
                if (nt) quantum_states.push({ type: 'independent', squares: [i], amplitudes: amps });
            }
        }
        const possible = skipMoves ? [] : this.getPossibleMoves();
        const merges = {}, splits = {};
        for (const m of possible) {
            if ('from_index1' in m && 'from_index2' in m) {
                const mergeMove = m;
                if (!merges[mergeMove.from_index1]) merges[mergeMove.from_index1] = [];
                if (!merges[mergeMove.from_index2]) merges[mergeMove.from_index2] = [];
                merges[mergeMove.from_index1].push(mergeMove.from_index2);
                merges[mergeMove.from_index2].push(mergeMove.from_index1);
            }
            else if ('to_index1' in m && 'to_index2' in m) {
                const splitMove = m;
                if (!splits[splitMove.from_index]) splits[splitMove.from_index] = [];
                splits[splitMove.from_index].push(splitMove.to_index1, splitMove.to_index2);
            }
        }
        return {
            board_size: this.boardSize, turn: this.turn, game_state: this.state, game_type: this.gameType,
            piece_map, next_piece_id: this.nextPieceId,
            multi_jump_square: this.multiJumpSquare, jumped_piece_ids_this_turn: [...this.jumpedPieceIdsThisTurn],
            possible_moves: possible, merge_partners: merges, split_destinations: splits, chances,
            move_history: [...this.moveHistory], init_params: JSON.parse(JSON.stringify(this.initParams)), quantum_states, stats: {
                white: { ...this.stats.white },
                black: { ...this.stats.black }
            },
            last_move: this.lastMove
        };
    }
}

// ==========================================
// 3. Serialization Helpers
// ==========================================
function moveToArray(move) {
    if ('to_index1' in move && 'to_index2' in move) {
        return [move.from_index, move.to_index1, move.to_index2];
    } else if ('from_index1' in move && 'from_index2' in move) {
        return [move.from_index1, move.from_index2, move.to_index];
    } else {
        return [move.from_index, move.to_index];
    }
}

function serializeBoardState(gameState) {
    const totalSquares = Math.floor((gameState.board_size * gameState.board_size) / 2);
    const board = [];
    for (let i = 0; i < totalSquares; i++) {
        const chance = gameState.chances[i] || 0;
        const piece = gameState.piece_map[i];
        if (chance === 0 || !piece) {
            board.push({ piece: null, crowned: false, probability: 0 });
        } else {
            const isWhite = piece.color === PieceColor.WHITE;
            board.push({
                piece: isWhite ? 'white' : 'black',
                crowned: piece.crowned,
                probability: chance
            });
        }
    }
    return board;
}

// ==========================================
// 4. Local Random AI Implementation
export { CheqqersGame, initEngine, serializeBoardState, moveToArray, PieceColor, GameState, GameType, SquareState, JointQuantumSystem };
