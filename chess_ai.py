"""
chess_ai.py
===========

A minimax AI with alpha-beta pruning for the ``chess_engine`` module.

Evaluation is material + piece-square tables (a well-established baseline
approach). Move ordering (captures first, MVV-LVA) dramatically improves
alpha-beta pruning efficiency.

Usage
-----
    >>> from chess_engine import Board
    >>> from chess_ai import get_best_move
    >>> board = Board()
    >>> move = get_best_move(board, depth=3)
    >>> board.make_move(move)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from chess_engine import Board, Color, Move, PieceType


# Centipawn values for each piece type.
PIECE_VALUES = {
    PieceType.PAWN:   100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK:   500,
    PieceType.QUEEN:  900,
    PieceType.KING:   20000,
}

# Piece-square tables, written from *White's* perspective.
# Row 0 of each table corresponds to the 8th rank (far side of the board).
# For Black, we mirror vertically when looking up values.

_PAWN_PST = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [  5,  10,  10, -20, -20,  10,  10,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0],
]
_KNIGHT_PST = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]
_BISHOP_PST = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]
_ROOK_PST = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   0,   5,   5,   0,   0,   0],
]
_QUEEN_PST = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
]
_KING_PST = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20],
]
_PST = {
    PieceType.PAWN:   _PAWN_PST,
    PieceType.KNIGHT: _KNIGHT_PST,
    PieceType.BISHOP: _BISHOP_PST,
    PieceType.ROOK:   _ROOK_PST,
    PieceType.QUEEN:  _QUEEN_PST,
    PieceType.KING:   _KING_PST,
}

# Large bound used for mate scores; comfortably above any material diff.
MATE = 100_000


def evaluate(board: Board) -> int:
    """Static evaluation from White's perspective (positive = good for White)."""
    score = 0
    for r in range(8):
        for c in range(8):
            p = board.squares[r][c]
            if p is None:
                continue
            val = PIECE_VALUES[p.type]
            if p.color is Color.WHITE:
                score += val + _PST[p.type][r][c]
            else:
                # Mirror table vertically for Black.
                score -= val + _PST[p.type][7 - r][c]
    return score


def _mvv_lva_key(board: Board, move: Move) -> int:
    """Order captures first: most-valuable-victim, least-valuable-attacker."""
    victim = board.piece_at(move.to_sq)
    if victim is None and not move.is_en_passant:
        return 0        # Quiet moves last.
    attacker = board.piece_at(move.from_sq)
    victim_val = PIECE_VALUES[victim.type] if victim else PIECE_VALUES[PieceType.PAWN]
    attacker_val = PIECE_VALUES[attacker.type] if attacker else 0
    # Higher key = searched first (we sort descending).
    return 10 * victim_val - attacker_val


class _SearchStats:
    """Lightweight counter object so callers can inspect search effort."""
    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = 0


def _negamax(board: Board, depth: int, alpha: int, beta: int,
             stats: _SearchStats) -> Tuple[int, Optional[Move]]:
    """Negamax search with alpha-beta pruning.

    Returns a ``(score, best_move)`` tuple where ``score`` is from the
    perspective of the side to move. The ``best_move`` is ``None`` at leaves
    and at terminal positions.
    """
    stats.nodes += 1

    moves = board.legal_moves()
    if not moves:
        # Terminal: checkmate (-MATE) or stalemate (0).
        if board.is_in_check(board.turn):
            # Prefer mates delivered sooner (shorter forced mates).
            return (-MATE + (100 - depth), None)
        return (0, None)

    if depth == 0:
        # Leaf: return static eval, flipped to the side-to-move perspective.
        sign = 1 if board.turn is Color.WHITE else -1
        return (sign * evaluate(board), None)

    moves.sort(key=lambda m: _mvv_lva_key(board, m), reverse=True)

    best_move = moves[0]
    best_score = -math.inf
    for move in moves:
        undo = board._apply_move(move)
        child_score, _ = _negamax(board, depth - 1, -beta, -alpha, stats)
        score = -child_score
        board._undo_move(undo)

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break        # Beta cutoff.
    return (best_score, best_move)


def get_best_move(board: Board, depth: int = 3,
                  return_stats: bool = False):
    """Return the engine's chosen move for the side to move.

    Parameters
    ----------
    board : Board
        The position to analyse. The board is restored to its original state.
    depth : int
        Search depth in plies. 2-4 is playable and fast in pure Python.
    return_stats : bool
        If True, returns ``(move, info_dict)`` instead of just the move.
    """
    stats = _SearchStats()
    score, move = _negamax(board, depth, -math.inf, math.inf, stats)
    if return_stats:
        return move, {"nodes": stats.nodes, "score_cp": score, "depth": depth}
    return move
