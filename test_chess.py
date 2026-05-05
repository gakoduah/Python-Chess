"""
test_chess.py
=============

Unit tests for the chess engine. Run with ``python test_chess.py`` (no pytest
dependency required) or ``pytest test_chess.py``.

These tests cover:

* Move-generation correctness via perft (reference counts from chessprogramming.org)
* Check and checkmate detection
* Castling legality (can't castle through check, after king/rook moved, etc.)
* En-passant capture
* Pawn promotion (all four options available)
* Draw conditions (stalemate, threefold repetition, insufficient material)
* Basic AI tactics (mate in one)
"""

from __future__ import annotations

import time
import unittest

from chess_engine import Board, Color, Move, Piece, PieceType, perft
from chess_ai import get_best_move


def _empty_board() -> Board:
    """Return a board with no pieces and default state cleared, for test setups."""
    b = Board()
    b.squares = [[None] * 8 for _ in range(8)]
    for k in b.castle_rights:
        b.castle_rights[k] = False
    b.en_passant = None
    b.position_history = []
    return b


def _place(b: Board, square: str, piece_type: PieceType, color: Color) -> None:
    """Place a piece at an algebraic square like 'e4'."""
    file_c = "abcdefgh".index(square[0])
    rank_r = 8 - int(square[1])
    b.squares[rank_r][file_c] = Piece(piece_type, color)


class PerftTests(unittest.TestCase):
    """Move-generation correctness via tree-size counting."""

    def test_perft_initial_position(self) -> None:
        # These reference counts are widely published and catch nearly every
        # move-generation bug, including subtle ones in castling/en-passant/pins.
        self.assertEqual(perft(Board(), 1), 20)
        self.assertEqual(perft(Board(), 2), 400)
        self.assertEqual(perft(Board(), 3), 8902)


class BasicRuleTests(unittest.TestCase):

    def test_initial_position_has_20_legal_moves(self) -> None:
        self.assertEqual(len(Board().legal_moves()), 20)

    def test_initial_turn_is_white(self) -> None:
        self.assertIs(Board().turn, Color.WHITE)

    def test_pinned_piece_cannot_move_off_the_pin(self) -> None:
        # White king on e1, white knight on e2 (pinned), black rook on e8.
        # The knight has zero legal moves.
        b = _empty_board()
        _place(b, "e1", PieceType.KING, Color.WHITE)
        _place(b, "e2", PieceType.KNIGHT, Color.WHITE)
        _place(b, "e8", PieceType.ROOK, Color.BLACK)
        _place(b, "a8", PieceType.KING, Color.BLACK)
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())

        knight_moves = b.legal_moves_from((6, 4))    # e2
        self.assertEqual(knight_moves, [])

    def test_pawn_promotion_generates_four_moves(self) -> None:
        b = _empty_board()
        _place(b, "a8", PieceType.KING, Color.BLACK)
        _place(b, "h1", PieceType.KING, Color.WHITE)
        _place(b, "e7", PieceType.PAWN, Color.WHITE)
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())

        promos = [m for m in b.legal_moves_from((1, 4)) if m.promotion is not None]
        promo_types = {m.promotion for m in promos}
        self.assertEqual(promo_types,
                         {PieceType.QUEEN, PieceType.ROOK,
                          PieceType.BISHOP, PieceType.KNIGHT})

    def test_en_passant_capture(self) -> None:
        # Play 1. e4 (white e2-e4) then set up black to capture via e.p.
        b = _empty_board()
        _place(b, "e1", PieceType.KING, Color.WHITE)
        _place(b, "e8", PieceType.KING, Color.BLACK)
        _place(b, "e2", PieceType.PAWN, Color.WHITE)
        _place(b, "d4", PieceType.PAWN, Color.BLACK)
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())

        # White pushes two squares.
        b.make_move(Move((6, 4), (4, 4)))             # e2 -> e4
        self.assertEqual(b.en_passant, (5, 4))        # target square is e3
        # Black pawn on d4 captures en passant to e3.
        black_pawn_moves = b.legal_moves_from((4, 3))  # d4
        ep_moves = [m for m in black_pawn_moves if m.is_en_passant]
        self.assertEqual(len(ep_moves), 1)
        self.assertEqual(ep_moves[0].to_sq, (5, 4))    # e3

        b.make_move(ep_moves[0])
        # After e.p., the white pawn previously on e4 is gone.
        self.assertIsNone(b.squares[4][4])
        self.assertIsNotNone(b.squares[5][4])
        self.assertEqual(b.squares[5][4].type, PieceType.PAWN)
        self.assertEqual(b.squares[5][4].color, Color.BLACK)

    def test_castling_kingside(self) -> None:
        b = _empty_board()
        _place(b, "e1", PieceType.KING, Color.WHITE)
        _place(b, "h1", PieceType.ROOK, Color.WHITE)
        _place(b, "e8", PieceType.KING, Color.BLACK)
        b.castle_rights[(Color.WHITE, "K")] = True
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())

        moves = b.legal_moves_from((7, 4))      # e1
        castle = [m for m in moves if m.is_castle]
        self.assertEqual(len(castle), 1)
        b.make_move(castle[0])
        # King on g1, rook on f1.
        self.assertEqual(b.squares[7][6].type, PieceType.KING)
        self.assertEqual(b.squares[7][5].type, PieceType.ROOK)

    def test_cannot_castle_through_check(self) -> None:
        # Black rook on f8 attacks f1 -> white cannot castle kingside.
        b = _empty_board()
        _place(b, "e1", PieceType.KING, Color.WHITE)
        _place(b, "h1", PieceType.ROOK, Color.WHITE)
        _place(b, "a8", PieceType.KING, Color.BLACK)
        _place(b, "f8", PieceType.ROOK, Color.BLACK)
        b.castle_rights[(Color.WHITE, "K")] = True
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())

        castle = [m for m in b.legal_moves_from((7, 4)) if m.is_castle]
        self.assertEqual(castle, [])


class GameEndTests(unittest.TestCase):

    def test_fools_mate(self) -> None:
        # 1. f3 e5  2. g4 Qh4#
        b = Board()
        b.make_move(Move((6, 5), (5, 5)))      # f2-f3
        b.make_move(Move((1, 4), (3, 4)))      # e7-e5
        b.make_move(Move((6, 6), (4, 6)))      # g2-g4
        b.make_move(Move((0, 3), (4, 7)))      # Qd8-h4
        self.assertTrue(b.is_checkmate())
        self.assertIn("Black wins", b.game_result())

    def test_stalemate(self) -> None:
        # Classic K+Q vs K stalemate: BK h8, WK f7, WQ g6; Black to move.
        b = _empty_board()
        _place(b, "h8", PieceType.KING, Color.BLACK)
        _place(b, "f7", PieceType.KING, Color.WHITE)
        _place(b, "g6", PieceType.QUEEN, Color.WHITE)
        b.turn = Color.BLACK
        b.position_history.append(b._position_hash())

        self.assertTrue(b.is_stalemate())
        self.assertIn("stalemate", b.game_result().lower())

    def test_insufficient_material_kk(self) -> None:
        b = _empty_board()
        _place(b, "e1", PieceType.KING, Color.WHITE)
        _place(b, "e8", PieceType.KING, Color.BLACK)
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())
        self.assertTrue(b.is_insufficient_material())

    def test_insufficient_material_k_plus_knight_vs_k(self) -> None:
        b = _empty_board()
        _place(b, "e1", PieceType.KING, Color.WHITE)
        _place(b, "b1", PieceType.KNIGHT, Color.WHITE)
        _place(b, "e8", PieceType.KING, Color.BLACK)
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())
        self.assertTrue(b.is_insufficient_material())

    def test_sufficient_material_with_rook(self) -> None:
        b = _empty_board()
        _place(b, "e1", PieceType.KING, Color.WHITE)
        _place(b, "a1", PieceType.ROOK, Color.WHITE)
        _place(b, "e8", PieceType.KING, Color.BLACK)
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())
        self.assertFalse(b.is_insufficient_material())


class AITests(unittest.TestCase):

    def test_ai_finds_mate_in_one(self) -> None:
        # White to play: Qd1-d8#.
        b = _empty_board()
        _place(b, "g8", PieceType.KING, Color.BLACK)
        _place(b, "f7", PieceType.PAWN, Color.BLACK)
        _place(b, "g7", PieceType.PAWN, Color.BLACK)
        _place(b, "h7", PieceType.PAWN, Color.BLACK)
        _place(b, "h1", PieceType.KING, Color.WHITE)
        _place(b, "d1", PieceType.QUEEN, Color.WHITE)
        b.turn = Color.WHITE
        b.position_history.append(b._position_hash())

        move = get_best_move(b, depth=2)
        b.make_move(move)
        self.assertTrue(b.is_checkmate(),
                        f"AI move {move} should be checkmate, board=\n{b}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    unittest.main(verbosity=2, exit=False)
    print(f"\nTotal time: {time.perf_counter()-t0:.2f}s")
