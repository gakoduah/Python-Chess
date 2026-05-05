"""
chess_engine.py
===============

A pure-Python chess engine implementing full 8x8 chess rules:

* All six piece types (King, Queen, Rook, Bishop, Knight, Pawn)
* Castling (kingside and queenside) with full legality checks
* En-passant capture
* Pawn promotion
* Check, checkmate, stalemate detection
* Draw detection: threefold repetition, 50-move rule, insufficient material

The engine has **no external dependencies** and is decoupled from any UI,
so it can be unit-tested, used in a notebook, or driven by an AI search.

Coordinate convention
---------------------
Squares are represented as ``(row, col)`` tuples.
``row=0`` is rank 8 (Black's back rank); ``row=7`` is rank 1 (White's back rank).
``col=0`` is file a; ``col=7`` is file h.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

Square = Tuple[int, int]


# ---------------------------------------------------------------------------
# Enums and simple data types
# ---------------------------------------------------------------------------
class Color(Enum):
    WHITE = 0
    BLACK = 1

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class PieceType(Enum):
    PAWN = "P"
    KNIGHT = "N"
    BISHOP = "B"
    ROOK = "R"
    QUEEN = "Q"
    KING = "K"


# Unicode symbols for display.
PIECE_SYMBOLS: Dict[Tuple[Color, PieceType], str] = {
    (Color.WHITE, PieceType.KING):   "\u2654",
    (Color.WHITE, PieceType.QUEEN):  "\u2655",
    (Color.WHITE, PieceType.ROOK):   "\u2656",
    (Color.WHITE, PieceType.BISHOP): "\u2657",
    (Color.WHITE, PieceType.KNIGHT): "\u2658",
    (Color.WHITE, PieceType.PAWN):   "\u2659",
    (Color.BLACK, PieceType.KING):   "\u265A",
    (Color.BLACK, PieceType.QUEEN):  "\u265B",
    (Color.BLACK, PieceType.ROOK):   "\u265C",
    (Color.BLACK, PieceType.BISHOP): "\u265D",
    (Color.BLACK, PieceType.KNIGHT): "\u265E",
    (Color.BLACK, PieceType.PAWN):   "\u265F",
}


@dataclass(frozen=True)
class Piece:
    type: PieceType
    color: Color

    def symbol(self) -> str:
        return PIECE_SYMBOLS[(self.color, self.type)]

    def __repr__(self) -> str:
        c = self.type.value
        return c.upper() if self.color is Color.WHITE else c.lower()


@dataclass(frozen=True)
class Move:
    """An immutable description of a move.

    ``promotion`` is set only for pawn promotions.
    ``is_castle`` / ``is_en_passant`` flag special moves so the engine can
    carry out the corresponding side effects.
    """
    from_sq: Square
    to_sq: Square
    promotion: Optional[PieceType] = None
    is_castle: bool = False
    is_en_passant: bool = False

    def uci(self) -> str:
        s = _square_name(self.from_sq) + _square_name(self.to_sq)
        if self.promotion:
            s += self.promotion.value.lower()
        return s

    def __repr__(self) -> str:
        return self.uci()


def _square_name(sq: Square) -> str:
    r, c = sq
    return "abcdefgh"[c] + str(8 - r)


# Movement vectors reused by the generator.
_KNIGHT_DELTAS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
_KING_DELTAS = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
_BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_QUEEN_DIRS = _BISHOP_DIRS + _ROOK_DIRS


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------
class Board:
    """Full chess board with move generation and legality checking."""

    def __init__(self) -> None:
        self.squares: List[List[Optional[Piece]]] = [[None] * 8 for _ in range(8)]
        self.turn: Color = Color.WHITE

        # Castling rights: (color, side) -> bool, where side is 'K' or 'Q'.
        self.castle_rights: Dict[Tuple[Color, str], bool] = {
            (Color.WHITE, "K"): True, (Color.WHITE, "Q"): True,
            (Color.BLACK, "K"): True, (Color.BLACK, "Q"): True,
        }
        # Square "behind" a pawn that just made a two-square push (target of
        # en-passant capture), or None.
        self.en_passant: Optional[Square] = None

        self.halfmove_clock: int = 0       # For the 50-move rule.
        self.fullmove_number: int = 1
        self.position_history: List[str] = []

        self._setup_initial_position()

    # ------------------------------------------------------------------
    # Setup and basic access
    # ------------------------------------------------------------------
    def _setup_initial_position(self) -> None:
        back = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        for c in range(8):
            self.squares[0][c] = Piece(back[c], Color.BLACK)
            self.squares[1][c] = Piece(PieceType.PAWN, Color.BLACK)
            self.squares[6][c] = Piece(PieceType.PAWN, Color.WHITE)
            self.squares[7][c] = Piece(back[c], Color.WHITE)
        self.position_history.append(self._position_hash())

    def piece_at(self, sq: Square) -> Optional[Piece]:
        r, c = sq
        if 0 <= r < 8 and 0 <= c < 8:
            return self.squares[r][c]
        return None

    def _position_hash(self) -> str:
        """Position signature for threefold repetition detection.

        Includes piece placement, side to move, castling rights and
        en-passant target (as required by FIDE's rule).
        """
        parts: List[str] = []
        for row in self.squares:
            for p in row:
                parts.append("." if p is None else repr(p))
        parts.append(str(self.turn.value))
        for (color, side), ok in self.castle_rights.items():
            if ok:
                parts.append(f"{color.value}{side}")
        if self.en_passant is not None:
            parts.append(f"ep{self.en_passant[0]},{self.en_passant[1]}")
        return "|".join(parts)

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------
    def pseudo_legal_moves(self, color: Optional[Color] = None) -> List[Move]:
        """All moves legal by piece-movement rules (may leave own king in check)."""
        if color is None:
            color = self.turn
        out: List[Move] = []
        for r in range(8):
            for c in range(8):
                p = self.squares[r][c]
                if p and p.color is color:
                    out.extend(self._moves_for_piece((r, c)))
        return out

    def legal_moves(self, color: Optional[Color] = None) -> List[Move]:
        """All fully legal moves for ``color`` (defaults to side to move)."""
        if color is None:
            color = self.turn
        return [m for m in self.pseudo_legal_moves(color) if self._is_legal(m)]

    def legal_moves_from(self, sq: Square) -> List[Move]:
        """Legal moves available to the piece currently on ``sq``."""
        p = self.piece_at(sq)
        if p is None or p.color is not self.turn:
            return []
        return [m for m in self._moves_for_piece(sq) if self._is_legal(m)]

    def _moves_for_piece(self, sq: Square) -> List[Move]:
        p = self.piece_at(sq)
        if p is None:
            return []
        t = p.type
        if t is PieceType.PAWN:
            return self._pawn_moves(sq, p)
        if t is PieceType.KNIGHT:
            return self._jump_moves(sq, p, _KNIGHT_DELTAS)
        if t is PieceType.BISHOP:
            return self._slide_moves(sq, p, _BISHOP_DIRS)
        if t is PieceType.ROOK:
            return self._slide_moves(sq, p, _ROOK_DIRS)
        if t is PieceType.QUEEN:
            return self._slide_moves(sq, p, _QUEEN_DIRS)
        if t is PieceType.KING:
            return self._king_moves(sq, p)
        return []

    def _pawn_moves(self, sq: Square, piece: Piece) -> List[Move]:
        r, c = sq
        direction = -1 if piece.color is Color.WHITE else 1
        start_rank = 6 if piece.color is Color.WHITE else 1
        promo_rank = 0 if piece.color is Color.WHITE else 7
        promo_types = [PieceType.QUEEN, PieceType.ROOK,
                       PieceType.BISHOP, PieceType.KNIGHT]
        moves: List[Move] = []

        # One square forward.
        r1 = r + direction
        if 0 <= r1 < 8 and self.squares[r1][c] is None:
            if r1 == promo_rank:
                for pt in promo_types:
                    moves.append(Move(sq, (r1, c), promotion=pt))
            else:
                moves.append(Move(sq, (r1, c)))
                # Two squares from starting rank.
                r2 = r + 2 * direction
                if r == start_rank and self.squares[r2][c] is None:
                    moves.append(Move(sq, (r2, c)))

        # Captures (including promotions).
        for dc in (-1, 1):
            nc = c + dc
            nr = r + direction
            if 0 <= nc < 8 and 0 <= nr < 8:
                target = self.squares[nr][nc]
                if target is not None and target.color is not piece.color:
                    if nr == promo_rank:
                        for pt in promo_types:
                            moves.append(Move(sq, (nr, nc), promotion=pt))
                    else:
                        moves.append(Move(sq, (nr, nc)))
                # En passant.
                if self.en_passant == (nr, nc) and target is None:
                    moves.append(Move(sq, (nr, nc), is_en_passant=True))
        return moves

    def _jump_moves(self, sq: Square, piece: Piece,
                    deltas: List[Tuple[int, int]]) -> List[Move]:
        r, c = sq
        out: List[Move] = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                tgt = self.squares[nr][nc]
                if tgt is None or tgt.color is not piece.color:
                    out.append(Move(sq, (nr, nc)))
        return out

    def _slide_moves(self, sq: Square, piece: Piece,
                     dirs: List[Tuple[int, int]]) -> List[Move]:
        r, c = sq
        out: List[Move] = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                tgt = self.squares[nr][nc]
                if tgt is None:
                    out.append(Move(sq, (nr, nc)))
                else:
                    if tgt.color is not piece.color:
                        out.append(Move(sq, (nr, nc)))
                    break
                nr += dr
                nc += dc
        return out

    def _king_moves(self, sq: Square, piece: Piece) -> List[Move]:
        moves = self._jump_moves(sq, piece, _KING_DELTAS)

        # Castling. Only legal if the king is not already in check and the
        # squares it crosses are not attacked.
        back_rank = 7 if piece.color is Color.WHITE else 0
        if sq == (back_rank, 4) and not self.is_in_check(piece.color):
            opp = piece.color.opponent
            # Kingside: squares f,g must be empty; rook on h; f,g not attacked.
            if self.castle_rights[(piece.color, "K")]:
                if (self.squares[back_rank][5] is None
                        and self.squares[back_rank][6] is None):
                    rook = self.squares[back_rank][7]
                    if (rook is not None and rook.type is PieceType.ROOK
                            and rook.color is piece.color
                            and not self._square_attacked((back_rank, 5), opp)
                            and not self._square_attacked((back_rank, 6), opp)):
                        moves.append(Move(sq, (back_rank, 6), is_castle=True))
            # Queenside: squares b,c,d must be empty; rook on a; c,d not attacked.
            if self.castle_rights[(piece.color, "Q")]:
                if (self.squares[back_rank][1] is None
                        and self.squares[back_rank][2] is None
                        and self.squares[back_rank][3] is None):
                    rook = self.squares[back_rank][0]
                    if (rook is not None and rook.type is PieceType.ROOK
                            and rook.color is piece.color
                            and not self._square_attacked((back_rank, 3), opp)
                            and not self._square_attacked((back_rank, 2), opp)):
                        moves.append(Move(sq, (back_rank, 2), is_castle=True))
        return moves

    # ------------------------------------------------------------------
    # Attack / check detection
    # ------------------------------------------------------------------
    def _square_attacked(self, sq: Square, by_color: Color) -> bool:
        """True if any piece of ``by_color`` attacks ``sq``."""
        tr, tc = sq
        # Pawn attacks: a pawn of ``by_color`` attacks ``sq`` from one rank
        # closer to its own side.
        pawn_dir = -1 if by_color is Color.WHITE else 1
        for dc in (-1, 1):
            pr, pc = tr - pawn_dir, tc - dc
            if 0 <= pr < 8 and 0 <= pc < 8:
                p = self.squares[pr][pc]
                if p is not None and p.color is by_color and p.type is PieceType.PAWN:
                    return True

        # Knight attacks.
        for dr, dc in _KNIGHT_DELTAS:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                p = self.squares[nr][nc]
                if p is not None and p.color is by_color and p.type is PieceType.KNIGHT:
                    return True

        # King attacks (adjacent squares).
        for dr, dc in _KING_DELTAS:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                p = self.squares[nr][nc]
                if p is not None and p.color is by_color and p.type is PieceType.KING:
                    return True

        # Sliding attacks along ranks/files: rook or queen.
        for dr, dc in _ROOK_DIRS:
            nr, nc = tr + dr, tc + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                p = self.squares[nr][nc]
                if p is not None:
                    if (p.color is by_color
                            and p.type in (PieceType.ROOK, PieceType.QUEEN)):
                        return True
                    break
                nr += dr
                nc += dc

        # Sliding attacks along diagonals: bishop or queen.
        for dr, dc in _BISHOP_DIRS:
            nr, nc = tr + dr, tc + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                p = self.squares[nr][nc]
                if p is not None:
                    if (p.color is by_color
                            and p.type in (PieceType.BISHOP, PieceType.QUEEN)):
                        return True
                    break
                nr += dr
                nc += dc
        return False

    def is_in_check(self, color: Color) -> bool:
        for r in range(8):
            for c in range(8):
                p = self.squares[r][c]
                if p is not None and p.type is PieceType.KING and p.color is color:
                    return self._square_attacked((r, c), color.opponent)
        return False

    def _is_legal(self, move: Move) -> bool:
        """A move is legal iff it does not leave the mover's king in check."""
        mover = self.piece_at(move.from_sq)
        if mover is None:
            return False
        undo = self._apply_move(move)
        in_check = self.is_in_check(mover.color)
        self._undo_move(undo)
        return not in_check

    # ------------------------------------------------------------------
    # Applying and undoing moves (used for search & legality checks)
    # ------------------------------------------------------------------
    def _apply_move(self, move: Move) -> Dict:
        """Apply ``move`` in place and return an undo record.

        This is the workhorse used by both :meth:`make_move` and the legality
        / search routines. Position history is intentionally *not* updated
        here, so that search can apply & undo millions of moves cheaply.
        """
        fr, fc = move.from_sq
        tr, tc = move.to_sq
        piece = self.squares[fr][fc]
        assert piece is not None, f"No piece at {move.from_sq}"

        captured = self.squares[tr][tc]
        captured_sq: Square = (tr, tc)

        undo: Dict = {
            "move": move,
            "piece": piece,
            "captured": captured,
            "captured_sq": captured_sq,
            "castle_rights": dict(self.castle_rights),
            "en_passant": self.en_passant,
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
            "turn": self.turn,
            "castle_rook_from": None,
            "castle_rook_to": None,
        }

        # Move the piece (promotion handled below).
        self.squares[tr][tc] = piece
        self.squares[fr][fc] = None

        if move.promotion is not None:
            self.squares[tr][tc] = Piece(move.promotion, piece.color)

        # En-passant: the captured pawn sits beside the moving pawn, not on to_sq.
        if move.is_en_passant:
            cap_r, cap_c = fr, tc
            undo["captured"] = self.squares[cap_r][cap_c]
            undo["captured_sq"] = (cap_r, cap_c)
            self.squares[cap_r][cap_c] = None

        # Castling: move the rook too.
        if move.is_castle:
            back = 7 if piece.color is Color.WHITE else 0
            if tc == 6:   # kingside
                rook = self.squares[back][7]
                self.squares[back][5] = rook
                self.squares[back][7] = None
                undo["castle_rook_from"] = (back, 7)
                undo["castle_rook_to"] = (back, 5)
            else:         # queenside (tc == 2)
                rook = self.squares[back][0]
                self.squares[back][3] = rook
                self.squares[back][0] = None
                undo["castle_rook_from"] = (back, 0)
                undo["castle_rook_to"] = (back, 3)

        # Update en-passant target (only set for double pawn pushes).
        if piece.type is PieceType.PAWN and abs(tr - fr) == 2:
            self.en_passant = ((fr + tr) // 2, fc)
        else:
            self.en_passant = None

        # Update castling rights.
        if piece.type is PieceType.KING:
            self.castle_rights[(piece.color, "K")] = False
            self.castle_rights[(piece.color, "Q")] = False
        if piece.type is PieceType.ROOK:
            back = 7 if piece.color is Color.WHITE else 0
            if fr == back and fc == 0:
                self.castle_rights[(piece.color, "Q")] = False
            elif fr == back and fc == 7:
                self.castle_rights[(piece.color, "K")] = False
        # Captured rook loses the right for its side.
        if captured is not None and captured.type is PieceType.ROOK:
            back = 7 if captured.color is Color.WHITE else 0
            if tr == back and tc == 0:
                self.castle_rights[(captured.color, "Q")] = False
            elif tr == back and tc == 7:
                self.castle_rights[(captured.color, "K")] = False

        # Half-move clock: reset on pawn move or capture.
        if piece.type is PieceType.PAWN or undo["captured"] is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.turn is Color.BLACK:
            self.fullmove_number += 1
        self.turn = self.turn.opponent
        return undo

    def _undo_move(self, undo: Dict) -> None:
        move: Move = undo["move"]
        fr, fc = move.from_sq
        tr, tc = move.to_sq

        # Restore moving piece (un-promote if needed via stored ``piece``).
        self.squares[fr][fc] = undo["piece"]
        self.squares[tr][tc] = None

        if undo["captured"] is not None:
            cr, cc = undo["captured_sq"]
            self.squares[cr][cc] = undo["captured"]

        if undo["castle_rook_from"] is not None:
            rf_r, rf_c = undo["castle_rook_from"]
            rt_r, rt_c = undo["castle_rook_to"]
            self.squares[rf_r][rf_c] = self.squares[rt_r][rt_c]
            self.squares[rt_r][rt_c] = None

        self.castle_rights = undo["castle_rights"]
        self.en_passant = undo["en_passant"]
        self.halfmove_clock = undo["halfmove_clock"]
        self.fullmove_number = undo["fullmove_number"]
        self.turn = undo["turn"]

    # ------------------------------------------------------------------
    # Public "permanent" move API
    # ------------------------------------------------------------------
    def make_move(self, move: Move) -> None:
        """Commit a move to the game and record the resulting position."""
        self._apply_move(move)
        self.position_history.append(self._position_hash())

    # ------------------------------------------------------------------
    # Game-end detection
    # ------------------------------------------------------------------
    def is_checkmate(self) -> bool:
        return self.is_in_check(self.turn) and not self.legal_moves()

    def is_stalemate(self) -> bool:
        return (not self.is_in_check(self.turn)) and not self.legal_moves()

    def is_insufficient_material(self) -> bool:
        """A simple (but standard) subset of FIDE's insufficient-material rule."""
        bishops_by_square_color: Dict[Color, List[int]] = {Color.WHITE: [], Color.BLACK: []}
        knights = 0
        other_material = False
        kings = 0
        for r in range(8):
            for c in range(8):
                p = self.squares[r][c]
                if p is None:
                    continue
                if p.type is PieceType.KING:
                    kings += 1
                elif p.type is PieceType.BISHOP:
                    bishops_by_square_color[p.color].append((r + c) % 2)
                elif p.type is PieceType.KNIGHT:
                    knights += 1
                else:
                    other_material = True
        if other_material:
            return False
        total_minor = knights + sum(len(v) for v in bishops_by_square_color.values())
        if total_minor == 0:
            return True                                   # K vs K
        if total_minor == 1:
            return True                                   # K+minor vs K
        # K+B vs K+B with both bishops on the same square colour.
        if (knights == 0
                and len(bishops_by_square_color[Color.WHITE]) == 1
                and len(bishops_by_square_color[Color.BLACK]) == 1
                and bishops_by_square_color[Color.WHITE][0]
                    == bishops_by_square_color[Color.BLACK][0]):
            return True
        return False

    def is_threefold_repetition(self) -> bool:
        if not self.position_history:
            return False
        return self.position_history.count(self.position_history[-1]) >= 3

    def is_fifty_move_rule(self) -> bool:
        return self.halfmove_clock >= 100  # 50 full moves = 100 half-moves

    def game_result(self) -> Optional[str]:
        """Human-readable result string, or ``None`` if the game is ongoing."""
        if self.is_checkmate():
            winner = "Black" if self.turn is Color.WHITE else "White"
            return f"{winner} wins by checkmate"
        if self.is_stalemate():
            return "Draw by stalemate"
        if self.is_insufficient_material():
            return "Draw by insufficient material"
        if self.is_threefold_repetition():
            return "Draw by threefold repetition"
        if self.is_fifty_move_rule():
            return "Draw by 50-move rule"
        return None

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------
    def ascii(self) -> str:
        lines = []
        for r in range(8):
            row = f"{8 - r} "
            for c in range(8):
                p = self.squares[r][c]
                row += (p.symbol() if p else ".") + " "
            lines.append(row.rstrip())
        lines.append("  a b c d e f g h")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.ascii()


# ---------------------------------------------------------------------------
# Perft: a standard correctness check for chess move generators.
# ---------------------------------------------------------------------------
def perft(board: Board, depth: int) -> int:
    """Count leaf nodes of the move tree at the given depth.

    Useful for validating move generation against published reference counts.
    """
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves():
        undo = board._apply_move(move)
        nodes += perft(board, depth - 1)
        board._undo_move(undo)
    return nodes
