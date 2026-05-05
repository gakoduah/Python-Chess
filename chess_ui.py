"""
chess_ui.py
===========

Interactive Jupyter UI for the chess engine, built with ``ipywidgets``.

The UI is intentionally thin: all chess knowledge lives in ``chess_engine``,
and the AI lives in ``chess_ai``. This module only handles clicks, rendering,
and threading the AI off the UI thread so the notebook stays responsive.

Usage (inside Jupyter)
----------------------
    from chess_ui import ChessUI
    ui = ChessUI(ai_depth=3)
    ui.display()
"""

from __future__ import annotations

import threading
from typing import List, Optional

import ipywidgets as widgets
from IPython.display import HTML, display

from chess_engine import Board, Color, Move, PIECE_SYMBOLS, PieceType
from chess_ai import get_best_move


# Colour palette (chess.com-ish).
_LIGHT = "#f0d9b5"
_DARK = "#b58863"
_SEL = "#ffff66"         # selected source square
_MOVE_EMPTY = "#baeda0"  # legal move to empty square
_MOVE_CAPT = "#e58e8e"   # legal capture square
_CHECK = "#ff6e6e"       # king-in-check square

# One-time CSS injection so chess-glyphs render at a readable size and
# the grid has no inter-button gap. Call :func:`_inject_css` before creating
# widgets; ``ChessUI`` does this automatically.
_CSS_INJECTED = False
_CSS = """
<style>
.chess-square {
    font-size: 34px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid #333 !important;
    border-radius: 0 !important;
    min-width: 56px !important;
    width: 56px !important;
    height: 56px !important;
    line-height: 56px !important;
    font-family: "Segoe UI Symbol","DejaVu Sans","Arial Unicode MS",sans-serif !important;
}
.chess-status { font-size: 16px; font-family: monospace; }
.chess-movelist {
    font-family: monospace; font-size: 13px;
    max-height: 220px; overflow-y: auto;
    border: 1px solid #ccc; padding: 4px;
}
</style>
"""


def _inject_css() -> None:
    global _CSS_INJECTED
    if not _CSS_INJECTED:
        display(HTML(_CSS))
        _CSS_INJECTED = True


def _algebraic(sq) -> str:
    r, c = sq
    return "abcdefgh"[c] + str(8 - r)


class ChessUI:
    """Interactive Jupyter chessboard playing against the built-in AI.

    Parameters
    ----------
    ai_depth : int
        Search depth for the opponent (default 3 — playable, under ~1s/move).
    player_color : Color
        Which side the human plays (default White).
    auto_promote_to : PieceType
        Piece type for automatic pawn promotion. Defaults to Queen.
    """

    def __init__(self,
                 ai_depth: int = 3,
                 player_color: Color = Color.WHITE,
                 auto_promote_to: PieceType = PieceType.QUEEN) -> None:
        _inject_css()
        self.ai_depth = ai_depth
        self.player_color = player_color
        self.auto_promote_to = auto_promote_to

        self.board = Board()
        self.selected: Optional = None
        self.legal_from_selected: List[Move] = []
        self.move_history: List[str] = []   # UCI-ish strings
        self._ai_thinking = False

        self._build_widgets()
        self._render()

        # If the human plays Black, let the AI open.
        if self.player_color is Color.BLACK:
            self._schedule_ai()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        # 8x8 grid of buttons.
        self.buttons = [[None] * 8 for _ in range(8)]
        grid_children = []
        for r in range(8):
            for c in range(8):
                btn = widgets.Button(
                    description="",
                    layout=widgets.Layout(width="56px", height="56px",
                                          padding="0px", margin="0px"),
                )
                btn.add_class("chess-square")
                btn.on_click(self._make_click_handler(r, c))
                self.buttons[r][c] = btn
                grid_children.append(btn)

        self.grid = widgets.GridBox(
            children=grid_children,
            layout=widgets.Layout(
                grid_template_columns="repeat(8, 56px)",
                grid_gap="0px",
                width="448px",
            ),
        )

        # Status / controls.
        self.status = widgets.HTML()
        self.status.add_class("chess-status")

        self.move_list = widgets.HTML("")
        self.move_list.add_class("chess-movelist")
        self.move_list.layout = widgets.Layout(width="170px")

        self.depth_slider = widgets.IntSlider(
            value=self.ai_depth, min=1, max=4, step=1,
            description="AI depth:",
            continuous_update=False,
            layout=widgets.Layout(width="260px"),
        )
        self.depth_slider.observe(self._on_depth_change, names="value")

        self.new_game_btn = widgets.Button(description="New game",
                                           button_style="primary")
        self.new_game_btn.on_click(lambda _: self._new_game())

        self.flip_btn = widgets.Button(description="Play other color")
        self.flip_btn.on_click(lambda _: self._flip_sides())

        controls = widgets.HBox([self.new_game_btn, self.flip_btn,
                                 self.depth_slider])
        right_panel = widgets.VBox([widgets.HTML("<b>Moves</b>"),
                                    self.move_list])
        board_with_labels = widgets.VBox([self.status, self.grid, controls])
        self.container = widgets.HBox([board_with_labels, right_panel])

    def _make_click_handler(self, r: int, c: int):
        def handler(_btn):
            self._handle_click((r, c))
        return handler

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------
    def _handle_click(self, sq) -> None:
        # Ignore input while AI is thinking, game over, or not player's turn.
        if self._ai_thinking or self.board.game_result() is not None:
            return
        if self.board.turn is not self.player_color:
            return

        piece = self.board.piece_at(sq)

        # No current selection: pick a friendly piece.
        if self.selected is None:
            if piece is not None and piece.color is self.player_color:
                self.selected = sq
                self.legal_from_selected = self.board.legal_moves_from(sq)
                self._render()
            return

        # Already selected: either move, reselect, or deselect.
        matching = [m for m in self.legal_from_selected if m.to_sq == sq]
        if matching:
            move = matching[0]
            # If several matches (different promotion choices), pick preferred.
            if move.promotion is not None:
                preferred = [m for m in matching
                             if m.promotion is self.auto_promote_to]
                move = preferred[0] if preferred else matching[0]
            self._make_player_move(move)
            return

        if piece is not None and piece.color is self.player_color:
            # Reselect a different friendly piece.
            self.selected = sq
            self.legal_from_selected = self.board.legal_moves_from(sq)
            self._render()
        else:
            # Click on an invalid square: just deselect.
            self.selected = None
            self.legal_from_selected = []
            self._render()

    def _make_player_move(self, move: Move) -> None:
        self._commit_move(move)
        self.selected = None
        self.legal_from_selected = []
        self._render()
        if self.board.game_result() is None:
            self._schedule_ai()

    def _commit_move(self, move: Move) -> None:
        mover = self.board.piece_at(move.from_sq)
        captured = self.board.piece_at(move.to_sq) is not None or move.is_en_passant
        self.board.make_move(move)

        # Build an annotated notation string for the sidebar.
        notation = move.uci()
        if move.is_castle:
            notation = "O-O" if move.to_sq[1] == 6 else "O-O-O"
        else:
            piece_letter = ""
            if mover is not None and mover.type is not PieceType.PAWN:
                piece_letter = mover.type.value
            sep = "x" if captured else ""
            notation = (f"{piece_letter}{_algebraic(move.from_sq)}"
                        f"{sep}{_algebraic(move.to_sq)}")
            if move.promotion is not None:
                notation += "=" + move.promotion.value
        if self.board.is_in_check(self.board.turn):
            notation += "#" if self.board.is_checkmate() else "+"
        self.move_history.append(notation)

    # ------------------------------------------------------------------
    # AI threading
    # ------------------------------------------------------------------
    def _schedule_ai(self) -> None:
        """Run the AI in a background thread to keep the UI responsive."""
        self._ai_thinking = True
        self._render()    # Redraw to show "AI thinking..." status.

        def worker():
            try:
                move = get_best_move(self.board, depth=self.ai_depth)
                if move is not None:
                    self._commit_move(move)
            finally:
                self._ai_thinking = False
                self._render()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self) -> None:
        legal_targets = {m.to_sq for m in self.legal_from_selected}

        # Locate the checked king (if any) for red-square highlighting.
        check_sq = None
        if self.board.is_in_check(self.board.turn):
            for r in range(8):
                for c in range(8):
                    p = self.board.squares[r][c]
                    if (p is not None and p.type is PieceType.KING
                            and p.color is self.board.turn):
                        check_sq = (r, c)
                        break
                if check_sq:
                    break

        for r in range(8):
            for c in range(8):
                btn = self.buttons[r][c]
                piece = self.board.squares[r][c]
                # Use a thin space (U+2009) for empty squares instead of
                # an empty string — ipywidgets sometimes fails to repaint a
                # button when its description is cleared to "", leaving a
                # ghost of the old glyph. A real-but-invisible character
                # forces a clean redraw.
                btn.description = piece.symbol() if piece else "\u2009"

                # Pick a colour for the square.
                if (r, c) == self.selected:
                    color = _SEL
                elif (r, c) in legal_targets:
                    color = _MOVE_CAPT if piece is not None else _MOVE_EMPTY
                elif (r, c) == check_sq:
                    color = _CHECK
                else:
                    color = _LIGHT if (r + c) % 2 == 0 else _DARK
                btn.style.button_color = color

        self._update_status()
        self._update_move_list()

    def _update_status(self) -> None:
        result = self.board.game_result()
        if result is not None:
            self.status.value = f"<b>Game over:</b> {result}"
            return
        turn_name = "White" if self.board.turn is Color.WHITE else "Black"
        if self._ai_thinking:
            self.status.value = (f"<b>{turn_name} to move</b> &nbsp;"
                                 f"<i>AI thinking…</i>")
            return
        extra = ""
        if self.board.is_in_check(self.board.turn):
            extra = " &nbsp; <span style='color:#b00'><b>CHECK</b></span>"
        self.status.value = f"<b>{turn_name} to move</b>{extra}"

    def _update_move_list(self) -> None:
        # Two half-moves per line: "1. e4 e5".
        lines = []
        for i in range(0, len(self.move_history), 2):
            n = i // 2 + 1
            white = self.move_history[i]
            black = self.move_history[i + 1] if i + 1 < len(self.move_history) else ""
            lines.append(f"{n}. {white} {black}")
        self.move_list.value = "<br>".join(lines) if lines else "<i>(no moves yet)</i>"

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------
    def _on_depth_change(self, change) -> None:
        self.ai_depth = int(change["new"])

    def _new_game(self) -> None:
        if self._ai_thinking:
            return                           # Wait for AI to finish first.
        self.board = Board()
        self.selected = None
        self.legal_from_selected = []
        self.move_history = []
        self._render()
        if self.player_color is Color.BLACK:
            self._schedule_ai()

    def _flip_sides(self) -> None:
        if self._ai_thinking:
            return
        self.player_color = self.player_color.opponent
        self._new_game()

    def display(self) -> None:
        display(self.container)
