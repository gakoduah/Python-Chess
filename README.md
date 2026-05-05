# Python Chess

A from-scratch 8×8 chess implementation in pure Python: full rules, a minimax + alpha-beta AI, an interactive `ipywidgets` board for Jupyter, and a unit-test suite with `perft` correctness checks.

Built as a Python port and extension of a 1D JavaScript chess variant — the original used only king, knight, and rook on a single rank; this version implements the complete game on a full 8×8 board.

---

## What's in here

| File | What it does |
|------|--------------|
| `chess_engine.py` | Board representation, move generation for all six piece types, castling, en passant, promotion, and game-end detection (checkmate, stalemate, threefold repetition, 50-move rule, insufficient material). Zero external dependencies. |
| `chess_ai.py`     | Negamax search with alpha-beta pruning, MVV-LVA move ordering, material + piece-square-table evaluation. |
| `chess_ui.py`     | Interactive `ipywidgets` board for Jupyter. Click to select, click to move. |
| `test_chess.py`   | 14 unit tests, including `perft(1..3)` against reference node counts. |
| `chess.ipynb`     | Notebook walking through engine sanity checks, the interactive game, AI analysis, and self-play. |

---

## Quick start

```bash
pip install ipywidgets notebook
jupyter notebook chess.ipynb
```

Then run the cells top-to-bottom. Cell 4 gives you the clickable board.

Prefer JupyterLab? `pip install jupyterlab` and `jupyter lab chess.ipynb` works the same.

---

## Using the engine directly

```python
from chess_engine import Board, Move
from chess_ai import get_best_move

board = Board()
print(board.ascii())

# Let the AI play a move
move = get_best_move(board, depth=3)
board.make_move(move)
print(board.ascii())
```

---

## Running the tests

```bash
python test_chess.py
```

Expected output: **14 tests passing in under a second.** The `perft` tests are
the most important — they validate that move generation produces exactly the
right number of positions at every depth, which catches nearly every
subtle bug in castling, en passant, pinned-piece handling, and promotion.

---

## Architecture notes

**Separation of concerns.** The engine knows nothing about the UI, the AI, or
how it's being driven. You can unit-test it, feed it positions from a file,
or plug in a different front end without touching `chess_engine.py`.

**Reversible move application.** `_apply_move(move)` returns an undo record
that `_undo_move(undo)` uses to perfectly restore board state. This is the
foundation the search relies on — millions of moves can be applied and undone
during a search without ever cloning the board.

**Search.** Negamax with alpha-beta pruning and MVV-LVA ordering (most-valuable
victim, least-valuable attacker). At depth 3, the engine explores ~1,000–3,000
nodes per move, which is the tradeoff the pruning buys us versus raw minimax.

**Evaluation.** Material in centipawns + standard piece-square tables from the
chess-programming literature. Good enough for a baseline; see "Extensions" in
the notebook for how to take it further (quiescence, transposition tables,
NNUE-style evaluation).

---

## Known limitations

- Pawn promotion in the UI auto-promotes to queen. ~99% of the time this is
  correct; a more complete UI would pop up a chooser.
- The engine is pure Python without bitboards — it's fast enough to be fun but
  will never match a C++ engine like Stockfish. Depth 4 takes a few seconds
  per move.
- No opening book or endgame tablebase.

None of these are hard to fix; see section 7 of the notebook for suggested
next steps.
