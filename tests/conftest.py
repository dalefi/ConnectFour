import sys
from pathlib import Path

import numpy as np
import pytest

# Projekt-Root auf den Pfad legen, damit "from src...." und "from mcts...." funktionieren
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ConnectFour import ConnectFour  # noqa: E402


@pytest.fixture
def empty_board():
    return ConnectFour()


def play(columns, board=None):
    """
    Spielt eine Zugfolge auf einem (optional vorgegebenen) Brett und gibt den
    Endzustand zurueck. Spieler wechseln sich ab, beginnend mit dem aktuellen.
    """
    state = board if board is not None else ConnectFour()
    for column in columns:
        state.make_move(column)
    return state


def board_from_rows(rows, current_player=1, last_move=None):
    """
    Baut ein Brett aus einer Liste von 6 Strings mit '.', 'x' (=1) und 'o' (=-1).
    Zeile 0 ist oben, Zeile 5 ist unten - genau wie in ConnectFour.board.
    """
    assert len(rows) == 6, "Ein Connect-Four-Brett hat 6 Zeilen"
    symbols = {".": 0, "x": 1, "o": -1}
    board = np.array(
        [[symbols[c] for c in row] for row in rows],
        dtype=np.int8,
    )
    assert board.shape == (6, 7), "Ein Connect-Four-Brett hat 7 Spalten"
    return ConnectFour(board=board, currentPlayer=current_player, last_move=last_move)


def full_board_winner(board):
    """
    Referenz-Implementierung: scannt das gesamte Brett. Bewusst naiv gehalten,
    damit sie als unabhaengige Kontrolle fuer die schnelle Variante dient.
    """
    for row in range(6):
        for col in range(7):
            player = board[row, col]
            if player == 0:
                continue
            for delta_row, delta_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_row = row + 3 * delta_row
                end_col = col + 3 * delta_col
                if not (0 <= end_row < 6 and 0 <= end_col < 7):
                    continue
                if all(
                    board[row + i * delta_row, col + i * delta_col] == player
                    for i in range(4)
                ):
                    return int(player)
    return 0
