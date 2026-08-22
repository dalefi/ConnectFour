import random

import numpy as np
import pytest

from conftest import board_from_rows
from src.ConnectFour import ConnectFour

pytest.importorskip("connect_four_ai", reason="connect-four-ai ist optional")

from src.solver_benchmark import (  # noqa: E402
    board_to_position,
    optimal_columns,
    score_loss,
)


# ---------------------------------------------------------------------------
# Umwandlung in die Bitboard-Darstellung des Solvers
# ---------------------------------------------------------------------------

def test_empty_board_has_no_moves_played():
    assert board_to_position(ConnectFour()).get_moves() == 0


def test_move_count_matches_the_number_of_stones():
    state = ConnectFour()
    for move_number, column in enumerate((3, 3, 4, 2, 4), start=1):
        state.make_move(column)
        assert board_to_position(state).get_moves() == move_number


def test_conversion_rejects_a_board_whose_player_contradicts_the_stone_count():
    """
    'x' ist im Solver immer der Anziehende; wer am Zug ist, folgt aus der Anzahl
    der Steine. Ein Brett, dessen currentPlayer dazu nicht passt, waere still falsch.
    """
    inconsistent = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "x......",
    ], current_player=1)  # nach einem Stein muss -1 am Zug sein

    with pytest.raises(ValueError):
        board_to_position(inconsistent)


def test_solver_agrees_with_our_win_detection_over_random_games():
    """
    Gegenprobe ueber zwei voellig unabhaengige Implementierungen: unsere
    Gewinnerkennung und das Bitboard des Solvers.
    """
    rng = random.Random(4711)
    checked = 0

    for _ in range(60):
        state = ConnectFour()
        while True:
            columns = [a.target_column for a in state.get_possible_actions()]
            column = rng.choice(columns)

            position = board_to_position(state)
            expected_win = state.take_action(
                next(a for a in state.get_possible_actions() if a.target_column == column)
            ).get_winner() != 0
            assert position.is_winning_move(column) == expected_win
            checked += 1

            state.make_move(column)
            if state.is_terminal():
                break

    assert checked > 800


# ---------------------------------------------------------------------------
# Optimale Zuege und Fehlergroesse
# ---------------------------------------------------------------------------

def test_optimal_columns_finds_an_immediate_win():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "..o....",
        "xxx.oo.",
    ], current_player=1)

    assert optimal_columns(state) == {3}


def test_optimal_columns_finds_the_only_defence():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "x......",
        "ooo.x.x",
    ], current_player=1)

    assert optimal_columns(state) == {3}


def test_optimal_columns_never_returns_a_full_column():
    state = ConnectFour()
    for _ in range(6):
        state.make_move(3)

    assert 3 not in optimal_columns(state)


def test_score_loss_is_zero_for_an_optimal_move():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "..o....",
        "xxx.oo.",
    ], current_player=1)

    assert score_loss(state, 3) == 0


def test_score_loss_is_positive_for_a_blunder():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "x......",
        "ooo.x.x",
    ], current_player=1)

    assert score_loss(state, 0) > 0
