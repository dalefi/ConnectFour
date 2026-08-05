import random

import numpy as np
import pytest

from conftest import board_from_rows, full_board_winner, play
from src.ConnectFour import Action, ConnectFour


# ---------------------------------------------------------------------------
# Gewinnerkennung
# ---------------------------------------------------------------------------

def test_empty_board_has_no_winner(empty_board):
    assert empty_board.get_winner() == 0


def test_detects_vertical_win():
    state = board_from_rows([
        ".......",
        ".......",
        "x......",
        "x......",
        "x......",
        "x......",
    ])
    assert state.get_winner() == 1


def test_detects_horizontal_win_for_player_minus_one():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "..oooo.",
    ])
    assert state.get_winner() == -1


def test_detects_descending_diagonal():
    state = board_from_rows([
        ".......",
        ".......",
        "x......",
        ".x.....",
        "..x....",
        "...x...",
    ])
    assert state.get_winner() == 1


def test_detects_ascending_diagonal():
    state = board_from_rows([
        ".......",
        ".......",
        "...o...",
        "..o....",
        ".o.....",
        "o......",
    ])
    assert state.get_winner() == -1


def test_detects_win_in_rightmost_column():
    state = board_from_rows([
        ".......",
        ".......",
        "......o",
        "......o",
        "......o",
        "......o",
    ])
    assert state.get_winner() == -1


def test_detects_win_in_top_row():
    state = board_from_rows([
        "xxxx...",
        "ooox...",
        "xxxo...",
        "ooox...",
        "xxxo...",
        "ooox...",
    ])
    assert state.get_winner() == 1


def test_three_in_a_row_is_not_a_win():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "xxx.ooo",
    ])
    assert state.get_winner() == 0


def test_gap_breaks_a_run_of_four():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "xx.xx..",
    ])
    assert state.get_winner() == 0


def test_winner_is_found_on_board_built_without_last_move():
    """
    Zustaende werden auch aus rohen Brettern erzeugt (z.B. tensor_to_state oder
    beim Nachspielen aus der Datenbank). Dann ist last_move unbekannt und die
    Gewinnerkennung darf sich nicht darauf verlassen.
    """
    state = ConnectFour(
        board=np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0],
            [0, 0, 1, 1, -1, 0, 0],
        ], dtype=np.int8),
        currentPlayer=1,
        last_move=None,
    )
    assert state.get_winner() == 0

    state.make_move(2)  # vierter Stein von Spieler 1 in Spalte 2
    assert state.get_winner() == 1


def test_winner_matches_reference_scan_over_random_games():
    """
    Unabhaengige Gegenprobe: die (schnelle) Gewinnerkennung muss auf jeder
    erreichbaren Stellung dasselbe liefern wie ein voller Brett-Scan.
    """
    rng = random.Random(20260805)
    positions_checked = 0

    for _ in range(400):
        state = ConnectFour()
        while not state.is_terminal():
            columns = [a.target_column for a in state.get_possible_actions()]
            state.make_move(rng.choice(columns))
            positions_checked += 1
            assert state.get_winner() == full_board_winner(state.board)

    assert positions_checked > 5000, "Zu wenig Stellungen geprueft"


# ---------------------------------------------------------------------------
# Terminalzustand und Belohnung
# ---------------------------------------------------------------------------

def test_empty_board_is_not_terminal(empty_board):
    assert empty_board.is_terminal() is False


def test_board_with_winner_is_terminal():
    state = play([0, 1, 0, 1, 0, 1, 0])
    assert state.is_terminal() is True


def test_reward_is_minus_one_for_the_player_who_must_move_after_losing():
    state = play([0, 1, 0, 1, 0, 1, 0])  # Spieler 1 gewinnt in Spalte 0
    assert state.get_current_player() == -1
    assert state.get_reward() == -1


def test_reward_is_zero_on_a_drawn_board():
    state = board_from_rows([
        "xoooxxx",
        "xoxoxoo",
        "oxoooxo",
        "oooxxxo",
        "xxxoxox",
        "xxoxoxo",
    ])
    assert state.get_winner() == 0
    assert state.board_is_full() is True
    assert state.is_terminal() is True
    assert state.get_reward() == 0


def test_reward_raises_when_the_game_is_still_running(empty_board):
    with pytest.raises(AssertionError):
        empty_board.get_reward()


def test_winner_takes_priority_over_a_full_board():
    state = board_from_rows([
        "xxxxoxo",
        "oooxxox",
        "xxxoxox",
        "oooxoxo",
        "xxxoxox",
        "oooxoxo",
    ])
    assert state.board_is_full() is True
    assert state.get_winner() == 1
    assert state.get_reward() == -1


# ---------------------------------------------------------------------------
# Zuege
# ---------------------------------------------------------------------------

def test_stone_falls_to_the_lowest_free_row(empty_board):
    position = empty_board.make_move(3)
    assert position == (5, 3)
    assert empty_board.board[5, 3] == 1


def test_stones_stack_on_top_of_each_other(empty_board):
    empty_board.make_move(3)
    position = empty_board.make_move(3)
    assert position == (4, 3)
    assert empty_board.board[4, 3] == -1


def test_make_move_switches_the_current_player(empty_board):
    assert empty_board.get_current_player() == 1
    empty_board.make_move(0)
    assert empty_board.get_current_player() == -1
    empty_board.make_move(0)
    assert empty_board.get_current_player() == 1


def test_make_move_records_the_last_move(empty_board):
    empty_board.make_move(6)
    assert empty_board.last_move == (5, 6)


def test_make_move_rejects_a_column_outside_the_board(empty_board):
    with pytest.raises(ValueError):
        empty_board.make_move(7)
    with pytest.raises(ValueError):
        empty_board.make_move(-1)


def test_make_move_rejects_a_full_column(empty_board):
    for _ in range(6):
        empty_board.make_move(2)
    assert empty_board.board[0, 2] != 0
    with pytest.raises(ValueError):
        empty_board.make_move(2)


def test_take_action_does_not_mutate_the_original(empty_board):
    original = empty_board.board.copy()
    child = empty_board.take_action(Action(target_column=3, player=1))

    assert np.array_equal(empty_board.board, original)
    assert empty_board.get_current_player() == 1
    assert child.board[5, 3] == 1
    assert child.get_current_player() == -1
    assert child.board is not empty_board.board


def test_take_action_result_knows_its_last_move(empty_board):
    child = empty_board.take_action(Action(target_column=1, player=1))
    assert child.last_move == (5, 1)


def test_copy_is_independent_of_the_original():
    state = play([3, 3, 4])
    duplicate = state.copy()

    assert np.array_equal(duplicate.board, state.board)
    assert duplicate.get_current_player() == state.get_current_player()
    assert duplicate.last_move == state.last_move

    duplicate.make_move(0)
    assert state.board[5, 0] == 0
    assert duplicate.board[5, 0] != 0


def test_copy_keeps_the_winner_consistent():
    state = play([0, 1, 0, 1, 0, 1, 0])
    duplicate = state.copy()
    assert duplicate.get_winner() == 1
    assert duplicate.is_terminal() is True


# ---------------------------------------------------------------------------
# Caching darf nicht veralten
# ---------------------------------------------------------------------------

def test_winner_is_recomputed_after_a_further_move():
    state = play([0, 1, 0, 1, 0, 1])
    assert state.get_winner() == 0  # fuellt einen eventuellen Cache

    state.make_move(0)  # vierter Stein von Spieler 1
    assert state.get_winner() == 1


def test_terminal_flag_is_recomputed_after_a_further_move():
    state = play([0, 1, 0, 1, 0, 1])
    assert state.is_terminal() is False

    state.make_move(0)
    assert state.is_terminal() is True


# ---------------------------------------------------------------------------
# Brett-Status
# ---------------------------------------------------------------------------

def test_board_is_full_only_with_42_stones(empty_board):
    assert empty_board.board_is_full() is False
    empty_board.make_move(0)
    assert empty_board.board_is_full() is False

    full = board_from_rows([
        "xoooxxx",
        "xoxoxoo",
        "oxoooxo",
        "oooxxxo",
        "xxxoxox",
        "xxoxoxo",
    ])
    assert full.board_is_full() is True


def test_possible_actions_on_an_empty_board(empty_board):
    columns = [a.target_column for a in empty_board.get_possible_actions()]
    assert columns == [0, 1, 2, 3, 4, 5, 6]


def test_possible_actions_exclude_full_columns(empty_board):
    for _ in range(6):
        empty_board.make_move(4)
    columns = [a.target_column for a in empty_board.get_possible_actions()]
    assert 4 not in columns
    assert len(columns) == 6


def test_possible_actions_carry_the_current_player(empty_board):
    empty_board.make_move(0)
    assert all(a.player == -1 for a in empty_board.get_possible_actions())


def test_full_board_has_no_possible_actions():
    full = board_from_rows([
        "xoooxxx",
        "xoxoxoo",
        "oxoooxo",
        "oooxxxo",
        "xxxoxox",
        "xxoxoxo",
    ])
    assert full.get_possible_actions() == []


# ---------------------------------------------------------------------------
# Zufaellige Startstellungen
# ---------------------------------------------------------------------------

def test_random_start_state_is_never_terminal():
    for _ in range(200):
        state = ConnectFour.random_start_state(max_random_moves=6)
        assert not state.is_terminal()


def test_random_start_state_respects_the_move_limit():
    for _ in range(100):
        state = ConnectFour.random_start_state(max_random_moves=4)
        assert np.count_nonzero(state.board) <= 4


def test_random_start_state_alternates_players_consistently():
    for _ in range(100):
        state = ConnectFour.random_start_state(max_random_moves=6)
        stones_of_one = int(np.count_nonzero(state.board == 1))
        stones_of_minus_one = int(np.count_nonzero(state.board == -1))
        expected_player = 1 if stones_of_one == stones_of_minus_one else -1
        assert state.get_current_player() == expected_player


# ---------------------------------------------------------------------------
# Geschwindigkeit der heissen Pfade
#
# Die Gewinnpruefung laeuft pro MCTS-Iteration mehrfach und dominierte frueher
# die gesamte Laufzeit. Die Schranken liegen bewusst weit weg von beiden Seiten -
# sie sollen einen Rueckfall auf den vollen Brett-Scan (Faktor ~100) fangen, nicht
# Schwankungen der Maschine. Referenz auf diesem Rechner:
#
#   is_terminal (kalt)   7 us   (vorher 208 us)
#   expand() 7 Kinder   85 us   (vorher 1340 us)
# ---------------------------------------------------------------------------

def _microseconds_per_call(function, repeats=2000):
    """
    Minimum aus mehreren Messreihen: der schnellste Durchlauf ist der, bei dem am
    wenigsten anderes dazwischenkam. Mittelwerte schwanken hier um Faktor 2-3.
    """
    import timeit
    batches = timeit.repeat(function, repeat=5, number=repeats)
    return min(batches) / repeats * 1e6


def test_terminal_check_is_cheap_when_the_last_move_is_known():
    state = play([3, 2, 4, 1, 5, 0, 3, 2, 4, 1])
    assert _microseconds_per_call(state.is_terminal) < 40.0


def test_expanding_all_children_is_cheap():
    """
    Ein MCTS-expand() erzeugt alle Kinder auf einmal; jedes Kind bestimmt dabei
    seinen Terminalstatus. Das ist der teuerste Einzelschritt der Suche.
    """
    state = play([3, 2, 4, 1, 5, 0, 3, 2, 4, 1])
    actions = state.get_possible_actions()

    def expand_all():
        for action in actions:
            child = state.take_action(action)
            child.is_terminal()

    assert _microseconds_per_call(expand_all, repeats=500) < 400.0


def test_repeated_terminal_checks_do_not_recompute():
    state = play([3, 2, 4, 1, 5, 0, 3, 2, 4, 1])
    state.is_terminal()
    assert _microseconds_per_call(state.is_terminal) < 5.0


# ---------------------------------------------------------------------------
# Typen an der Datenbank-Grenze
# ---------------------------------------------------------------------------

def test_winner_is_a_plain_python_int():
    state = play([0, 1, 0, 1, 0, 1, 0])
    winner = state.get_winner()
    assert winner == 1
    assert type(winner) is int


def test_reward_is_a_plain_python_number():
    state = play([0, 1, 0, 1, 0, 1, 0])
    assert type(state.get_reward()) is int


# ---------------------------------------------------------------------------
# Minimax-Referenzgegner
# ---------------------------------------------------------------------------

def test_minimax_takes_an_immediate_win():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "..o....",
        "xxx.oo.",
    ], current_player=1)
    assert state.get_best_move(depth=2) == 3


def test_minimax_blocks_an_immediate_loss():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "ooo.x.x",
    ], current_player=1)
    assert state.get_best_move(depth=4) == 3


def test_minimax_returns_a_playable_column():
    state = ConnectFour.random_start_state(max_random_moves=6)
    legal = [a.target_column for a in state.get_possible_actions()]
    assert state.get_best_move(depth=4) in legal
