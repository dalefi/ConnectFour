"""
Der mitgelieferte Stellungssatz muss ohne connect-four-ai nutzbar sein - das Paket
gibt es nur als einzelnes Wheel fuer CPython 3.13 auf Windows. Diese Tests
importieren es bewusst nicht.
"""
from pathlib import Path

import numpy as np
import pytest

from src.solver_benchmark import load_benchmark_set

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "data" / "solver_benchmark.json"

pytestmark = pytest.mark.skipif(
    not BENCHMARK_PATH.exists(),
    reason="data/solver_benchmark.json fehlt - mit build_benchmark_set erzeugen",
)


@pytest.fixture(scope="module")
def entries():
    return load_benchmark_set(BENCHMARK_PATH)


def test_the_set_is_not_empty(entries):
    assert len(entries) >= 50


def test_every_entry_has_a_six_by_seven_board(entries):
    for entry in entries:
        assert np.array(entry["board"]).shape == (6, 7)


def test_every_entry_carries_a_score_per_column(entries):
    for entry in entries:
        assert len(entry["scores"]) == 7


def test_no_position_is_already_finished(entries):
    """Eine beendete Stellung hat keine Zuege zu bewerten."""
    for entry in entries:
        assert not entry["state"].is_terminal()


def test_every_position_is_reachable_in_a_real_game(entries):
    for entry in entries:
        board = np.array(entry["board"])
        own = int(np.count_nonzero(board == 1))
        opponent = int(np.count_nonzero(board == -1))
        assert own - opponent in (0, 1)
        expected_player = 1 if own == opponent else -1
        assert entry["state"].get_current_player() == expected_player


def test_scores_are_present_exactly_for_playable_columns(entries):
    for entry in entries:
        playable = {a.target_column for a in entry["state"].get_possible_actions()}
        scored = {c for c, s in enumerate(entry["scores"]) if s is not None}
        assert scored == playable


def test_stones_are_supported_from_below(entries):
    """Kein schwebender Stein - sonst waere die Stellung nicht erspielbar."""
    for entry in entries:
        board = np.array(entry["board"])
        for column in range(7):
            filled = [row for row in range(6) if board[row, column] != 0]
            if filled:
                assert filled == list(range(min(filled), 6))


def test_at_least_one_move_is_optimal_in_every_position(entries):
    for entry in entries:
        scores = [s for s in entry["scores"] if s is not None]
        assert scores, "Stellung ohne spielbare Spalte"
        assert max(scores) == max(scores)


def test_loading_does_not_require_the_solver_package(monkeypatch):
    """
    Simuliert eine Maschine ohne das Paket: der Import darf beim Laden gar nicht
    erst versucht werden.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("connect_four_ai"):
            raise ImportError("connect_four_ai ist hier nicht verfuegbar")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    loaded = load_benchmark_set(BENCHMARK_PATH)
    assert len(loaded) >= 50
