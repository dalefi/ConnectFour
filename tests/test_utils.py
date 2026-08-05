import os
import time

import pytest

from src.utils import (
    calculate_percentages,
    gating_win_rate,
    latest_model_path,
    temperature_for_move,
)


# ---------------------------------------------------------------------------
# Neuesten Checkpoint finden
# ---------------------------------------------------------------------------

def test_latest_model_path_picks_the_newest_checkpoint(tmp_path):
    older = tmp_path / "cfnet_alt.pt"
    newer = tmp_path / "cfnet_neu.pt"
    older.write_bytes(b"x")
    time.sleep(0.01)
    newer.write_bytes(b"x")
    os.utime(older, (1, 1))

    assert latest_model_path(tmp_path) == str(newer)


def test_latest_model_path_ignores_other_files(tmp_path):
    (tmp_path / "notizen.txt").write_text("kein Modell")
    (tmp_path / "cfnet.pt").write_bytes(b"x")

    assert latest_model_path(tmp_path) == str(tmp_path / "cfnet.pt")


def test_latest_model_path_returns_none_for_an_empty_directory(tmp_path):
    assert latest_model_path(tmp_path) is None


def test_latest_model_path_returns_none_for_a_missing_directory(tmp_path):
    assert latest_model_path(tmp_path / "gibtsnicht") is None


# ---------------------------------------------------------------------------
# Temperatur-Zeitplan
# ---------------------------------------------------------------------------

def test_early_moves_are_sampled_with_full_temperature():
    assert temperature_for_move(0) == 1.0
    assert temperature_for_move(7) == 1.0


def test_later_moves_are_played_greedily():
    """
    Ueber die ganze Partie proportional zu den Besuchen zu ziehen verschenkt
    gewonnene Stellungen und vergiftet damit die Value-Targets.
    """
    assert temperature_for_move(8) == 0.0
    assert temperature_for_move(30) == 0.0


def test_exploration_length_is_configurable():
    assert temperature_for_move(3, exploration_moves=2) == 0.0
    assert temperature_for_move(1, exploration_moves=2) == 1.0


# ---------------------------------------------------------------------------
# Gating-Kriterium
# ---------------------------------------------------------------------------

def test_gating_ignores_draws():
    """
    Unentschieden im Nenner macht die 55%-Huerde unerreichbar. AlphaGo Zero
    zaehlt nur entschiedene Partien.
    """
    assert gating_win_rate({"challenger": 6, "champion": 4, "draw": 90}) == pytest.approx(0.6)


def test_gating_counts_only_decided_games():
    assert gating_win_rate({"challenger": 30, "champion": 20, "draw": 0}) == pytest.approx(0.6)


def test_gating_returns_zero_when_every_game_is_drawn():
    assert gating_win_rate({"challenger": 0, "champion": 0, "draw": 50}) == 0.0


def test_gating_tolerates_missing_keys():
    assert gating_win_rate({"challenger": 3}) == pytest.approx(1.0)
    assert gating_win_rate({}) == 0.0


# ---------------------------------------------------------------------------
# Bestehende Prozentrechnung
# ---------------------------------------------------------------------------

def test_calculate_percentages_sums_to_one():
    percentages = calculate_percentages({"challenger": 1, "champion": 2, "draw": 1})
    assert sum(percentages.values()) == pytest.approx(1.0)
    assert percentages["champion"] == pytest.approx(0.5)


def test_calculate_percentages_handles_an_empty_tally():
    assert calculate_percentages({"challenger": 0, "champion": 0}) == {
        "challenger": 0.0,
        "champion": 0.0,
    }
