import json

import numpy as np
import pytest

from src.ConnectFour import ConnectFour
from src.example_games import (
    make_start_positions,
    model_generation_order,
    play_example_game,
    write_game,
    write_manifest,
)
from src.example_games_viewer import build_viewer


class UniformEvaluator:
    """
    Minimaler Ersatz fuer den NeuralNetBatcher: Gleichverteilung und Value 0.
    Damit haengt das Ergebnis allein am Suchbaum und die Partien laufen schnell.
    """

    def __init__(self):
        self.model = None

    async def get_policy_value(self, state_tensor):
        return 0.0, np.ones(7) / 7


# ---------------------------------------------------------------------------
# Startstellungen
# ---------------------------------------------------------------------------

def test_start_positions_are_reproducible_for_a_given_seed():
    """
    Der Kern des ganzen Vorhabens: alle Modelle muessen dieselben Stellungen
    spielen, sonst ist "gleiche Stellung, verschiedenes Modell" nicht moeglich.
    """
    first = make_start_positions(8, seed=123)
    second = make_start_positions(8, seed=123)

    assert first == second
    assert make_start_positions(8, seed=124) != first


def test_first_start_position_is_the_empty_board():
    """Die interessanteste einzelne Vergleichsstellung im Vortrag."""
    positions = make_start_positions(5, seed=1)

    assert positions[0]["num_random_moves"] == 0
    assert np.array_equal(positions[0]["board"], np.zeros((6, 7), dtype=int).tolist())
    assert positions[0]["current_player"] == 1


def test_start_positions_are_distinct_playable_and_reachable():
    positions = make_start_positions(10, max_random_moves=4, seed=7)

    boards = [json.dumps(position["board"]) for position in positions]
    assert len(set(boards)) == len(positions)

    for position in positions:
        board = np.array(position["board"], dtype=np.int8)
        state = ConnectFour(board=board, currentPlayer=position["current_player"])

        assert not state.is_terminal()

        # Steinzahlen muessen zum Spieler am Zug passen, sonst ist die Stellung
        # unerreichbar und jede Bewertung darauf sinnlos.
        difference = int(np.count_nonzero(board == 1)) - int(np.count_nonzero(board == -1))
        expected_player = 1 if difference == 0 else -1
        assert position["current_player"] == expected_player
        assert int(np.count_nonzero(board)) == position["num_random_moves"]


# ---------------------------------------------------------------------------
# Reihenfolge der Modelle
# ---------------------------------------------------------------------------

def test_generation_order_puts_the_initial_model_first():
    """
    Alphabetisch landet 'cfnet_initial' hinter den Zeitstempeln - der Dateiname
    darf die Trainingsreihenfolge also nicht bestimmen.
    """
    paths = [
        "accepted_models/cfnet_20260806_015829.pt",
        "accepted_models/cfnet_initial.pt",
        "accepted_models/cfnet_20260805_223338.pt",
    ]

    ordered = [path.name for path in model_generation_order(paths)]

    assert ordered == [
        "cfnet_initial.pt",
        "cfnet_20260805_223338.pt",
        "cfnet_20260806_015829.pt",
    ]


# ---------------------------------------------------------------------------
# Partie-Aufzeichnung
# ---------------------------------------------------------------------------

async def record_a_game():
    positions = make_start_positions(3, max_random_moves=4, seed=42)
    return await play_example_game(
        batcher=UniformEvaluator(),
        start_position=positions[2],
        iteration_limit=20,
    )


@pytest.mark.asyncio
async def test_recorded_moves_replay_into_the_stored_boards():
    """
    Faengt Off-by-one, das Verwechseln von "Brett vor/nach dem Zug" und einen
    falsch gespeicherten played_move: alles Fehler, die im Viewer still bleiben.
    """
    game = await record_a_game()
    start = game["start_position"]

    state = ConnectFour(
        board=np.array(start["board"], dtype=np.int8),
        currentPlayer=start["current_player"],
    )

    for move in game["moves"]:
        assert move["board"] == state.board.tolist()
        assert move["current_player"] == state.get_current_player()
        state.make_move(move["played_move"])

    assert state.is_terminal()
    assert game["outcome"]["winner"] == int(state.get_winner())
    assert game["outcome"]["num_moves"] == len(game["moves"])


@pytest.mark.asyncio
async def test_every_move_carries_both_evaluations_and_both_policies():
    game = await record_a_game()

    for move in game["moves"]:
        assert -1.0 <= move["nn_eval"] <= 1.0
        assert -1.0 <= move["mcts_value"] <= 1.0
        assert sum(move["nn_policy"]) == pytest.approx(1.0, abs=1e-4)
        assert sum(move["mcts_policy"]) == pytest.approx(1.0, abs=1e-4)
        assert sum(move["visits"]) > 0
        assert move["solver_scores"] is None


@pytest.mark.asyncio
async def test_policies_never_recommend_a_full_column():
    game = await record_a_game()

    for move in game["moves"]:
        board = np.array(move["board"], dtype=np.int8)
        for column in range(7):
            if board[0, column] != 0:
                assert move["nn_policy"][column] == 0.0
                assert move["mcts_policy"][column] == 0.0


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_viewer_embeds_the_games_it_finds(tmp_path):
    positions = make_start_positions(1, seed=5)
    game = await play_example_game(
        batcher=UniformEvaluator(),
        start_position=positions[0],
        iteration_limit=20,
    )

    config = {"iteration_limit": 20, "seed": 5}
    model = {"tag": "cfnet_test", "path": "accepted_models/cfnet_test.pt", "generation": 0}
    write_game(tmp_path / "cfnet_test", 0, game, model, config)
    write_manifest(tmp_path, positions, [model], config)

    viewer = build_viewer(tmp_path)
    html = viewer.read_text(encoding="utf-8")

    assert viewer.name == "viewer.html"
    assert "cfnet_test" in html
    assert str(game["moves"][0]["played_move"]) in html
    assert "EXAMPLE_GAMES" in html
