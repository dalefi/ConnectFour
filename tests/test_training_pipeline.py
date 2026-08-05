import numpy as np
import pytest
import torch

from conftest import board_from_rows
from src.CFNet import CFNet
from src.ConnectFour import ConnectFour
from src.generate_training_data import MoveDataset
from src.selfplay_parallel import update_statistics_and_db


class RecordingDatabase:
    """Nimmt statt der echten Datenbank auf, was geschrieben werden wuerde."""

    def __init__(self):
        self.games = []

    def insert_selfplay_game(self, winner, moves, model_1_tag, model_2_tag):
        self.games.append({
            "winner": winner,
            "moves": moves,
            "model_1_tag": model_1_tag,
            "model_2_tag": model_2_tag,
        })


def play_to_the_end(columns):
    """Spielt die Zugfolge und protokolliert sie im Format der Selfplay-Schleife."""
    state = ConnectFour()
    game_moves = []

    for move_number, column in enumerate(columns):
        game_moves.append({
            "move_number": move_number,
            "board_state": state.board.tolist(),
            "policy": (np.ones(7) / 7).tolist(),
            "nn_policy": (np.ones(7) / 7).tolist(),
            "nn_eval": 0.0,
            "current_player": state.get_current_player(),
            "model_role": "current" if state.get_current_player() == 1 else "updated",
        })
        state.make_move(column)

    assert state.is_terminal(), "Testaufbau: die Partie muss zu Ende gespielt sein"
    return state, game_moves


# ---------------------------------------------------------------------------
# Value-Targets: Vorzeichen ueber die ganze Partie
# ---------------------------------------------------------------------------

def test_value_target_is_plus_one_exactly_for_the_winner():
    """
    Jede Stellung wird mit dem Ausgang aus Sicht des Spielers am Zug etikettiert.
    Ein Vorzeichenfehler hier ist unsichtbar und macht das Training wertlos.
    """
    state, game_moves = play_to_the_end([0, 1, 0, 1, 0, 1, 0])  # Spieler 1 gewinnt
    winner = state.get_winner()
    assert winner == 1

    database = RecordingDatabase()
    statistics = {"wins_current_model": 0, "wins_updated_model": 0, "draws": 0}

    update_statistics_and_db(
        state, "current", game_moves, statistics, database, "champion_tag", "challenger_tag"
    )

    stored = database.games[0]["moves"]
    assert len(stored) == len(game_moves)

    for move in stored:
        expected = 1.0 if move["current_player"] == winner else -1.0
        assert move["value"] == pytest.approx(expected)


def test_the_last_recorded_position_belongs_to_the_winner():
    state, game_moves = play_to_the_end([0, 1, 0, 1, 0, 1, 0])

    database = RecordingDatabase()
    update_statistics_and_db(
        state, "current", game_moves,
        {"wins_current_model": 0, "wins_updated_model": 0, "draws": 0},
        database, "champion_tag", "challenger_tag"
    )

    last_move = database.games[0]["moves"][-1]
    assert last_move["current_player"] == state.get_winner()
    assert last_move["value"] == pytest.approx(1.0)


def test_value_targets_alternate_between_consecutive_positions():
    state, game_moves = play_to_the_end([0, 1, 0, 1, 0, 1, 0])

    database = RecordingDatabase()
    update_statistics_and_db(
        state, "current", game_moves,
        {"wins_current_model": 0, "wins_updated_model": 0, "draws": 0},
        database, "champion_tag", "challenger_tag"
    )

    values = [move["value"] for move in database.games[0]["moves"]]
    for earlier, later in zip(values, values[1:]):
        assert earlier == pytest.approx(-later)


def test_a_drawn_game_labels_every_position_with_zero():
    board = board_from_rows([
        "xoooxx.",
        "xoxoxoo",
        "oxoooxo",
        "oooxxxo",
        "xxxoxox",
        "xxoxoxo",
    ], current_player=1)
    # letzte freie Zelle schliessen -> Remis
    state = board
    game_moves = [{
        "move_number": 0,
        "board_state": state.board.tolist(),
        "policy": (np.ones(7) / 7).tolist(),
        "nn_policy": (np.ones(7) / 7).tolist(),
        "nn_eval": 0.0,
        "current_player": 1,
        "model_role": "current",
    }]
    state.make_move(6)
    assert state.is_terminal() and state.get_winner() == 0

    database = RecordingDatabase()
    statistics = {"wins_current_model": 0, "wins_updated_model": 0, "draws": 0}
    update_statistics_and_db(
        state, "current", game_moves, statistics, database, "champion_tag", "challenger_tag"
    )

    assert statistics["draws"] == 1
    assert database.games[0]["winner"] == "draw"
    assert all(move["value"] == pytest.approx(0.0) for move in database.games[0]["moves"])


def test_the_winning_role_is_recorded_for_gating():
    state, game_moves = play_to_the_end([0, 1, 0, 1, 0, 1, 0])

    database = RecordingDatabase()
    statistics = {"wins_current_model": 0, "wins_updated_model": 0, "draws": 0}
    update_statistics_and_db(
        state, "updated", game_moves, statistics, database, "champion_tag", "challenger_tag"
    )

    assert statistics["wins_updated_model"] == 1
    assert database.games[0]["winner"] == "challenger"


# ---------------------------------------------------------------------------
# Ein echter Trainingsschritt
# ---------------------------------------------------------------------------

class SyntheticMove:
    def __init__(self, board, policy, value, current_player):
        self.board_state = board
        self.policy = policy
        self.value = value
        self.current_player = current_player


def make_learnable_dataset(size=256, seed=0):
    """
    Stellungen, in denen Value und Policy eindeutig aus dem Brett folgen:
    der Wert haengt an der Steindifferenz, die Policy zeigt auf die leerste Spalte.
    """
    rng = np.random.RandomState(seed)
    moves = []

    for _ in range(size):
        board = np.zeros((6, 7), dtype=np.int8)
        for column in range(7):
            height = rng.randint(0, 5)
            for row in range(height):
                board[5 - row, column] = 1 if (row + column) % 2 == 0 else -1

        heights = (board != 0).sum(axis=0)
        target_column = int(np.argmin(heights))
        policy = np.zeros(7, dtype=np.float32)
        policy[target_column] = 1.0

        value = float(np.tanh((board == 1).sum() - (board == -1).sum()))
        moves.append(SyntheticMove(board.tolist(), policy.tolist(), value, 1))

    return MoveDataset(moves, augment=False)


@pytest.mark.slow
def test_training_reduces_the_loss_on_a_learnable_dataset():
    """
    Ende-zu-Ende-Kontrolle ueber Kodierung, Netz und Loss: auf Daten mit klarem
    Muster muss der Loss deutlich fallen.
    """
    torch.manual_seed(0)
    dataset = make_learnable_dataset()
    model = CFNet()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = dataset.inputs
    policies = dataset.policies
    values = dataset.values

    losses = []
    for step in range(60):
        optimizer.zero_grad()
        out = model(inputs)
        loss, _, _ = model.alphaloss(out["value"], out["policy"], values, policies)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.6, f"Loss faellt kaum: {losses[0]:.3f} -> {losses[-1]:.3f}"


@pytest.mark.slow
def test_policy_kl_falls_during_training():
    torch.manual_seed(0)
    dataset = make_learnable_dataset(size=128)
    model = CFNet()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def current_kl():
        model.eval()
        with torch.no_grad():
            out = model(dataset.inputs)
            kl = model.policy_kl(out["policy"], dataset.policies).item()
        model.train()
        return kl

    before = current_kl()
    for _ in range(60):
        optimizer.zero_grad()
        out = model(dataset.inputs)
        loss, _, _ = model.alphaloss(out["value"], out["policy"], dataset.values, dataset.policies)
        loss.backward()
        optimizer.step()

    assert current_kl() < before
