import numpy as np
import pytest
import torch

from conftest import board_from_rows
from src.CFNet import CFNet, encode_board, mirror_board, mirror_policy, state_to_tensor
from src.ConnectFour import ConnectFour


# ---------------------------------------------------------------------------
# Eingabekodierung: kanonisch aus Sicht des Spielers am Zug
# ---------------------------------------------------------------------------

def test_encoding_has_two_binary_planes():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "xo.....",
    ], current_player=1)

    tensor = state_to_tensor(state)

    assert tensor.shape == (2, 6, 7)
    assert set(tensor.unique().tolist()) <= {0.0, 1.0}


def test_first_plane_holds_the_stones_of_the_player_to_move():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "xo.....",
    ], current_player=1)

    own, opponent = state_to_tensor(state)

    assert own[5, 0] == 1.0 and own[5, 1] == 0.0
    assert opponent[5, 1] == 1.0 and opponent[5, 0] == 0.0


def test_the_same_position_from_the_other_side_swaps_the_planes():
    as_player_one = state_to_tensor(board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "xo.....",
    ], current_player=-1))

    own, opponent = as_player_one
    assert own[5, 1] == 1.0     # -1 ist am Zug, also gehoert 'o' zu "own"
    assert opponent[5, 0] == 1.0


def test_colour_swapped_positions_encode_identically():
    """
    Der Kern der Kanonisierung: eine Stellung und ihr farbvertauschtes Gegenstueck
    sind dasselbe Problem. Das Netz soll das geschenkt bekommen, statt es lernen
    zu muessen.
    """
    original = board_from_rows([
        ".......",
        ".......",
        ".......",
        "..x....",
        "..o....",
        ".xox...",
    ], current_player=1)

    swapped = board_from_rows([
        ".......",
        ".......",
        ".......",
        "..o....",
        "..x....",
        ".oxo...",
    ], current_player=-1)

    assert torch.equal(state_to_tensor(original), state_to_tensor(swapped))


def test_empty_board_encodes_to_all_zeros():
    tensor = state_to_tensor(ConnectFour())
    assert torch.count_nonzero(tensor) == 0


def test_encoding_rejects_something_that_is_not_a_state():
    with pytest.raises(TypeError):
        state_to_tensor("kein Spielzustand")


# ---------------------------------------------------------------------------
# Trainings- und Suchkodierung muessen identisch sein
# ---------------------------------------------------------------------------

def test_batch_encoding_agrees_with_single_encoding():
    """
    Der Datensatz kodiert den ganzen Buffer auf einmal (vektorisiert), die Suche
    kodiert einzelne Stellungen. Beide Wege muessen exakt dasselbe liefern.
    """
    from src.CFNet import encode_boards

    rng = np.random.RandomState(3)
    boards = rng.randint(-1, 2, size=(64, 6, 7)).astype(np.int8)
    players = rng.choice([-1, 1], size=64).astype(np.int8)

    batch = encode_boards(boards, players)

    assert batch.shape == (64, 2, 6, 7)
    for index in range(64):
        assert torch.equal(batch[index], encode_board(boards[index], int(players[index])))


def test_dataset_keeps_its_tensors_in_one_block():
    """
    Frueher wurde pro Zugriff np.array/torch.tensor/torch.stack aufgerufen - bei
    100k Zuegen und 20 Epochen sind das Millionen Python-Konvertierungen.
    """
    from src.generate_training_data import MoveDataset

    class Row:
        board_state = [[0] * 7 for _ in range(5)] + [[1, -1, 0, 0, 0, 0, 0]]
        policy = [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        value = 0.25
        current_player = 1

    dataset = MoveDataset([Row() for _ in range(10)], augment=False)

    assert dataset.inputs.shape == (10, 2, 6, 7)
    assert dataset.policies.shape == (10, 7)
    assert dataset.values.shape == (10,)


def test_dataset_length_matches_the_number_of_moves():
    from src.generate_training_data import MoveDataset

    class Row:
        board_state = [[0] * 7 for _ in range(6)]
        policy = [1 / 7] * 7
        value = 0.0
        current_player = 1

    assert len(MoveDataset([Row() for _ in range(5)], augment=False)) == 5


def test_dataset_encoding_matches_search_encoding():
    """
    Waehrend der Suche kommt der Tensor aus state_to_tensor, beim Training aus
    dem Datensatz. Weichen die beiden ab, trainiert das Netz auf einer anderen
    Repraesentation als es spaeter sieht - und das faellt sonst nirgends auf.
    """
    from src.generate_training_data import MoveDataset

    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        "..x....",
        "..o....",
        ".xox...",
    ], current_player=-1)

    class Row:
        board_state = state.board.tolist()
        policy = [1 / 7] * 7
        value = 0.5
        current_player = -1

    dataset = MoveDataset([Row()], augment=False)
    dataset_tensor, _, _ = dataset[0]

    assert torch.equal(dataset_tensor, state_to_tensor(state))


# ---------------------------------------------------------------------------
# Spiegelsymmetrie
# ---------------------------------------------------------------------------

def test_mirroring_flips_the_board_left_to_right():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
    ], dtype=np.int8)

    mirrored = mirror_board(board)

    assert mirrored[5, 6] == 1
    assert mirrored[5, 5] == -1
    assert mirrored[5, 0] == 0


def test_mirroring_a_policy_reverses_the_columns():
    policy = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02])
    assert mirror_policy(policy) == pytest.approx(policy[::-1])


def test_mirroring_twice_is_the_identity():
    board = np.random.RandomState(0).randint(-1, 2, size=(6, 7)).astype(np.int8)
    assert np.array_equal(mirror_board(mirror_board(board)), board)


def test_a_mirrored_position_keeps_its_winner():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "xxxx...",
    ])
    mirrored = ConnectFour(board=mirror_board(state.board))
    assert mirrored.get_winner() == state.get_winner()


def test_augmentation_mirrors_board_and_policy_together():
    from src.generate_training_data import MoveDataset

    class Row:
        board_state = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0, 0],
        ]
        policy = [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        value = 0.25
        current_player = 1

    plain = MoveDataset([Row()], augment=False)
    board_plain, policy_plain, value_plain = plain[0]

    mirrored_dataset = MoveDataset([Row()], augment=True)
    mirrored_dataset._force_mirror = True
    board_mirrored, policy_mirrored, value_mirrored = mirrored_dataset[0]

    assert torch.equal(torch.flip(board_plain, dims=[2]), board_mirrored)
    assert policy_mirrored == pytest.approx(torch.flip(policy_plain, dims=[0]))
    assert value_mirrored == pytest.approx(value_plain), "Spiegeln aendert den Ausgang nicht"


def test_augmentation_off_is_deterministic():
    from src.generate_training_data import MoveDataset

    class Row:
        board_state = [[0] * 7 for _ in range(5)] + [[1, -1, 0, 0, 0, 0, 0]]
        policy = [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        value = 0.25
        current_player = 1

    dataset = MoveDataset([Row()], augment=False)
    first = dataset[0][0]
    for _ in range(20):
        assert torch.equal(dataset[0][0], first)


# ---------------------------------------------------------------------------
# Netz
# ---------------------------------------------------------------------------

def test_forward_returns_value_and_policy_with_the_right_shapes():
    model = CFNet()
    model.eval()

    batch = torch.zeros(4, 2, 6, 7)
    out = model(batch)

    assert out["value"].shape == (4, 1)
    assert out["policy"].shape == (4, 7)


def test_value_head_stays_within_minus_one_and_one():
    model = CFNet()
    model.eval()

    batch = torch.randn(16, 2, 6, 7)
    values = model(batch)["value"]

    assert bool((values >= -1).all()) and bool((values <= 1).all())


def test_forward_accepts_a_game_state_directly():
    model = CFNet()
    model.eval()

    out = model(ConnectFour())

    assert out["value"].shape == (1, 1)
    assert out["policy"].shape == (1, 7)


def test_evaluation_is_deterministic():
    """
    Zwei gestapelte Dropout-Schichten mit p=0.5 haben den Value-Head massiv
    gedaempft. Im eval-Modus muss dieselbe Eingabe dieselbe Ausgabe liefern.
    """
    model = CFNet()
    model.eval()

    batch = torch.randn(8, 2, 6, 7)
    with torch.no_grad():
        first = model(batch)["value"]
        second = model(batch)["value"]

    assert torch.allclose(first, second)


def test_network_has_no_dropout_by_default():
    model = CFNet()
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert all(layer.p == 0.0 for layer in dropout_layers)


def test_a_single_position_can_be_evaluated_in_eval_mode():
    """
    Batchnorm mit Batchgroesse 1 wuerde im train-Modus scheitern; die Suche
    bewertet aber staendig einzelne Stellungen.
    """
    model = CFNet()
    model.eval()
    out = model(torch.zeros(1, 2, 6, 7))
    assert out["value"].shape == (1, 1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_alphaloss_reports_its_two_components():
    model = CFNet()

    value_prediction = torch.zeros(4, 1)
    policy_logits = torch.zeros(4, 7)
    value_target = torch.zeros(4)
    policy_target = torch.full((4, 7), 1 / 7)

    total, value_loss, policy_loss = model.alphaloss(
        value_prediction, policy_logits, value_target, policy_target
    )

    assert total == pytest.approx(float(value_loss) + float(policy_loss))
    assert float(value_loss) == pytest.approx(0.0, abs=1e-6)


def test_policy_kl_is_zero_for_a_perfect_prediction():
    """
    Der Cross-Entropy-Term enthaelt die Entropie des Ziels als nicht reduzierbaren
    Sockel (~1.95 bei Gleichverteilung). Die KL zeigt den echten Fortschritt.
    """
    model = CFNet()
    policy_target = torch.full((4, 7), 1 / 7)
    perfect_logits = torch.zeros(4, 7)

    kl = model.policy_kl(perfect_logits, policy_target)

    assert float(kl) == pytest.approx(0.0, abs=1e-6)


def test_policy_kl_is_positive_for_a_wrong_prediction():
    model = CFNet()
    policy_target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    wrong_logits = torch.zeros(1, 7)

    assert float(model.policy_kl(wrong_logits, policy_target)) > 1.0


def test_value_loss_grows_with_the_error():
    model = CFNet()
    policy_logits = torch.zeros(2, 7)
    policy_target = torch.full((2, 7), 1 / 7)

    _, small_error, _ = model.alphaloss(
        torch.zeros(2, 1), policy_logits, torch.full((2,), 0.1), policy_target
    )
    _, large_error, _ = model.alphaloss(
        torch.zeros(2, 1), policy_logits, torch.full((2,), 0.9), policy_target
    )

    assert float(large_error) > float(small_error)
