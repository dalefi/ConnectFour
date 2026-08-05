import math

import numpy as np
import pytest

from conftest import board_from_rows
from mcts.searcher.mcts_searcher import TreeNode, mcts_searcher
from src.ConnectFour import Action, ConnectFour


# ---------------------------------------------------------------------------
# Ein einfacher, deterministischer Ersatz fuer den NeuralNetBatcher.
# Kein Mock-Framework: eine echte, minimale Implementierung derselben
# async-Schnittstelle, damit die Suche allein am Baum gemessen wird.
# ---------------------------------------------------------------------------

class ScriptedEvaluator:
    """
    Liefert feste Policy/Value-Paare. Standard: Gleichverteilung und Value 0,
    sodass jedes Suchergebnis ausschliesslich aus dem Baum stammt.
    """

    def __init__(self, policy=None, value=0.0, policy_by_board=None):
        self.policy = np.ones(7) / 7 if policy is None else np.asarray(policy, dtype=float)
        self.value = value
        self.policy_by_board = policy_by_board or {}
        self.model = None
        self.calls = 0

    async def get_policy_value(self, state_tensor):
        self.calls += 1
        key = tuple(np.asarray(state_tensor[0]).flatten().tolist())
        if key in self.policy_by_board:
            return self.policy_by_board[key]
        return self.value, self.policy.copy()


def make_searcher(evaluator=None, iteration_limit=200, exploration_constant=1.5):
    return mcts_searcher(
        iteration_limit=iteration_limit,
        exploration_constant=exploration_constant,
        batcher=evaluator or ScriptedEvaluator(),
    )


def entropy(distribution):
    distribution = np.asarray(distribution, dtype=float)
    nonzero = distribution[distribution > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


# ---------------------------------------------------------------------------
# PUCT-Formel
# ---------------------------------------------------------------------------

def test_ucb_matches_the_alphazero_puct_formula():
    parent = TreeNode(ConnectFour(), parent=None)
    parent.numVisits = 100
    parent.policy = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

    child = TreeNode(ConnectFour(), parent=parent)
    child.numVisits = 9
    child.totalReward = -3.0  # aus Sicht des Kindes

    action = Action(target_column=0, player=1)
    exploration_constant = 1.5

    expected_q = 3.0 / 9.0  # -(totalReward / numVisits)
    expected_u = exploration_constant * 0.5 * math.sqrt(100) / (1 + 9)

    assert parent.get_ucb(action, child, exploration_constant) == pytest.approx(
        expected_q + expected_u
    )


def test_exploration_term_shrinks_as_the_child_is_visited():
    """
    Der Kern des alten Fehlers: sqrt(N/(n+1)) faellt viel zu langsam, deshalb
    konvergierte die Suche nie. Korrekt ist P*sqrt(N)/(1+n) - der Term muss
    umgekehrt proportional zu den Kindbesuchen abfallen.
    """
    parent = TreeNode(ConnectFour(), parent=None)
    parent.numVisits = 400
    parent.policy = np.ones(7) / 7
    action = Action(target_column=3, player=1)

    def exploration_only(child_visits):
        child = TreeNode(ConnectFour(), parent=parent)
        child.numVisits = child_visits
        child.totalReward = 0.0
        return parent.get_ucb(action, child, 1.5)

    at_10 = exploration_only(10)
    at_100 = exploration_only(100)

    assert at_100 == pytest.approx(at_10 * (1 + 10) / (1 + 100), rel=1e-9)


@pytest.mark.asyncio
async def test_more_iterations_produce_a_sharper_policy():
    """
    Regressionstest fuer den nicht konvergierenden Suchbaum: mehr Iterationen
    muessen zu einer entschiedeneren Besuchsverteilung fuehren.
    """
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "..o....",
        "xxx.oo.",
    ], current_player=1)

    _, _, policy_few = await make_searcher(iteration_limit=60).search(state)
    _, _, policy_many = await make_searcher(iteration_limit=600).search(state)

    assert entropy(policy_many) < entropy(policy_few) - 0.2


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------

def test_backpropagate_alternates_the_sign_up_the_tree():
    root = TreeNode(ConnectFour(), parent=None)
    child = TreeNode(ConnectFour(), parent=root)
    grandchild = TreeNode(ConnectFour(), parent=child)

    mcts_searcher.backpropagate(grandchild, 1.0)

    assert grandchild.totalReward == pytest.approx(1.0)
    assert child.totalReward == pytest.approx(-1.0)
    assert root.totalReward == pytest.approx(1.0)
    assert (grandchild.numVisits, child.numVisits, root.numVisits) == (1, 1, 1)


# ---------------------------------------------------------------------------
# Maskierung unmoeglicher Zuege
# ---------------------------------------------------------------------------

def test_masking_removes_full_columns():
    state = ConnectFour()
    for _ in range(6):
        state.make_move(3)

    node = TreeNode(state, parent=None)
    masked = mcts_searcher.mask_invalid_actions(node, np.ones(7) / 7)

    assert masked[3] == 0.0
    assert masked.sum() == pytest.approx(1.0)


def test_masking_falls_back_to_uniform_when_the_network_gives_zero():
    """
    Frueher wurde hier durch 0 geteilt; numpy warnt dabei nur und liefert NaN,
    der bare except hat also nie gegriffen.
    """
    state = ConnectFour()
    node = TreeNode(state, parent=None)

    masked = mcts_searcher.mask_invalid_actions(node, np.zeros(7))

    assert not np.isnan(masked).any()
    assert masked.sum() == pytest.approx(1.0)
    assert masked == pytest.approx(np.ones(7) / 7)


def test_masking_keeps_the_relative_weights_of_legal_moves():
    state = ConnectFour()
    for _ in range(6):
        state.make_move(0)

    node = TreeNode(state, parent=None)
    raw = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
    masked = mcts_searcher.mask_invalid_actions(node, raw)

    assert masked[0] == 0.0
    assert masked[1] / masked[2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Dirichlet-Rauschen an der Wurzel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_root_noise_is_off_by_default():
    evaluator = ScriptedEvaluator(policy=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    searcher = make_searcher(evaluator, iteration_limit=5)

    await searcher.search(ConnectFour())

    assert searcher.root.policy == pytest.approx(searcher.root.prior)


@pytest.mark.asyncio
async def test_root_noise_perturbs_the_root_prior():
    evaluator = ScriptedEvaluator(policy=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    searcher = make_searcher(evaluator, iteration_limit=5)

    await searcher.search(ConnectFour(), add_noise=True)

    assert searcher.root.policy.sum() == pytest.approx(1.0)
    assert searcher.root.policy != pytest.approx(searcher.root.prior)


@pytest.mark.asyncio
async def test_root_noise_never_makes_an_illegal_move_possible():
    state = ConnectFour()
    for _ in range(6):
        state.make_move(2)

    searcher = make_searcher(iteration_limit=5)
    await searcher.search(state, add_noise=True)

    assert searcher.root.policy[2] == 0.0


@pytest.mark.asyncio
async def test_reported_network_policy_is_free_of_noise():
    """
    nn_policy wird als Diagnosewert in die Datenbank geschrieben und muss die
    unveraenderte Netzausgabe sein, nicht die verrauschte Suchwurzel.
    """
    evaluator = ScriptedEvaluator(policy=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    searcher = make_searcher(evaluator, iteration_limit=5)

    _, nn_policy, _ = await searcher.search(ConnectFour(), add_noise=True)

    assert nn_policy == pytest.approx([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


@pytest.mark.asyncio
async def test_noise_only_applies_to_the_root_not_to_children():
    evaluator = ScriptedEvaluator(policy=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    searcher = make_searcher(evaluator, iteration_limit=200)

    await searcher.search(ConnectFour(), add_noise=True)

    visited_children = [c for c in searcher.root.children.values() if c.prior is not None]
    assert visited_children, "Die Suche hat keine Kinder ausgewertet"
    for child in visited_children:
        assert child.policy == pytest.approx(child.prior)


# ---------------------------------------------------------------------------
# Temperatur
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_temperature_one_returns_visit_proportions():
    searcher = make_searcher(iteration_limit=200)
    await searcher.search(ConnectFour(), temperature=1.0)

    visits = np.zeros(7)
    for action, child in searcher.root.children.items():
        visits[action.target_column] = child.numVisits

    policy = searcher.get_policy_from_child_visits(temperature=1.0)
    assert policy == pytest.approx(visits / visits.sum())


@pytest.mark.asyncio
async def test_temperature_zero_is_greedy():
    searcher = make_searcher(iteration_limit=200)
    await searcher.search(ConnectFour())

    policy = searcher.get_policy_from_child_visits(temperature=0.0)

    assert policy.sum() == pytest.approx(1.0)
    assert sorted(policy)[-1] == pytest.approx(1.0)
    assert np.count_nonzero(policy) == 1


@pytest.mark.asyncio
async def test_lower_temperature_sharpens_the_policy():
    searcher = make_searcher(iteration_limit=300)
    await searcher.search(ConnectFour())

    warm = searcher.get_policy_from_child_visits(temperature=1.0)
    cool = searcher.get_policy_from_child_visits(temperature=0.25)

    assert entropy(cool) < entropy(warm)


# ---------------------------------------------------------------------------
# Terminale Knoten
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_losing_terminal_child_is_valued_minus_one():
    """
    Nach einem Gewinnzug ist der Spieler am Zug der Verlierer, der Knotenwert
    aus dessen Sicht also -1.
    """
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "..o....",
        "xxx.oo.",
    ], current_player=1)

    searcher = make_searcher(iteration_limit=200)
    await searcher.search(state)

    winning_child = next(
        child for action, child in searcher.root.children.items()
        if action.target_column == 3
    )
    assert winning_child.is_terminal
    assert winning_child.value == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Wiederverwendung des Suchbaums
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subtree_is_reused_after_advancing_the_root():
    searcher = make_searcher(iteration_limit=200)
    state = ConnectFour()
    await searcher.search(state)

    chosen_column = 3
    visits_before = next(
        child.numVisits for action, child in searcher.root.children.items()
        if action.target_column == chosen_column
    )
    assert visits_before > 1, "Testaufbau: das Kind sollte besucht worden sein"

    state.make_move(chosen_column)
    await searcher.search(state)

    assert searcher.root.numVisits > 200, (
        "Die Wurzel muss die Besuche aus dem wiederverwendeten Teilbaum behalten"
    )


@pytest.mark.asyncio
async def test_an_unrelated_position_starts_a_fresh_tree():
    searcher = make_searcher(iteration_limit=200)
    await searcher.search(ConnectFour())

    unrelated = board_from_rows([
        ".......",
        ".......",
        ".......",
        "..xo...",
        "..ox...",
        "..xo...",
    ], current_player=1)
    await searcher.search(unrelated)

    assert searcher.root.numVisits <= 201


def _assert_tree_is_consistent(node, depth=0):
    """
    Jedes Kind muss genau einen Stein mehr haben als sein Elternknoten und der
    andere Spieler muss am Zug sein. Bricht das, zeigt der Baum auf eine andere
    Stellung als die, fuer die er wiederverwendet wird.
    """
    stones = int(np.count_nonzero(node.state.board))
    for child in node.children.values():
        assert int(np.count_nonzero(child.state.board)) == stones + 1
        assert child.state.get_current_player() == -node.state.get_current_player()
        if depth < 2:
            _assert_tree_is_consistent(child, depth + 1)


@pytest.mark.asyncio
async def test_reuse_survives_the_caller_mutating_the_state_in_place():
    """
    Die Aufrufer spielen mit state.make_move() auf demselben Objekt weiter, das
    beim vorherigen Aufruf die Wurzel war. Der Suchbaum darf davon nichts merken.
    """
    searcher = make_searcher(iteration_limit=200)
    state = ConnectFour()

    await searcher.search(state)
    state.make_move(3)
    await searcher.search(state)

    assert int(np.count_nonzero(searcher.root.state.board)) == 1
    _assert_tree_is_consistent(searcher.root)


@pytest.mark.asyncio
async def test_tree_stays_consistent_over_a_whole_game():
    searcher = make_searcher(iteration_limit=120)
    state = ConnectFour()

    for column in (3, 3, 4, 2, 4, 1):
        await searcher.search(state)
        _assert_tree_is_consistent(searcher.root)
        state.make_move(column)


@pytest.mark.asyncio
async def test_reused_root_still_returns_a_valid_distribution():
    searcher = make_searcher(iteration_limit=150)
    state = ConnectFour()

    for column in (3, 3, 2):
        await searcher.search(state)
        state.make_move(column)

    _, _, policy = await searcher.search(state)
    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0).all()


# ---------------------------------------------------------------------------
# Spielstaerke der reinen Suche (ohne trainiertes Netz)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_finds_an_immediate_win():
    """
    Mit gleichverteiltem Prior und Value 0 stammt jede Information aus dem Baum.
    Einen Zug vor dem Sieg muss die Suche den Gewinnzug finden.
    """
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        "..o....",
        "xxx.oo.",
    ], current_player=1)

    _, _, policy = await make_searcher(iteration_limit=400).search(state)

    assert int(np.argmax(policy)) == 3
    assert policy[3] > 0.5


@pytest.mark.asyncio
async def test_search_blocks_an_immediate_loss():
    state = board_from_rows([
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "ooo.x.x",
    ], current_player=1)

    _, _, policy = await make_searcher(iteration_limit=600).search(state)

    assert int(np.argmax(policy)) == 3


@pytest.mark.asyncio
async def test_search_returns_a_probability_distribution_over_legal_moves():
    state = ConnectFour()
    for _ in range(6):
        state.make_move(0)

    _, _, policy = await make_searcher(iteration_limit=100).search(state)

    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0).all()
    assert policy[0] == 0.0


@pytest.mark.asyncio
async def test_search_reports_the_network_value_of_the_root():
    evaluator = ScriptedEvaluator(value=0.42)
    nn_eval, _, _ = await make_searcher(evaluator, iteration_limit=20).search(ConnectFour())
    assert nn_eval == pytest.approx(0.42)
