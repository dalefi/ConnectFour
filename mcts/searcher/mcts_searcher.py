from __future__ import division

import math
import random
from time import time
import numpy as np
import torch

from mcts.base.base import BaseState
from src.CFNet import state_to_tensor


class TreeNode:
    def __init__(self, state, parent, policy=None, value=0):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        # prior: unveraenderte (maskierte) Netzausgabe
        # policy: das, was die Suche benutzt - an der Wurzel ggf. mit Dirichlet-Rauschen
        self.prior = policy
        self.policy = policy
        self.value = value
        self.children = {}

    def __str__(self):
        s = ["totalReward: %s" % self.totalReward,
             "numVisits: %d" % self.numVisits,
             "isTerminal: %s" % self.is_terminal,
             "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))

    def get_ucb(self, action, child, exploration_value):
        """
        PUCT nach AlphaZero:  Q + c * P(a) * sqrt(N_parent) / (1 + N_child)

        Der Explorationsterm faellt umgekehrt proportional zu den Kindbesuchen ab.
        Nur dadurch konzentriert sich die Suche mit steigender Iterationszahl -
        eine Variante wie sqrt(N/(1+n)) behaelt einen konstanten Explorationsanteil
        und konvergiert nie.
        """
        if child.numVisits == 0:
            q_value = 0
        else:
            q_value = - (child.totalReward / child.numVisits)

        prior = self.policy[action.target_column]
        exploration = exploration_value * prior * math.sqrt(self.numVisits) / (1 + child.numVisits)

        return q_value + exploration


class mcts_searcher:
    # Unterhalb dieser Temperatur wird deterministisch der meistbesuchte Zug gewaehlt.
    MIN_TEMPERATURE = 0.05

    def __init__(self,
                 time_limit: int = None,
                 iteration_limit: int = None,
                 exploration_constant: float = 1.5,
                 device=None,
                 batcher=None,
                 dirichlet_alpha: float = 1.0,
                 noise_fraction: float = 0.25):

        self.root = None
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = time_limit
            self.limit_type = 'time'
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.search_limit = iteration_limit
            self.limit_type = 'iterations'

        self.exploration_constant = exploration_constant
        self.device = device
        self.batcher = batcher
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction

        # Der Batcher haelt das Netz; in Tests kann auch ein reiner Evaluator
        # ohne .model uebergeben werden.
        self.neural_net = getattr(batcher, "model", None)
        if self.neural_net is not None:
            self.neural_net.eval()

    @torch.no_grad()
    async def search(self, initial_state: BaseState = None, add_noise: bool = False,
                     temperature: float = 1.0):
        """
        add_noise:   Dirichlet-Rauschen auf den Wurzel-Prior. Beim Erzeugen von
                     Trainingsdaten immer an, beim Bewerten/Spielen aus.
        temperature: Schaerfe der zurueckgegebenen Besuchsverteilung.
        """
        if initial_state.is_terminal():
            raise ValueError("Auf einer beendeten Stellung kann nicht gesucht werden")

        self.root = self._reusable_node(initial_state)
        if self.root is None:
            # Kopie: der Aufrufer spielt auf seinem Objekt weiter (make_move mutiert
            # in place), der Baum muss aber seine eigene Stellung behalten.
            self.root = TreeNode(initial_state.copy(), parent=None)
        self.root.parent = None

        if self.root.prior is None:
            value, raw_policy = await self.batcher.get_policy_value(state_to_tensor(self.root.state))
            self.root.value = value
            self.root.prior = self.mask_invalid_actions(self.root, raw_policy)

        # Wurzel-Policy immer frisch aus dem sauberen Prior ableiten, damit sich
        # Rauschen ueber mehrere Suchen hinweg nicht aufsummiert.
        self.root.policy = self.root.prior.copy()
        if add_noise:
            self._apply_root_noise()

        if self.limit_type == 'time':
            time_limit = time() + self.timeLimit / 1000
            while time() < time_limit:
                await self.execute_round()
        else:
            for i in range(self.search_limit):
                await self.execute_round()

        nn_eval = self.root.value
        nn_policy = self.root.prior.copy()
        mcts_policy = self.get_policy_from_child_visits(temperature=temperature)

        return nn_eval, nn_policy, mcts_policy

    def _apply_root_noise(self):
        """
        Mischt Dirichlet-Rauschen unter den Wurzel-Prior. Ohne das kann die Suche
        nur Zuege vertiefen, die das Netz ohnehin schon mag - es gaebe also keine
        Quelle fuer echte Policy-Verbesserung.
        """
        legal_columns = [a.target_column for a in self.root.state.get_possible_actions()]
        if not legal_columns:
            return

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_columns))
        policy = self.root.prior.copy()
        for index, column in enumerate(legal_columns):
            policy[column] = ((1 - self.noise_fraction) * policy[column]
                              + self.noise_fraction * noise[index])

        self.root.policy = policy

    def _reusable_node(self, state):
        """
        Sucht die Stellung im bestehenden Baum (Wurzel, Kinder, Enkel) und gibt den
        passenden Knoten zurueck. So bleiben die Besuche des Teilbaums nach einem
        gespielten Zug erhalten, statt jedes Mal bei null anzufangen.
        """
        if self.root is None:
            return None

        def matches(node):
            return (node.state.get_current_player() == state.get_current_player()
                    and np.array_equal(node.state.board, state.board))

        if matches(self.root):
            return self.root

        for child in self.root.children.values():
            if matches(child):
                return child
            for grandchild in child.children.values():
                if matches(grandchild):
                    return grandchild

        return None

    async def execute_round(self):
        """
        execute a selection-expansion-simulation-backpropagation round
        """

        node = await self.select_node(self.root)
        self.backpropagate(node, node.value)

    async def select_node(self, node: TreeNode):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = await self.get_best_child(node, self.exploration_constant)
            else:
                await self.expand(node)
                return node

        return node

    @staticmethod
    async def expand(node: TreeNode) -> bool:

        possible_actions = node.state.get_possible_actions()

        for action in possible_actions:
            newNode = TreeNode(state=node.state.take_action(action), parent=node)
            node.children[action] = newNode

        if len(possible_actions) == len(node.children):
            node.is_fully_expanded = True

        return True

    @staticmethod
    def backpropagate(node: TreeNode, value: float):
        while node is not None:
            node.numVisits += 1
            node.totalReward += value
            node = node.parent
            value *= (-1) # need to flip the value for the parent node, because it belongs to the opponent

    async def get_best_child(self, node: TreeNode, exploration_value: float = None) -> TreeNode:

        best_value = float("-inf")
        best_nodes = []

        for action, child in node.children.items():
            node_value = node.get_ucb(action, child, exploration_value)
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)

        # choose among the best children and immediately set policy and value for this child
        chosen_child = random.choice(best_nodes)

        # mehrfachaufrufe minimieren
        if chosen_child.prior is None:
            if chosen_child.is_terminal:
                chosen_child.value = chosen_child.state.get_reward()
                chosen_child.prior = np.ones(7) / 7
            else:
                val, pol = await self.batcher.get_policy_value(state_to_tensor(chosen_child.state))
                chosen_child.value = val
                chosen_child.prior = self.mask_invalid_actions(chosen_child, pol)
            chosen_child.policy = chosen_child.prior

        return chosen_child

    def get_policy_from_child_visits(self, temperature=1.0):
        visits = np.zeros(7)
        for action, child in self.root.children.items():
            visits[action.target_column] = child.numVisits

        total_visits = visits.sum()
        if total_visits == 0:
            raise RuntimeError("Die Wurzel hat keine besuchten Kinder - Suche zu kurz?")

        if temperature < self.MIN_TEMPERATURE:
            greedy_policy = np.zeros(7)
            greedy_policy[int(visits.argmax())] = 1.0
            return greedy_policy

        scaled_visits = visits ** (1.0 / temperature)
        return scaled_visits / scaled_visits.sum()

    @staticmethod
    def mask_invalid_actions(node, policy):
        # create a mask that gets rid of impossible moves
        valid_moves_mask = np.zeros(7)
        for action in node.state.get_possible_actions():
            valid_moves_mask[action.target_column] = 1.0

        masked_policy = np.asarray(policy, dtype=float) * valid_moves_mask
        total = masked_policy.sum()

        if total <= 0:
            # Das Netz gibt allen legalen Zuegen Gewicht 0. Statt durch 0 zu teilen
            # (das ergibt still NaN) auf Gleichverteilung zurueckfallen.
            return valid_moves_mask / valid_moves_mask.sum()

        return masked_policy / total
