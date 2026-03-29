from __future__ import division

import math
import random
from time import time
import numpy as np
import torch

from mcts.base.base import BaseState
from src.CFNet import state_to_tensor
from src.utils import timing


class TreeNode:
    def __init__(self, state, parent, policy=None, value=0):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.policy = policy # np.array
        self.value = value # float
        self.children = {}

    def __str__(self):
        s = ["totalReward: %s" % self.totalReward,
             "numVisits: %d" % self.numVisits,
             "isTerminal: %s" % self.is_terminal,
             "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


    def get_ucb(self, action, child, exploration_value):
        """
        Calculates ucb value for given child at state 'self'
        """

        if child.numVisits == 0:
            q_value = 0
        else:
            q_value = - (child.totalReward / child.numVisits)
        return q_value + exploration_value * math.sqrt(self.numVisits / (child.numVisits + 1)) * self.policy[action.target_column]



class mcts_searcher:
    def __init__(self,
                 time_limit: int = None,
                 iteration_limit: int = None,
                 exploration_constant: float = 2,
                 device=None,
                 batcher=None):

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
        self.neural_net = batcher.model
        self.device = device
        self.batcher = batcher
        
        # put the net into eval mode
        self.neural_net.eval()


    @torch.no_grad()
    async def search(self, initial_state: BaseState = None):

        # create the root node at the starting position from where the algorithm is called
        self.root = TreeNode(initial_state, parent=None)

        # immediately set policy and value for root node
        val, pol = await self.batcher.get_policy_value(state_to_tensor(self.root.state))
        self.root.value, self.root.policy = val, self.mask_invalid_actions(self.root, pol)

        # determine how long the algo is supposed to run
        if self.limit_type == 'time':
            time_limit = time() + self.timeLimit / 1000
            while time() < time_limit:
                await self.execute_round()
        else:
            for i in range(self.search_limit):
                await self.execute_round()


        nn_eval = self.root.value
        nn_policy = self.root.policy
        mcts_policy = self.get_policy_from_child_visits(temperature=1)

        return nn_eval, nn_policy, mcts_policy


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
        if chosen_child.policy is None:
            if chosen_child.is_terminal:
                chosen_child.value = chosen_child.state.get_reward()
                chosen_child.policy = np.ones(7) / 7
            else:
                val, pol = await self.batcher.get_policy_value(state_to_tensor(chosen_child.state))
                chosen_child.value = val
                chosen_child.policy = self.mask_invalid_actions(chosen_child, pol)

        return chosen_child

    def get_policy_from_child_visits(self, temperature = 1):
        mcts_policy = np.zeros(7)
        for action, child in self.root.children.items():
            mcts_policy[action.target_column] = child.numVisits ** (1/temperature)

        mcts_policy = mcts_policy/mcts_policy.sum()

        return mcts_policy

    @staticmethod
    def mask_invalid_actions(node, policy):
        # create a mask that gets rid of impossible moves
        valid_actions = node.state.get_possible_actions()
        valid_actions = np.array([action.target_column for action in valid_actions])
        valid_moves_mask = np.zeros(7)
        for idx in valid_actions:
            valid_moves_mask[idx] = 1
        masked_policy = policy*valid_moves_mask
        try:
            masked_policy /= np.sum(masked_policy)
        except:
            print(f"DIVISION BY ZERO! policy = {policy}, valid_moves_mask = {valid_moves_mask}, masked_policy = {masked_policy}, boardstate = {node.state}")

        return masked_policy

