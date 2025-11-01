from __future__ import division

import math
import random
import time
import numpy as np
import torch

from mcts.base.base import BaseState
from CFNet import CFNet

class TreeNode:
    def __init__(self, state, parent, policy=None, value=0):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.policy = policy # a dict with actions (target columns) as keys and their respective probabilities as predicted by the network
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
        Calculates ucb value for given child at state 'self'
        """

        if child.numVisits == 0:
            q_value = 0
        else:
            q_value = - (child.totalReward / child.numVisits)
        return q_value + exploration_value * math.sqrt(self.numVisits / (child.numVisits + 1)) * self.policy[action.target_column]



class MCTS_custom:
    def __init__(self,
                 time_limit: int = None,
                 iteration_limit: int = None,
                 exploration_constant: float = 2,
                 neural_net=None):

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
        self.neural_net = neural_net
        
        # put the net into eval mode
        self.neural_net.eval()

    @torch.no_grad()
    def search(self, initial_state: BaseState = None, need_details: bool = None, return_value_and_policy: bool = None):

        # create the root node at the starting position from where the algorithm is called
        self.root = TreeNode(initial_state, parent=None)

        # immediately set policy and value for root node
        self.set_policy_and_value(self.root)

        # determine how long the algo is supposed to run
        if self.limit_type == 'time':
            time_limit = time.time() + self.timeLimit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()

        # choose the best child and the best action to return them
        best_child = self.get_best_child(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is best_child).__next__()

        # return the policy
        if return_value_and_policy:
            mcts_policy = self.get_policy_from_child_visits(temperature=1)
            return mcts_policy
        if need_details:
            return action, best_child.totalReward / best_child.numVisits
        else:
            return action

    def execute_round(self):
        """
        execute a selection-expansion-simulation-backpropagation round
        """

        node = self.select_node(self.root)
        self.backpropagate(node, node.value)

    
    def select_node(self, node: TreeNode):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                self.expand(node)
                return node

        return node

    def set_policy_and_value(self, node: TreeNode):
        if node.is_terminal:
            node.value = node.state.get_reward()
            node.policy = None
        else:
            value, policy = self.neural_net(node.state).items()
            value = value[1].item()
            policy = policy[1].detach().numpy()[0]
            policy = self.mask_invalid_actions(node, policy)

            # set value and policy of node
            node.policy = policy
            node.value = value

        return True

    @staticmethod
    def expand(node: TreeNode) -> bool:

        possible_actions = node.state.get_possible_actions()



        try:
            assert len(possible_actions) == len(node.policy[node.policy!=0])
        except AssertionError:
            node.state.display_board()
            print(f"possible_actions: {possible_actions}")
            print(f"node.policy: {node.policy}")

        for action in possible_actions:
            newNode = TreeNode(state=node.state.take_action(action), parent=node)
            node.children[action] = newNode

        # das hier sollte immer getriggert werden, da ich sofort alle kinder erzeuge
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


    def get_best_child(self, node: TreeNode, exploration_value: float = None) -> TreeNode:

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
        self.set_policy_and_value(chosen_child)

        return chosen_child


    def get_policy_from_child_visits(self, temperature = 1):
        mcts_policy = np.empty(7)
        for action, child in self.root.children.items():
            mcts_policy[action.target_column] = child.numVisits ** (1/temperature)

        mcts_policy = mcts_policy/mcts_policy.sum()

        return mcts_policy


    def mask_invalid_actions(self, node, policy):
        # create a mask that gets rid of impossible moves
        valid_actions = node.state.get_possible_actions()
        valid_actions = np.array([action.target_column for action in valid_actions])
        valid_moves_mask = np.zeros(7)
        for idx in valid_actions:
            valid_moves_mask[idx] = 1
        masked_policy = policy*valid_moves_mask
        masked_policy /= np.sum(masked_policy)

        return masked_policy










