from __future__ import division

import math
import random
import time
import numpy as np
import torch

from mcts.base.base import BaseState
from CFNet import CFNet


def random_policy(state: BaseState) -> float:
    while not state.is_terminal():
        try:
            action = np.random.choice(state.get_possible_actions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.take_action(action)
    return state.get_reward()

def neural_network_policy(state: BaseState, neural_net = None) -> float:
    """
    Instead of randomly choosing the next action, we rather take the output of a neural network
    as a probability distribution and sample from that distribution.
    """

    """
    # get a prediction of the next best move given a state
    move_probabilities = neural_net(state)['policy'].detach().numpy()[0]
    try:
        action = np.random.choice(state.get_possible_actions(), p=move_probabilities)
    except IndexError:
        raise Exception("Non-terminal state has no possible actions: " + str(state))
    state = state.take_action(action)
    """
    # Ich glaube ich brauche nur die Evaluierung des Modells für die aktuelle Position
    
    # now we calculate the value that the network gives us for the new position and return it
    evaluation = neural_net(state)['value'].item()
    
    return evaluation


class TreeNode:
    def __init__(self, state, parent, prior=0, policy=None, value=0):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.prior = prior # prior as observed from parent node
        self.policy = policy
        self.value = value
        self.children = {}

    def __str__(self):
        s = ["totalReward: %s" % self.totalReward,
             "numVisits: %d" % self.numVisits,
             "isTerminal: %s" % self.is_terminal,
             "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


    def get_ucb(self, child, exploration_value):
        """
        Calculates ucb value.
        """

        if child.numVisits == 0:
            q_value = 0
        else:
            #should be between 0 and 1 and we flip it to choose bad position for opponent
            q_value = -(child.totalReward / child.numVisits)
        return q_value + exploration_value * math.sqrt(self.numVisits / (child.numVisits + 1)) * child.prior
        


class MCTS_custom:
    def __init__(self,
                 time_limit: int = None,
                 iteration_limit: int = None,
                 exploration_constant: float = 2,
                 rollout_policy=None,
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
        self.rollout_policy = rollout_policy
        self.neural_net = neural_net
        
        # put the net into eval mode
        self.neural_net.eval()

    @torch.no_grad()
    def search(self, initial_state: BaseState = None, need_details: bool = None, return_probabilities: bool = None):
        self.root = TreeNode(initial_state, parent=None, prior=1)

        if self.limit_type == 'time':
            time_limit = time.time() + self.timeLimit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()

        best_child = self.get_best_child(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is best_child).__next__()


        # print all children of root
        print("Anzahl Kinder der root", len(self.root.children))
        for action, child in self.root.children.items():
            print("action", action, "leads to ")
            print(child)
        if best_child.numVisits == 0:
            print(self.root)
            print(best_child)

        if return_probabilities:
            child_values = self.get_policy_from_child_visits(temperature=1)
            return action, best_child.totalReward / best_child.numVisits, child_values
        if need_details:
            return action, best_child.totalReward / best_child.numVisits
        else:
            return action

    def execute_round(self):
        """
        execute a selection-expansion-simulation-backpropagation round
        """
        node, value = self.select_node(self.root)
        #reward = self.neural_net(node.state)['value'].item()
        self.backpropogate(node, value)

    
    def select_node(self, node: TreeNode) -> TreeNode:
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                # calculate the network's prior
                value, policy = self.neural_net(node.state).items() # ['policy'].detach().numpy()[0]
                value = value[1].item()
                policy = policy[1].detach().numpy()[0]
                
                policy = self.mask_invalid_actions(node, policy)

                self.expand(node, policy)
                return node, value
                
        value, _ = self.neural_net(node.state).items()
        value = value[1].item()
        return node, value

    
    def expand(self, node: TreeNode, policy) -> TreeNode:
        actions = node.state.get_possible_actions()
        if len(actions) != len(policy[policy!=0]):
            print(actions)
            print(policy)
        
        assert len(actions) == len(policy[policy!=0])
        
        for action, prob in zip(actions, policy):
            newNode = TreeNode(state=node.state.take_action(action), parent=node, prior=prob)
            node.children[action] = newNode
            if len(actions) == len(node.children):
                node.is_fully_expanded = True
        return True


    def backpropogate(self, node: TreeNode, value: float):
        while node is not None:
            node.numVisits += 1
            node.totalReward += value
            node = node.parent
            value *= (-1) # need to flip the value for the parent node, because it belongs to the opponent


    def get_best_child(self, node: TreeNode, exploration_value: float = None) -> TreeNode:
        """
        Incorporating a prior from a neural network.
        """
        best_value = float("-inf")
        best_nodes = []
        
        for action, child in node.children.items():
            node_value = node.get_ucb(child, exploration_value)
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        return np.random.choice(best_nodes)

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










