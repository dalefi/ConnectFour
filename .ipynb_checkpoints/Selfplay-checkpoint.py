"""
A script for selfplay
"""

import numpy as np
from copy import deepcopy

import ConnectFour
import CFNet
from mcts_custom.searcher.mcts_custom import MCTS_custom


def Selfplay(num_iterations = 1000):
    """
    Need 2 agents that play a game.
    Need to record the game and its outcome as well as the output policies from MCTS.
    """

    games = []

    for _ in range(num_iterations):
    
        # this can be improved
        game = []
        
        current_state = ConnectFour.ConnectFour()
        
        board = np.zeros((6,7))
        board[:,0] = np.array([-1,1,-1,1,-1,1])
        current_state.board = board
        
        TrainNet = CFNet.CFNet()
        searcher = MCTS_custom(iteration_limit=100, neural_net=TrainNet)
    
        while not current_state.is_terminal():
            current_state.display_board()
            _, _, mcts_policy = searcher.search(initial_state=current_state, return_probabilities=True)
    
            # save state and policy
            game.append([deepcopy(current_state), mcts_policy])
            #sample next move
            next_move = np.random.choice(range(7), p=mcts_policy)
            current_state.make_move(next_move)
    
        # once the game is finished return the scores to the game
        result = (-1)*current_state.get_reward()
    
        #print(f"Result: {result}")
    
        for i in range(len(game)-1, -1, -1):
            game[i].append(result)
            result *= (-1)
    
        games.append(game)

    return games