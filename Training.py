"""
Script to train a model for ConnectFour. The training data, i.e. the played games need to be generated first.
"""

import numpy as np
import torch

def Training(model=None, games=None, training_iterations=None):
    """
    games is a nested list of games as returned by Selfplay.
    """

    model.train()
    optimizer = torch.optim.SGD(
    model.parameters(), lr = 0.02, momentum=0.9, weight_decay=1e-5
    )

    # unpack training data
    train_data = []
    mcts_policies = []
    labels = []
    
    for i in range(len(games)):
        for j in range(3):
            train_data.append(CFNet.state_to_tensor(games[i][j][0]))
            mcts_policies.append(TestGame[i][j][1])
            labels.append(TestGame[i][j][2])
    
    training_iterations = len(train_data) if training_iterations is None

    # sample game
    train_idx = np.random.randint(0,len(train_data))
    value, policy = model(train_data[train_idx])