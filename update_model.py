from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
import random

def update_model(input_model=None, trainData=None, num_epochs=100):
    """
    Given a model and training data this function updates the model for a given number of epochs.
    """

    print("UPDATING MODEL ...")
    model = deepcopy(input_model)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,  # Try 1e-3 or 5e-4
        weight_decay=1e-4  # Helps regularization
    )
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        dataloader = DataLoader(trainData, batch_size=32, shuffle=True)
        for batch in dataloader:
            # forward pass through model
            nn_value = model(batch[0])['value']
            nn_policy = model(batch[0])['policy']

            # calculate loss
            loss = model.alphaloss(nn_value, nn_policy, batch[2], batch[1])

            # backpropagation
            loss.backward()
            optimizer.step()

    print("MODEL UPDATED!")
    return model