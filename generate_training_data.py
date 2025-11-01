from tqdm.notebook import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy

import ConnectFour
import CFNet
from mcts_custom.searcher.mcts_custom import MCTS_custom


def generate_dataset(target_dataset_size=1000, iteration_limit=250, model=None):
    """
    Generates a dataset of games by playing against itself.
    """
    print("GENERATING DATASET ...")

    games = {}
    game_nr = 0
    current_dataset_size = 0

    pbar = tqdm(total=target_dataset_size, desc="Collecting moves")

    while current_dataset_size < target_dataset_size:
        game = {}
        current_state = ConnectFour.ConnectFour()
        searcher = MCTS_custom(iteration_limit=iteration_limit, neural_net=model)
        move = 0

        while not current_state.is_terminal():
            mcts_policy = searcher.search(initial_state=current_state, return_value_and_policy=True)
            game[f"move_{move}"] = {
                "boardstate": deepcopy(current_state),
                "mcts_policy": mcts_policy
            }
            next_move = np.random.choice(range(7), p=mcts_policy)
            current_state.make_move(next_move)
            move += 1

        result = -current_state.get_reward()
        for i in range(len(game) - 1, -1, -1):
            game[f"move_{i}"]["result"] = result
            result *= (-1)

        games[f"game{game_nr}"] = game
        current_dataset_size += len(game)
        pbar.update(len(game))
        game_nr += 1

    pbar.close()

    all_moves = [
        move_data
        for game in games.values()
        for move_data in game.values()
    ]

    print("DATASET CREATED!")
    return MoveDataset(all_moves)


class MoveDataset(Dataset):
    def __init__(self, moves):
        self.moves = moves

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        move = self.moves[idx]
        boardstate = CFNet.state_to_tensor(move["boardstate"])
        policy = torch.tensor(move["mcts_policy"], dtype=torch.float32)
        result = torch.tensor(move["result"], dtype=torch.float32)
        return boardstate, policy, result
