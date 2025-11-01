import numpy as np
from tqdm.notebook import tqdm

from MCTS_custom.mcts_custom.searcher.mcts_custom import MCTS_custom
from ConnectFour import ConnectFour

def update_winner_score(result, wins_current_model, wins_updated_model, draws):
    if result == 1:
        wins_current_model += 1
    elif result == -1:
        wins_updated_model += 1
    else:
        draws += 1
    return wins_current_model, wins_updated_model, draws

def selfplay(current_model=None, updated_model=None, num_games=100, iteration_limit=250):
    """
    This script lets two models play against each other.
    It returns the win rate of the updated model. If it is above 50% the updated model replaces the current model.
    """

    print("Starting selfplay...")

    wins_current_model = 0
    wins_updated_model = 0
    draws = 0

    searcher_current_model = MCTS_custom(iteration_limit=iteration_limit, neural_net=current_model)
    searcher_updated_model = MCTS_custom(iteration_limit=iteration_limit, neural_net=updated_model)

    half_games = max(num_games // 2, 1)

    print("Playing first half (current model starts)...")
    for _ in tqdm(range(half_games), desc="First half"):
        current_state = ConnectFour()

        while True:
            current_model_mcts_policy = searcher_current_model.search(initial_state=current_state, return_value_and_policy=True)
            next_move = np.random.choice(range(7), p=current_model_mcts_policy)
            current_state.make_move(next_move)

            if current_state.is_terminal():
                winner = current_state.get_winner()
                wins_current_model, wins_updated_model, draws = update_winner_score(winner, wins_current_model, wins_updated_model, draws)
                break

            updated_model_mcts_policy = searcher_updated_model.search(initial_state=current_state, return_value_and_policy=True)
            next_move = np.random.choice(range(7), p=updated_model_mcts_policy)
            current_state.make_move(next_move)

            if current_state.is_terminal():
                winner = current_state.get_winner()
                wins_current_model, wins_updated_model, draws = update_winner_score(winner, wins_current_model, wins_updated_model, draws)
                break

    print("Playing second half (updated model starts)...")
    for _ in tqdm(range(half_games), desc="Second half"):
        current_state = ConnectFour()

        while True:
            updated_model_mcts_policy = searcher_updated_model.search(initial_state=current_state, return_value_and_policy=True)
            next_move = np.random.choice(range(7), p=updated_model_mcts_policy)
            current_state.make_move(next_move)

            if current_state.is_terminal():
                winner = current_state.get_winner()
                wins_current_model, wins_updated_model, draws = update_winner_score(winner, wins_current_model, wins_updated_model, draws)
                break

            current_model_mcts_policy = searcher_current_model.search(initial_state=current_state, return_value_and_policy=True)
            next_move = np.random.choice(range(7), p=current_model_mcts_policy)
            current_state.make_move(next_move)

            if current_state.is_terminal():
                winner = current_state.get_winner()
                wins_current_model, wins_updated_model, draws = update_winner_score(winner, wins_current_model, wins_updated_model, draws)
                break

    win_rate = wins_updated_model / num_games
    draw_rate = draws / num_games
    loss_rate = 1 - win_rate - draw_rate

    print("Finished selfplay!")
    print("Win rate:", win_rate, "Draw rate:", draw_rate, "Loss rate:", loss_rate)

    return win_rate, draw_rate, loss_rate
