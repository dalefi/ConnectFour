import os
import numpy as np
import torch
from mcts.searcher.mcts_searcher import mcts_searcher
from src.ConnectFour import ConnectFour
from src.CFNet import CFNet
from src.database.db_handler import DatabaseHandler
from src.utils import timing

@timing
def selfplay(current_model=None,
             current_model_id=None,
             updated_model=None,
             updated_model_id=None,
             num_games=100,
             iteration_limit=250,
             device=device):
    print("Starting selfplay...")

    db = DatabaseHandler()
    statistics = {"wins_current_model": 0, "wins_updated_model": 0, "draws": 0}
    half_games = max(num_games // 2, 1)

    # --- Hälfte 1: current_model startet ---
    for _ in range(half_games):
        run_selfplay_game("current",
                        "updated",
                          current_model,
                          updated_model,
                          db,
                          statistics,
                          current_model_id,
                          updated_model_id,
                          iteration_limit,
                          device
                        )

    # --- Hälfte 2: updated_model startet ---
    for _ in range(half_games):
        run_selfplay_game("updated",
                        "current",
                          updated_model,
                          current_model,
                          db,
                          statistics,
                          current_model_id,
                          updated_model_id,
                          iteration_limit,
                          device)

    win_rate = statistics["wins_updated_model"] / num_games
    draw_rate = statistics["draws"] / num_games
    loss_rate = statistics["wins_current_model"] / num_games

    print(f"Final statistics: {statistics}")
    print("Finished selfplay!")
    print("Win rate:", win_rate, "Draw rate:", draw_rate, "Loss rate:", loss_rate)

    return win_rate, draw_rate, loss_rate


def run_selfplay_game(first_player_role,
                      second_player_role,
                      first_model,
                      second_model,
                      db,
                      statistics,
                      current_model_id,
                      updated_model_id,
                      iteration_limit,
                      device):

    running_state = ConnectFour()
    game_moves = []
    move_number = 0

    searchers = {
        first_player_role: mcts_searcher(iteration_limit=iteration_limit, neural_net=first_model, device=device),
        second_player_role: mcts_searcher(iteration_limit=iteration_limit, neural_net=second_model, device=device)
    }

    while True:
        # First player move
        running_state = make_selfplay_move(first_player_role, running_state, searchers[first_player_role],
                                           game_moves, move_number)
        move_number += 1
        if running_state.is_terminal():
            update_statistics_and_db(running_state, first_player_role, game_moves,
                                     statistics, db, current_model_id, updated_model_id)
            break

        # Second player move
        running_state = make_selfplay_move(second_player_role, running_state, searchers[second_player_role],
                                           game_moves, move_number)
        move_number += 1
        if running_state.is_terminal():
            update_statistics_and_db(running_state, second_player_role, game_moves,
                                     statistics, db, current_model_id, updated_model_id)
            break


def make_selfplay_move(player_role, running_state, mcts_searcher, game_moves, move_number):

    nn_eval, nn_policy, policy = mcts_searcher.search(initial_state=running_state, return_value_and_policy=True)

    # Dirichlet-Noise beim ersten Zug
    if move_number == 0:
        epsilon = 0.25
        alpha = 0.3
        noise = np.random.dirichlet([alpha] * len(policy))
        policy = (1 - epsilon) * policy + epsilon * noise

    # Sampling nach Policy
    next_move = np.random.choice(len(policy), p=policy)

    game_moves.append({
        "move_number": move_number,
        "board_state": running_state.board.tolist(),
        "policy": policy.tolist(),
        "nn_policy": nn_policy.tolist(),
        "value": nn_eval,
        "nn_eval": nn_eval,
        "current_player": running_state.get_current_player(),
        "model_role": player_role
    })

    running_state.make_move(next_move)
    return running_state



def update_statistics_and_db(
    running_state,
    last_player,
    game_moves,
    statistics,
    db,
    current_model_id,
    updated_model_id
):
    winner = int(running_state.get_winner())

    if winner == 0:
        statistics["draws"] += 1
    elif last_player == "current":
        statistics["wins_current_model"] += 1
    else:
        statistics["wins_updated_model"] += 1

    # --- Result aus Sicht des letzten gespeicherten Moves ---
    value = -running_state.get_reward()

    selfplay_moves = []

    for move in reversed(game_moves):
        selfplay_moves.append({
            "boardstate": move["board_state"],
            "mcts_policy": np.array(move["policy"]),
            "nn_policy": np.array(move["nn_policy"]),
            "current_player": int(move["current_player"]),
            "result": float(value),
            "value": value,
            "nn_eval": move["nn_eval"],
            "model_role":move["model_role"]
        })
        value *= -1

    selfplay_moves.reverse()

    db.insert_selfplay_game(
        winner=last_player,
        moves=selfplay_moves,
        current_model_id=current_model_id,
        updated_model_id=updated_model_id
    )



if __name__ == "__main__":
    model_output_dir = os.path.join("..", "models")
    model_files = [f for f in os.listdir(model_output_dir) if f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_output_dir}")

    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_output_dir, f)), reverse=True)
    latest_model_path = os.path.join(model_output_dir, model_files[0])

    updated_model = CFNet()
    updated_model.load_state_dict(torch.load(latest_model_path))
    updated_model.eval()
    updated_model.tag = "updated_model"  # wichtig für DB

    test_model = CFNet()
    test_model.tag = "current_model"

    db = DatabaseHandler()
    current_model_id=1
    updated_model_id=2
    db.ensure_model_exists(model_id=current_model_id)
    db.ensure_model_exists(model_id=updated_model_id)

    selfplay(current_model=test_model,
             current_model_id=current_model_id,
             updated_model=updated_model,
             updated_model_id=updated_model_id,
             num_games=200,
             iteration_limit=400)
