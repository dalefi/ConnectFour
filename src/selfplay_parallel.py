import asyncio
import os
import numpy as np
from tqdm.asyncio import tqdm

from mcts.searcher.mcts_searcher import mcts_searcher
from src.ConnectFour import ConnectFour
from src.CFNet import load_model
from src.NeuralNetBatcher import NeuralNetBatcher
from src.database.db_handler import DatabaseHandler
from src.utils import get_filename


def update_statistics_and_db(
    running_state,
    last_player,
    game_moves,
    statistics,
    db,
    current_model_tag,
    updated_model_tag
):
    winner = int(running_state.get_winner())

    if winner == 0:
        statistics["draws"] += 1
        game_winner = "draw"
    elif last_player == "current":
        statistics["wins_current_model"] += 1
        game_winner = "champion"
    else:
        statistics["wins_updated_model"] += 1
        game_winner = "challenger"

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
        winner=game_winner,
        moves=selfplay_moves,
        model_1_tag=current_model_tag,
        model_2_tag=updated_model_tag
    )


async def play_one_game_async(
        game_id,
        first_player_role,
        second_player_role,
        batcher_current,
        batcher_updated,
        iteration_limit,
        db,
        statistics,
        stats_lock,
        device,
        pbar,
        active_games_status
):
    running_state = ConnectFour()
    game_moves = []
    move_number = 0

    # Zuordnung: Welches Modell nutzt welchen Batcher?
    batchers = {
        "current": batcher_current,
        "updated": batcher_updated
    }

    # Rollenverteilung für dieses spezifische Spiel
    player_roles = {1: first_player_role, -1: second_player_role}

    while not running_state.is_terminal():
        current_role = player_roles[running_state.get_current_player()]
        current_batcher = batchers[current_role]

        # STATUS UPDATE für die Anzeige
        active_games_status[game_id] = f"G{game_id:02d}:{current_role[0].upper()}{move_number:02d}"
        # Wir zeigen z.B. die letzten 16 aktiven Spiele in der Description an
        status_str = " | ".join(list(active_games_status.values())[-16:])
        pbar.set_description(f"Selfplay [{status_str}]")
        pbar.refresh()

        # MCTS Searcher für diesen Zug erstellen
        searcher = mcts_searcher(
            iteration_limit=iteration_limit,
            batcher=current_batcher,
            device=device
        )

        # Suche asynchron ausführen
        nn_eval, nn_policy, mcts_policy = await searcher.search(initial_state=running_state)

        # Dirichlet-Noise beim ersten Zug
        if move_number == 0:
            epsilon, alpha = 0.25, 0.3
            noise = np.random.dirichlet([alpha] * len(mcts_policy))
            mcts_policy = (1 - epsilon) * mcts_policy + epsilon * noise

        # Sampling nach Policy
        next_move = np.random.choice(len(mcts_policy), p=mcts_policy)

        game_moves.append({
            "move_number": move_number,
            "board_state": running_state.board.tolist(),
            "policy": mcts_policy.tolist(),
            "nn_policy": nn_policy.tolist(),
            "nn_eval": nn_eval,
            "current_player": running_state.get_current_player(),
            "model_role": current_role
        })

        running_state.make_move(next_move)
        move_number += 1

    async with stats_lock:
        update_statistics_and_db(running_state,
                                 player_roles[running_state.get_current_player() * -1],
                                 game_moves,
                                 statistics,
                                 db,
                                 batcher_current.model.tag,
                                 batcher_updated.model.tag
                                )

        # Postfix im Lock updaten, damit die Zahlen immer stimmen
        pbar.set_postfix({
            "U-Win": statistics["wins_updated_model"],
            "C-Win": statistics["wins_current_model"],
            "Draw": statistics["draws"]
        })

    pbar.update(1)


async def selfplay_parallel(
        current_model,
        updated_model,
        num_games=10,
        iteration_limit=400,
        device="cuda"
):

    db = DatabaseHandler()
    stats_lock = asyncio.Lock()
    statistics = {"wins_current_model": 0, "wins_updated_model": 0, "draws": 0}
    active_games_status = {}

    # Zwei Batcher initialisieren, damit beide Modelle parallel auf die GPU können
    batcher_current = NeuralNetBatcher(current_model, device, batch_size=16)
    batcher_updated = NeuralNetBatcher(updated_model, device, batch_size=16)

    semaphore = asyncio.Semaphore(16)
    pbar = tqdm(total=num_games, desc="Selfplay Tournament")

    async def sem_task(game_id, f_role, s_role):
        async with semaphore:
            await play_one_game_async(game_id,
                                      f_role,
                                      s_role,
                                      batcher_current,
                                      batcher_updated,
                                      iteration_limit,
                                      db,
                                      statistics,
                                      stats_lock,
                                      device,
                                      pbar,
                                      active_games_status)

    tasks = []
    half_games = num_games // 2

    for i in range(num_games):
        if i < half_games:
            tasks.append(sem_task(i, "current", "updated"))
        else:
            tasks.append(sem_task(i, "updated", "current"))

    # Alle Spiele parallel ausführen
    await asyncio.gather(*tasks)
    pbar.close()

    win_rate = statistics["wins_updated_model"] / num_games
    print(f"Final statistics: {statistics}")
    return win_rate


def process_entry_selfplay(champion_model_path,
                           challenger_model_path,
                           iteration_limit,
                           num_validation_games):

    champion_model_name = get_filename(champion_model_path)
    challenger_model_name = get_filename(challenger_model_path)

    champion_model = load_model(
        model_path=champion_model_path,
        model_tag=champion_model_name
    )

    challenger_model = load_model(
        challenger_model_path,
        challenger_model_name
    )


    asyncio.run(selfplay_parallel(
        champion_model,
        challenger_model,
        num_games=num_validation_games,
        iteration_limit=iteration_limit
    ))


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_output_dir = os.path.join(project_root, "models")
    accepted_model_output_dir = os.path.join(project_root, "accepted_models")

    v0model = load_model(model_path=None, model_tag="untrained_model")

    champion_model = "cfnet_20260207_130926.pt"
    challenger_model = "cfnet_20260212_140300.pt"

    champion_model_path = os.path.join(accepted_model_output_dir, champion_model)
    champion_model = load_model(model_path=champion_model_path, model_tag=champion_model)

    challenger_model_path = os.path.join(model_output_dir, challenger_model)
    challenger_model = load_model(model_path=challenger_model_path, model_tag=challenger_model)


    asyncio.run(selfplay_parallel(
        champion_model,
        challenger_model,
        num_games=100,
        iteration_limit=400
    ))