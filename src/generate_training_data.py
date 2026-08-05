import time
import traceback
import multiprocessing as mp
import asyncio
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.asyncio import tqdm

from src import ConnectFour
from src.CFNet import CFNet, encode_boards, load_model
from src.NeuralNetBatcher import NeuralNetBatcher
from mcts.searcher.mcts_searcher import mcts_searcher
from src.database.db_handler import DatabaseHandler
from src.selfplay_parallel import update_statistics_and_db
from src.utils import temperature_for_move


def boardstate_to_list(boardstate):
    arr = boardstate.board
    return arr.tolist()


class MoveDataset(Dataset):
    """
    augment: spiegelt jede Stellung mit 50% Wahrscheinlichkeit links-rechts.
             Connect Four ist symmetrisch, das verdoppelt die effektive Datenmenge.
    """

    def __init__(self, moves, augment=True):
        self.augment = augment
        # Nur fuer Tests: erzwingt das Spiegeln statt es auszuwuerfeln.
        self._force_mirror = False

        # Den ganzen Buffer einmal vektorisiert umwandeln statt pro Zugriff.
        boards = np.array([move.board_state for move in moves], dtype=np.int8).reshape(-1, 6, 7)
        players = np.array([move.current_player for move in moves], dtype=np.int8)

        self.inputs = encode_boards(boards, players)
        self.policies = torch.from_numpy(
            np.array([move.policy for move in moves], dtype=np.float32).reshape(-1, 7)
        )
        self.values = torch.from_numpy(
            np.array([move.value for move in moves], dtype=np.float32)
        )

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        policy = self.policies[idx]

        if self.augment and (self._force_mirror or np.random.random() < 0.5):
            # Spiegeln an der Spaltenachse: Brett und Policy muessen zusammen kippen.
            input_tensor = torch.flip(input_tensor, dims=[2])
            policy = torch.flip(policy, dims=[0])

        return input_tensor, policy, self.values[idx]

async def generate_dataset(
        duration_hours=1.0,
        iteration_limit=400,
        model=None,
        num_parallel_games=16,
        batch_size=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Starting generate_dataset on {device}")

    active_games_status = {}

    duration_seconds = duration_hours * 3600
    start_time = time.time()
    end_time = start_time + duration_seconds

    batcher = NeuralNetBatcher(model=model, device=device, batch_size=batch_size)
    db = DatabaseHandler()

    stats = {"total_moves": 0, "game_nr": 0}
    stop_event = asyncio.Event()

    master_pbar = tqdm(
        total=duration_seconds,
        desc="Dataset",
        unit="s",
        position=0,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining} {postfix}"
    )

    async def timer_manager():
        while time.time() < end_time and not stop_event.is_set():
            elapsed = time.time() - start_time
            master_pbar.n = int(elapsed)

            # Statistiken im Master-Balken anzeigen
            mps = stats["total_moves"] / max(1, int(elapsed))

            master_pbar.set_postfix({
                "Moves": stats["total_moves"],
                "MPS": f"{mps:.2f}",
            })

            status_str = " | ".join(list(active_games_status.values())[-16:])

            master_pbar.set_description(
                f"Dataset [{status_str}]"
            )

            master_pbar.refresh()
            await asyncio.sleep(1)

        stop_event.set()

    async def worker(worker_id):

        active_games_status[worker_id] = f"W{worker_id:02d}:START"

        while not stop_event.is_set():
            try:

                game_data, winner = await play_one_game(
                    batcher,
                    iteration_limit,
                    worker_id,
                    stats,
                    active_games_status,
                )

                stats["game_nr"] += 1

                # Speichern in DB
                db.insert_training_game(
                    winner=winner,
                    moves=game_data,
                    model_tag=model.tag
                )

                tqdm.write(f" [✓] Worker-{worker_id:02d} finished game ({len(game_data)} moves)")

            except Exception as e:
                tqdm.write(f"Error in Worker-{worker_id}: {e}")
                print(f"\nCRITICAL ERROR in Worker-{worker_id}:")
                print(traceback.format_exc())
                stop_event.set()
                break


    timer_task = asyncio.create_task(timer_manager())

    # Erstelle die Worker-Tasks
    print(f"DEBUG: Launching {num_parallel_games} workers...")
    workers = [worker(i) for i in range(num_parallel_games)]

    # Nutze wait statt gather für bessere Kontrolle bei Fehlern
    await asyncio.gather(*workers)
    master_pbar.close()
    print("DEBUG: All workers finished.")


async def play_one_game(batcher,
                        iteration_limit,
                        worker_id,
                        stats,
                        active_games_status):

    current_state = ConnectFour.ConnectFour.random_start_state(max_random_moves=6)
    # Ein Searcher pro Partie: so bleibt der Teilbaum ueber die Zuege hinweg erhalten.
    searcher = mcts_searcher(iteration_limit=iteration_limit, batcher=batcher)
    game_history = []

    # Sicherstellen, dass move_number existiert
    move_count = 0

    while not current_state.is_terminal():
        if active_games_status:
            active_games_status[worker_id] = f"W{worker_id:02d}:M{move_count:02d}"

        # Dirichlet-Rauschen an der Wurzel ist beim Erzeugen von Trainingsdaten
        # zwingend - sonst kann die Suche den Prior des Netzes nie verlassen.
        nn_eval, nn_policy, mcts_policy = await searcher.search(
            current_state,
            add_noise=True,
            temperature=1.0
        )

        game_history.append({
            "boardstate": boardstate_to_list(current_state),
            "mcts_policy": mcts_policy.tolist(),
            "nn_policy": nn_policy.tolist(),
            "nn_eval": float(nn_eval),
            "current_player": int(current_state.get_current_player())
        })

        # Trainingsziel bleibt die Besuchsverteilung bei Temperatur 1 (oben);
        # gezogen wird nur in der Eroeffnung zufaellig, danach greedy.
        move_policy = searcher.get_policy_from_child_visits(
            temperature=temperature_for_move(move_count)
        )
        next_move = int(np.random.choice(7, p=move_policy))
        current_state.make_move(next_move)
        move_count += 1

        stats["total_moves"] += 1

    print(f"[Worker-{worker_id}] Finished game after {move_count} moves.")
    active_games_status.pop(worker_id, None)

    result = -current_state.get_reward()
    winner = int(current_state.get_winner())

    # Backfill der Resultate
    current_res = result
    for move in reversed(game_history):
        move["result"] = current_res
        current_res *= -1

    return game_history, winner


def process_entry_generate_dataset(duration_hours,
                                   iteration_limit,
                                   model_path,
                                   model_tag,
                                   num_parallel_games,
                                   batch_size):

    generating_model = load_model(
        model_path=model_path,
        model_tag=model_tag
    )

    asyncio.run(generate_dataset(
        duration_hours=duration_hours,
        iteration_limit=iteration_limit,
        model=generating_model,
        num_parallel_games=num_parallel_games,
        batch_size=batch_size
    ))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_output_dir = os.path.join(project_root, "accepted_models")

    generating_model_path = os.path.join(
        model_output_dir,
        "cfnet_20260207_130926.pt"
    )

    num_instances = 4
    processes = []

    for i in range(num_instances):
        p = mp.Process(
            target=process_entry_generate_dataset,
            args=(
                12.0,
                400,
                generating_model_path,
                f"cfnet_20260207_130926_instance_{i}",
                16,
                16
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
