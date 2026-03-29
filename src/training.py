import asyncio
import multiprocessing as mp
import os

import torch

from src.CFNet import load_model
from src.database.db_handler import DatabaseHandler
from generate_training_data import MoveDataset, process_entry_generate_dataset
from src.selfplay_parallel import selfplay_parallel, process_entry_selfplay
from src.utils import timing, get_filename, calculate_percentages
from update_model import update_model


@timing
def train_model(
    num_iterations=10,
    dataset_generation_time=1.0,
    mcts_iteration_limit=400,
    num_training_epochs=20,
    num_validation_games=200,
    generating_model_path=None
):
    """
    AlphaZero-Training mit optionalem Start von einem vortrainierten Modell.
    """

    db = DatabaseHandler()

    for iteration in range(num_iterations):
        model_name = get_filename(generating_model_path)
        generating_model = load_model(generating_model_path, model_name)

        print(f"\n=== ITERATION {iteration} | aktuelles Modell: {model_name} ===")

        mp.set_start_method("spawn", force=True)

        num_instances = 8
        processes = []

        for i in range(num_instances):
            p = mp.Process(
                target=process_entry_generate_dataset,
                args=(
                    dataset_generation_time,
                    mcts_iteration_limit,
                    generating_model_path,
                    f"{model_name}_instance_{i}",
                    16,
                    16
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


        buffer_size = 100000
        moves = db.load_moves_for_training(num_moves=buffer_size)

        print(f"→ {len(moves)} Moves geladen (Modell {model_name})")

        updated_model_path = update_model(generating_model_path,
                                     MoveDataset(moves),
                                     num_epochs=num_training_epochs)

        updated_model = load_model(updated_model_path, get_filename(updated_model_path))

        mp.set_start_method("spawn", force=True)

        num_instances = 8
        processes = []

        for i in range(num_instances):
            p = mp.Process(
                target=process_entry_selfplay,
                args=(
                    generating_model_path,
                    updated_model_path,
                    mcts_iteration_limit,
                    num_validation_games,
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


        selfplay_statistics = db.get_selfplay_statistics_from_database(challenger_model_tag=updated_model.tag)
        percentages = calculate_percentages(selfplay_statistics)

        print(f"The results are in. Challenger: {percentages['challenger']}, Champion: {percentages['champion']}, Draws: {percentages['draw']}")

        win_rate = percentages["challenger"]

        print(f"Vergleich Version {generating_model.tag} vs Version {updated_model.tag}: {win_rate*100:.1f}% Winrate")

        # --- Entscheidung ---
        if win_rate > 0.55:
            print(f"Neues Modell {updated_model.tag} akzeptiert!")

            # Dateiname inkl. Model ID und Iteration
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)

            # Gehe eine Ebene hoch zum Projekt-Root und dann in die Zielordner
            accepted_model_output_dir = os.path.join(project_root, "accepted_models")

            os.makedirs(accepted_model_output_dir, exist_ok=True)
            accepted_model_path = os.path.join(accepted_model_output_dir, updated_model.tag + '.pt')

            torch.save(updated_model.state_dict(), accepted_model_path)
            print(f"Model saved to {accepted_model_path}")

            generating_model_path = accepted_model_path

    return generating_model



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_output_dir = os.path.join(project_root, "accepted_models")

    generating_model_path = os.path.join(
        model_output_dir,
        "cfnet_20260215_224459.pt"
    )

    final_model = train_model(num_iterations=10,
                              dataset_generation_time=0.5,
                              mcts_iteration_limit=400,
                              num_training_epochs=20,
                              num_validation_games=50,
                              generating_model_path=generating_model_path
                             )