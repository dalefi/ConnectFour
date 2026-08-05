import asyncio
import multiprocessing as mp
import os

import torch

from src.benchmark_vs_minimax import run_benchmark
from src.CFNet import load_model
from src.database.db_handler import DatabaseHandler
from src.generate_training_data import MoveDataset, process_entry_generate_dataset
from src.selfplay_parallel import selfplay_parallel, process_entry_selfplay
from src.utils import timing, get_filename, gating_win_rate, enable_utf8_console
from src.update_model import update_model


def run_in_processes(target, args_per_process):
    """Startet je einen Prozess pro Argument-Tupel und wartet auf alle."""
    mp.set_start_method("spawn", force=True)

    processes = []
    for args in args_per_process:
        process = mp.Process(target=target, args=args)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


@timing
def train_model(
    num_iterations=10,
    dataset_generation_time=1.0,
    mcts_iteration_limit=400,
    num_training_epochs=4,
    num_validation_games=200,
    generating_model_path=None,
    num_worker_processes=8,
    games_per_worker=32,
    nn_batch_size=32,
    anchor_depths=(2, 4),
    anchor_pairs=10,
):
    """
    AlphaZero-Training mit optionalem Start von einem vortrainierten Modell.

    num_worker_processes / games_per_worker / nn_batch_size steuern den Durchsatz:
    Jeder Prozess faehrt `games_per_worker` Partien nebenlaeufig und buendelt die
    Netzauswertungen zu Batches von `nn_batch_size`. Mehr gleichzeitige Partien pro
    Prozess = vollere Batches = bessere GPU-Auslastung. Die Prozesse selbst braucht
    es, weil die Baumsuche in Python laeuft und damit CPU-gebunden ist.
    """

    enable_utf8_console()
    db = DatabaseHandler()

    for iteration in range(num_iterations):
        model_name = get_filename(generating_model_path)
        generating_model = load_model(generating_model_path, model_name)

        print(f"\n=== ITERATION {iteration} | aktuelles Modell: {model_name} ===")

        run_in_processes(
            process_entry_generate_dataset,
            [
                (
                    dataset_generation_time,
                    mcts_iteration_limit,
                    generating_model_path,
                    f"{model_name}_instance_{i}",
                    games_per_worker,
                    nn_batch_size,
                )
                for i in range(num_worker_processes)
            ],
        )

        buffer_size = 100000
        moves = db.load_moves_for_training(num_moves=buffer_size)

        print(f"→ {len(moves)} Moves geladen (Modell {model_name})")

        updated_model_path = update_model(generating_model_path,
                                     MoveDataset(moves),
                                     num_epochs=num_training_epochs)

        updated_model = load_model(updated_model_path, get_filename(updated_model_path))

        # Die Bewertungspartien werden auf die Prozesse *aufgeteilt*, nicht pro
        # Prozess neu gespielt - sonst laufen num_validation_games * num_prozesse.
        games_per_process = max(num_validation_games // num_worker_processes, 1)

        run_in_processes(
            process_entry_selfplay,
            [
                (
                    generating_model_path,
                    updated_model_path,
                    mcts_iteration_limit,
                    games_per_process,
                )
                for _ in range(num_worker_processes)
            ],
        )

        selfplay_statistics = db.get_selfplay_statistics_from_database(challenger_model_tag=updated_model.tag)

        print(f"The results are in. Challenger: {selfplay_statistics['challenger']}, "
              f"Champion: {selfplay_statistics['champion']}, Draws: {selfplay_statistics['draw']}")

        # Nur entschiedene Partien zaehlen - Remis im Nenner macht die Huerde
        # von der Remisquote abhaengig statt von der Spielstaerke.
        win_rate = gating_win_rate(selfplay_statistics)

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

        # --- Fester Massstab ---
        # Champion-vs-Challenger ist ein relatives Mass und kann im Kreis laufen.
        # Minimax mit fester Tiefe ist der einzige Anker, der ueber alle Iterationen
        # hinweg vergleichbar bleibt.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        asyncio.run(run_benchmark(
            model_path=generating_model_path,
            depths=list(anchor_depths),
            num_pairs_per_depth=anchor_pairs,
            iteration_limit=mcts_iteration_limit,
            device=device,
        ))

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
                              num_training_epochs=4,
                              num_validation_games=200,
                              generating_model_path=generating_model_path
                             )