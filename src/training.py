import datetime
import multiprocessing as mp
import os
import time

import torch

from src import training_log
from src.CFNet import create_initial_model, load_model
from src.database.db_handler import DatabaseHandler
from src.generate_training_data import MoveDataset, process_entry_generate_dataset
from src.selfplay_parallel import selfplay_parallel, process_entry_selfplay
from src.utils import (
    timing,
    get_filename,
    gating_win_rate,
    enable_utf8_console,
    even_games_per_worker,
)
from src.update_model import update_model, OPTIMIZER_DESC, SCHEDULER_DESC

GATING_WIN_RATE_THRESHOLD = 0.55


def accepted_models_dir():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "accepted_models")


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

    if generating_model_path is None:
        # Erster Lauf: mit einem untrainierten Netz anfangen.
        generating_model_path = create_initial_model(accepted_models_dir())

    for iteration in range(num_iterations):
        model_name = get_filename(generating_model_path)
        generating_model = load_model(generating_model_path, model_name)

        print(f"\n=== ITERATION {iteration} | aktuelles Modell: {model_name} ===")

        datagen_since = datetime.datetime.now(datetime.timezone.utc)
        datagen_start = time.time()
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
        datagen_duration = time.time() - datagen_start
        datagen_stats = db.get_training_game_stats(
            model_tag_prefix=f"{model_name}_instance_",
            since=datagen_since,
        )
        datagen_stats["duration_seconds"] = round(datagen_duration, 1)

        buffer_size = 100000
        moves = db.load_moves_for_training(num_moves=buffer_size)

        print(f"→ {len(moves)} Moves geladen (Modell {model_name})")

        update_start = time.time()
        updated_model_path, update_history = update_model(
            generating_model_path,
            MoveDataset(moves),
            num_epochs=num_training_epochs,
            batch_size=nn_batch_size,
        )
        update_duration = time.time() - update_start

        updated_model = load_model(updated_model_path, get_filename(updated_model_path))

        model_update_stats = {
            "num_training_examples": len(moves),
            "num_epochs": num_training_epochs,
            "batch_size": nn_batch_size,
            "optimizer": OPTIMIZER_DESC,
            "scheduler": SCHEDULER_DESC,
            "duration_seconds": round(update_duration, 1),
            "model_tag_before": generating_model.tag,
            "model_tag_after": updated_model.tag,
            **{f"{key}_per_epoch": values for key, values in update_history.items()},
        }

        # Die Bewertungspartien werden auf die Prozesse *aufgeteilt*, nicht pro
        # Prozess neu gespielt - sonst laufen num_validation_games * num_prozesse.
        # Gerade Anzahl je Prozess, damit beide Seiten gleich oft anfangen.
        games_per_process = even_games_per_worker(num_validation_games, num_worker_processes)

        gating_start = time.time()
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
        gating_duration = time.time() - gating_start

        selfplay_statistics = db.get_selfplay_statistics_from_database(challenger_model_tag=updated_model.tag)

        print(f"The results are in. Challenger: {selfplay_statistics['challenger']}, "
              f"Champion: {selfplay_statistics['champion']}, Draws: {selfplay_statistics['draw']}")

        # Nur entschiedene Partien zaehlen - Remis im Nenner macht die Huerde
        # von der Remisquote abhaengig statt von der Spielstaerke.
        win_rate = gating_win_rate(selfplay_statistics)

        print(f"Vergleich Version {generating_model.tag} vs Version {updated_model.tag}: {win_rate*100:.1f}% Winrate")

        gating_stats = {
            "num_games": sum(selfplay_statistics.values()),
            "wins_champion": selfplay_statistics["champion"],
            "wins_challenger": selfplay_statistics["challenger"],
            "draws": selfplay_statistics["draw"],
            "win_rate_challenger": round(win_rate, 4),
            "duration_seconds": round(gating_duration, 1),
            **db.get_selfplay_move_count_stats(challenger_model_tag=updated_model.tag),
        }

        # --- Entscheidung ---
        promoted = win_rate > GATING_WIN_RATE_THRESHOLD
        gating_stats["promoted"] = promoted
        if promoted:
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

        hyperparameters = {
            "dataset_generation_time_hours": dataset_generation_time,
            "mcts_iteration_limit": mcts_iteration_limit,
            "num_training_epochs": num_training_epochs,
            "num_validation_games": num_validation_games,
            "num_worker_processes": num_worker_processes,
            "games_per_worker": games_per_worker,
            "nn_batch_size": nn_batch_size,
            "gating_win_rate_threshold": GATING_WIN_RATE_THRESHOLD,
            **training_log.mcts_defaults(),
        }

        log_path = training_log.write_iteration_log(
            iteration=iteration,
            hyperparameters=hyperparameters,
            datagen=datagen_stats,
            gating=gating_stats,
            model_update=model_update_stats,
        )
        print(f"Iterations-Log geschrieben: {log_path}")

    return generating_model



if __name__ == "__main__":
    # Pfad zu einem vorhandenen Checkpoint eintragen, um dort weiterzumachen.
    # None = bei einem frischen, untrainierten Netz anfangen.
    generating_model_path = None

    final_model = train_model(num_iterations=10,
                              dataset_generation_time=0.5,
                              mcts_iteration_limit=400,
                              num_training_epochs=4,
                              num_validation_games=200,
                              generating_model_path=generating_model_path
                             )