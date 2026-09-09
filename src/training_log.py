import datetime
import inspect
import os
import subprocess

from mcts.searcher.mcts_searcher import mcts_searcher


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def mcts_defaults() -> dict:
    """
    exploration_constant/dirichlet_alpha/noise_fraction werden nirgends von
    training.py durchgereicht, sondern kommen aus den Konstruktor-Defaults von
    mcts_searcher. Per inspect auslesen statt hier zu duplizieren, damit das
    Log nicht stillschweigend veraltet, wenn sich die Defaults dort aendern.
    """
    params = inspect.signature(mcts_searcher.__init__).parameters
    keys = ("exploration_constant", "dirichlet_alpha", "noise_fraction")
    return {key: params[key].default for key in keys}


def _iteration_log_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "data", "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def write_iteration_log(
    iteration: int,
    hyperparameters: dict,
    datagen: dict,
    gating: dict,
    model_update: dict,
) -> str:
    """
    Schreibt die Metadaten einer Trainingsiteration als lesbare .txt-Datei nach
    data/training_logs/. Gedacht zum Ueberwachen/Vergleichen von Trainingslaeufen,
    daher menschenlesbar (key: value) statt maschinenoptimiert.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(_iteration_log_dir(), f"iteration_{iteration:04d}_{timestamp}.txt")

    lines = [f"iteration: {iteration}", f"timestamp: {timestamp}", f"git_commit: {get_git_commit()}"]

    sections = {
        "hyperparameters": hyperparameters,
        "selfplay_datagen": datagen,
        "gating_match": gating,
        "model_update": model_update,
    }

    for title, values in sections.items():
        lines.append("")
        lines.append(f"[{title}]")
        for key, value in values.items():
            lines.append(f"{key}: {value}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return log_path
