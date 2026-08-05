import os
import sys
from pathlib import Path

import torch
from functools import wraps
from time import time


def save_model(model, filename):
    os.makedirs("models", exist_ok=True)

    # Example: save a model
    model_path = os.path.join("models", f"{filename}.pt")
    torch.save(model.state_dict(), model_path)

def save_data(data, filename):
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Example: save training data (e.g. a list of (state, policy, value) tuples)
    data_path = os.path.join("data", f"{filename}.pt")
    torch.save(data, data_path)

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"{f.__qualname__} took {te - ts:.6f}s")
        return result
    return wrap


def get_filename(path_str: str) -> str:
    return Path(path_str).stem


def enable_utf8_console() -> None:
    """
    Windows-Konsolen laufen standardmaessig unter cp1252. Zeichen wie '->' als
    Pfeil oder Rahmenlinien in den Fortschrittsausgaben loesen dort einen
    UnicodeEncodeError aus und reissen den ganzen Lauf mit.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def temperature_for_move(move_number: int, exploration_moves: int = 8) -> float:
    """
    Temperatur fuer die Zugauswahl im Selfplay.

    Die ersten Zuege werden proportional zu den MCTS-Besuchen gesampelt, damit die
    Partien auseinanderlaufen. Danach wird der meistbesuchte Zug gespielt - sonst
    wirft die Engine gewonnene Stellungen weg und die Value-Targets werden falsch.
    """
    return 1.0 if move_number < exploration_moves else 0.0


def gating_win_rate(stats: dict[str, int]) -> float:
    """
    Anteil gewonnener Partien des Herausforderers unter den *entschiedenen* Partien.

    Unentschieden gehoeren nicht in den Nenner: sonst haengt die Promotionshuerde
    an der Remisquote statt an der Spielstaerke.
    """
    challenger_wins = stats.get("challenger", 0)
    champion_wins = stats.get("champion", 0)
    decided_games = challenger_wins + champion_wins

    if decided_games == 0:
        return 0.0

    return challenger_wins / decided_games


def calculate_percentages(stats: dict[str, int]) -> dict[str, float]:
    total = sum(stats.values())

    if total == 0:
        return {key: 0.0 for key in stats}

    return {
        key: (value / total)
        for key, value in stats.items()
    }