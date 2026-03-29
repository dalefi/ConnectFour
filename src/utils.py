import os
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


def calculate_percentages(stats: dict[str, int]) -> dict[str, float]:
    total = sum(stats.values())

    if total == 0:
        return {key: 0.0 for key in stats}

    return {
        key: (value / total)
        for key, value in stats.items()
    }