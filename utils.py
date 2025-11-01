import os
import torch

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