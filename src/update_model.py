import datetime
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.CFNet import CFNet, load_model
from src.database.db_handler import DatabaseHandler
from src.generate_training_data import MoveDataset
from src.utils import timing, get_filename


@timing
def update_model(input_model_path=None,
                 train_data=None,
                 num_epochs=4,
                 batch_size=256):

    model_name = get_filename(input_model_path)
    input_model = load_model(input_model_path,model_tag=model_name)

    print(f"UPDATING MODEL {input_model.tag}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_model.train()

    optimizer = torch.optim.AdamW(
        input_model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # Scheduler initialisieren
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1,
    )

    # DataLoader einmal bauen, nicht pro Epoche neu.
    dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda")
    )

    history = {"total": [], "value": [], "policy": [], "kl": []}

    for epoch in range(num_epochs):
        running = {"total": 0.0, "value": 0.0, "policy": 0.0, "kl": 0.0}

        for batch in dataloader:
            optimizer.zero_grad()

            states = batch[0].to(device, non_blocking=True)
            policy_targets = batch[1].to(device)
            value_targets = batch[2].to(device)

            # forward pass through model
            out = input_model(states)
            nn_value = out['value']
            nn_policy = out['policy']

            # calculate loss
            loss, value_loss, policy_loss = input_model.alphaloss(
                nn_value,
                nn_policy,
                value_targets,
                policy_targets
            )

            # backpropagation
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                kl = input_model.policy_kl(nn_policy, policy_targets)

            running["total"] += loss.item()
            running["value"] += value_loss.item()
            running["policy"] += policy_loss.item()
            running["kl"] += kl.item()

        num_batches = len(dataloader)
        for key in running:
            history[key].append(running[key] / num_batches)

        scheduler.step(history["total"][-1])

        # Der Policy-Anteil enthaelt die Entropie des MCTS-Ziels als festen Sockel.
        # Aussagekraeftig ist die KL - sie kann tatsaechlich gegen 0 gehen.
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"total {history['total'][-1]:.4f} | "
            f"value {history['value'][-1]:.4f} | "
            f"policy {history['policy'][-1]:.4f} | "
            f"policy-KL {history['kl'][-1]:.4f}"
        )

    print("MODEL UPDATED!")

    # Ermittle das Verzeichnis, in dem dieses Skript liegt (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Gehe eine Ebene hoch zum Projekt-Root und dann in die Zielordner
    project_root = os.path.dirname(script_dir)
    loss_output_dir = os.path.join(project_root, "data", "losses")
    model_output_dir = os.path.join(project_root, "models")

    os.makedirs(loss_output_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_path = os.path.join(loss_output_dir, f"training_loss_{timestamp}.png")
    model_path = os.path.join(model_output_dir, f"cfnet_{timestamp}.pt")

    input_model.tag = f"cfnet_{timestamp}"

    torch.save(input_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # --- Plot Loss ---
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["total"], label="Total")
    plt.plot(epochs, history["value"], label="Value (MSE)")
    plt.plot(epochs, history["kl"], label="Policy KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    return model_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    accepted_models_dir = os.path.join(project_root, "accepted_models")

    model_name = "cfnet_20260207_130926.pt"

    updating_model_path = os.path.join(accepted_models_dir, model_name)
    updating_model = load_model(model_path=updating_model_path, model_tag=model_name)

    db = DatabaseHandler()
    buffer_size = 100000

    moves = db.load_moves_for_training(num_moves=buffer_size)

    new_model = update_model(updating_model, MoveDataset(moves), num_epochs=20)
