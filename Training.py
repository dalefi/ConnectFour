from tqdm.notebook import tqdm
import selfplay
from generate_training_data import generate_dataset
from update_model import update_model
from CFNet import CFNet
import utils

def Training(num_iterations=100, target_dataset_size=1000, iteration_limit=250, num_training_epochs=10, num_validation_games=20):
    """
    This script wraps the whole training process:

    1. Initialize a model.
    2. Generate training data.
    3. Update the model.
    4. Validate if the model performs better than the old model.
    5. If yes replace current model, if no discard it.

    """

    print()
    current_model = CFNet()

    for iteration in tqdm(range(num_iterations), desc="Training Iterations"):
        training_data = generate_dataset(target_dataset_size=target_dataset_size, iteration_limit=iteration_limit, model=current_model)
        utils.save_data(training_data, f"training_data_at_{iteration}")
        updated_model = update_model(current_model, training_data, num_epochs=num_training_epochs)
        win_rate, draw_rate, loss_rate = selfplay.selfplay(current_model, updated_model, num_games=num_validation_games)

        if win_rate > 0.5:
            current_model = updated_model
            utils.save_model(current_model, f"accepted_at_{iteration}")
        else:
            utils.save_model(updated_model, f"rejected_at_{iteration}")

    return current_model