import numpy as np

from src.ConnectFour import ConnectFour
from src.database.db_handler import DatabaseHandler
from src.database.db_models import TrainingGame, SelfplayGame


def display_final_boards(game_type="training", game_id=None):
    """
    Displays final boards of games for a given model.

    If game_id is provided, only that specific game is shown.
    """

    db = DatabaseHandler()
    session = db.session

    if game_type == "training":
        GameClass = TrainingGame
    elif game_type == "selfplay":
        GameClass = SelfplayGame
    else:
        raise ValueError(f"Unknown game_type: {game_type}")

    query = session.query(GameClass)

    if game_id is not None:
        query = query.filter(GameClass.id == game_id)

    games = query.all()

    if not games:
        print("No games found.")
        return

    print(f"Found {len(games)} {game_type} game(s)")

    for idx, game in enumerate(games, 1):
        last_move = max(game.moves, key=lambda m: m.move_number)
        final_board_array = np.array(last_move.board_state)

        board_obj = ConnectFour()
        board_obj.board = final_board_array

        print(f"\nGame {idx} | game_id={game.id} | Winner = {game.winner}")
        board_obj.display_board()


if __name__ == "__main__":
    display_final_boards(game_type="training", game_id=11531)  # "training" oder "selfplay"
