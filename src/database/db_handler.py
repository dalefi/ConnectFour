from sqlalchemy import func

from src.database.db_models import get_engine, get_session, SelfplayGame, SelfplayMove, TrainingGame, TrainingMove


class DatabaseHandler:
    def __init__(self, user="daniel", password="connectfour", host="localhost", port=5432, db="connectfour"):
        self.engine = get_engine(user, password, host, port, db)
        self.session = get_session(self.engine)

    def insert_training_game(self, winner, moves, model_tag):
        """
        Speichert ein TrainingGame + Moves in der DB.
        moves: Liste von Dicts mit Keys:
            "boardstate" (ConnectFour oder List[List[int]]),
            "mcts_policy" (np.ndarray oder List[float]),
            "result" (float),
            "current_player" (int)
        """

        game = TrainingGame(winner=winner,
                            model_tag=model_tag,
                            move_count=len(moves))

        self.session.add(game)
        self.session.flush()

        for move_number, move in enumerate(moves):
            # Board normalisieren
            board = move["boardstate"]
            if hasattr(board, "board"):  # ConnectFour-Objekt
                board = board.board.tolist()

            # Policy normalisieren
            policy = move["mcts_policy"]
            if hasattr(policy, "tolist"):  # np.ndarray
                policy = policy.tolist()

            nn_policy = move["nn_policy"]
            if hasattr(nn_policy, "tolist"):
                nn_policy = nn_policy.tolist()

            self.session.add(
                TrainingMove(
                    game_id=game.id,
                    move_number=move_number,
                    board_state=board,
                    policy=policy,
                    nn_policy=nn_policy,
                    value=float(move["result"]),
                    nn_evaluation=float(move["nn_eval"]),
                    current_player=int(move["current_player"])
                )
            )

        self.session.commit()

    def insert_selfplay_game(self, winner, moves, model_1_tag, model_2_tag):
        """
        Speichert ein SelfplayGame + Moves in der DB.
        moves: Liste von Dicts mit Keys:
            "board_state" (List[List[int]]),
            "policy" (np.ndarray oder List[float]),
            "value" (float),
            "current_player" (int),
            "model_role" ("current" | "updated")
        """
        game = SelfplayGame(
            winner=winner,
            move_count=len(moves),
            model_1_tag=model_1_tag,
            model_2_tag=model_2_tag
        )
        self.session.add(game)
        self.session.flush()

        for move_number, move in enumerate(moves):
            # Policy normalisieren
            policy = move["mcts_policy"]
            if hasattr(policy, "tolist"):
                policy = policy.tolist()

            nn_policy = move["nn_policy"]
            if hasattr(nn_policy, "tolist"):
                nn_policy = nn_policy.tolist()

            self.session.add(
                SelfplayMove(
                    game_id=game.id,
                    move_number=move_number,
                    board_state=move["boardstate"],
                    policy=policy,
                    nn_policy=nn_policy,
                    value=float(move["value"]),
                    nn_evaluation=float(move["nn_eval"]),
                    current_player=int(move["current_player"]),
                    model_role=move["model_role"]
                )
            )

        self.session.commit()


    def load_moves_for_training(self, num_moves: int):
        return (
            self.session
            .query(TrainingMove)
            .order_by(TrainingMove.id.desc())
            .limit(num_moves)
            .all()
        )


    def get_selfplay_statistics_from_database(self, challenger_model_tag: str) -> dict[str, int]:
        rows = (
            self.session
            .query(
                SelfplayGame.winner,
                func.count(SelfplayGame.winner)
            )
            .filter(SelfplayGame.model_2_tag == challenger_model_tag)
            .group_by(SelfplayGame.winner)
            .all()
        )

        # in Dictionary umwandeln
        stats = {winner: count for winner, count in rows}

        # optional: fehlende Keys mit 0 auffüllen
        for key in ("champion", "challenger", "draw"):
            stats.setdefault(key, 0)

        return stats
