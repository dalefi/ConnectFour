import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mcts.base.base import BaseState, BaseAction


class ConnectFour(BaseState):
    """
    A class for a connect four game. Contains a board(state) and who's turn it is.
    Players are 1 and -1. Empty spaces are 0, each space where a player's stone is
    has their number.
    """

    # Spaltenreihenfolge: Mitte zuerst – verbessert Alpha-Beta-Pruning erheblich
    COLUMN_ORDER = sorted(range(7), key=lambda c: abs(c - 3))  # [3,2,4,1,5,0,6]

    def __init__(self, board=None, currentPlayer=1, last_move=(0, 0)):
        """
        board: np.array of shape (6,7) with 0 for empty, 1 for Player 1's coin and -1 accordingly
        currentPlayer: +-1 for the current player
        last_move: tuple (x,y) for position of last played coin
        """
        if board is None:
            board = np.zeros((6, 7), dtype=np.int8)

        self.board = board
        self.currentPlayer = currentPlayer
        self.last_move = last_move

    def __str__(self):
        return str((self.board, f"Turn: Player {self.currentPlayer}"))

    def get_current_player(self):
        return self.currentPlayer

    def get_possible_actions(self):
        available_columns = np.where(self.board[0, :] == 0)[0]
        possibleActions = [Action(target_column=i, player=self.currentPlayer) for i in available_columns]
        return possibleActions

    def get_reward(self):
        # Always return -1 to the player whose turn it is now -> they lost
        if self.vertical_check() != 0:
            reward = -1
        elif self.horizontal_check() != 0:
            reward = -1
        elif self.diagonal_check() != 0:
            reward = -1
        elif self.board_is_full():
            reward = 0
        else:
            raise AssertionError("Game hasn't finished but there is supposedly a reward")
        return reward

    def is_terminal(self):
        return self.game_over()

    def take_action(self, action):
        newState = ConnectFour(
            board=self.board.copy(),
            currentPlayer=self.currentPlayer,
            last_move=self.last_move
        )
        newState.make_move(action.target_column, action.player)
        return newState

    def switch_player(self):
        return (-1) * self.currentPlayer

    def vertical_check(self):
        for col_idx in range(self.board.shape[1]):
            for row_idx in range(self.board.shape[0] - 3):
                window = self.board[row_idx:row_idx + 4, col_idx]
                if np.all(window == window[0]) and window[0] != 0:
                    return window[0]
        return 0

    def horizontal_check(self):
        for row_idx in range(self.board.shape[0]):
            for col_idx in range(self.board.shape[1] - 3):
                window = self.board[row_idx, col_idx:col_idx + 4]
                if np.all(window == window[0]) and window[0] != 0:
                    return window[0]
        return 0

    def diagonal_check(self):
        rows, cols = self.board.shape

        # Check \ direction
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [self.board[r + i][c + i] for i in range(4)]
                if window[0] != 0 and all(cell == window[0] for cell in window):
                    return window[0]

        # Check / direction
        for r in range(3, rows):
            for c in range(cols - 3):
                window = [self.board[r - i][c + i] for i in range(4)]
                if window[0] != 0 and all(cell == window[0] for cell in window):
                    return window[0]

        return 0

    def board_is_full(self):
        return not np.isin(self.board, 0).any()

    def game_over(self):
        if abs(self.vertical_check()) or abs(self.horizontal_check()) or abs(self.diagonal_check()):
            return True
        elif self.board_is_full():
            return True
        return False

    def get_winner(self):
        for check in [self.vertical_check, self.horizontal_check, self.diagonal_check]:
            winner = check()
            if winner != 0:
                return winner
        return 0

    def make_move(self, target_column=None, currentPlayer=None):
        if currentPlayer is None:
            currentPlayer = self.currentPlayer

        played_move = None

        if target_column not in range(7):
            raise ValueError("This is not a column you can play in!")

        if self.board[0, target_column] != 0:
            raise ValueError("This column is full!")

        for i in range(6):
            if self.board[5 - i, target_column] != 0:
                i = i + 1
            else:
                self.board[5 - i, target_column] = currentPlayer
                played_move = (5 - i, target_column)
                self.last_move = played_move
                break

        self.currentPlayer = self.switch_player()
        return played_move

    def random_move(self):
        available_columns = np.where(self.board[0, :] == 0)[0]
        random_column = np.random.choice(available_columns)
        self.make_move(random_column)

    def display_board(self):
        board = self.board
        rows, cols = board.shape

        cell_size = 0.6
        fig, ax = plt.subplots(figsize=(cols * cell_size, rows * cell_size + 0.5))
        ax.set_aspect('equal')
        ax.set_facecolor('blue')

        for r in range(rows):
            for c in range(cols):
                value = board[r, c]
                if value == -1:
                    color = 'red'
                elif value == 1:
                    color = 'yellow'
                else:
                    color = 'white'
                circle = plt.Circle((c + 0.5, r + 0.5), 0.25, color=color, ec='black')
                ax.add_patch(circle)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        plt.grid(False)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        red_patch = mpatches.Patch(color='red', label='Player -1 (Red)')
        yellow_patch = mpatches.Patch(color='yellow', label='Player 1 (Yellow)')
        ax.legend(
            handles=[red_patch, yellow_patch],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=False
        )
        plt.show()

    @staticmethod
    def random_start_state(max_random_moves=6, max_retries=100):
        for _ in range(max_retries):
            state = ConnectFour()
            num_moves = np.random.randint(0, max_random_moves + 1)
            valid = True

            for _ in range(num_moves):
                if state.is_terminal():
                    valid = False
                    break
                possible_actions = state.get_possible_actions()
                if not possible_actions:
                    valid = False
                    break
                action = np.random.choice(possible_actions)
                state = state.take_action(action)

            if valid and not state.is_terminal():
                return state

        raise RuntimeError(
            "Failed to generate a non-terminal random start state. "
            "Try lowering max_random_moves."
        )

    # -------------------------------------------------------------------------
    # Minimax – Variante 1: MIT Heuristik (schnell, empfohlen für Praxis)
    # -------------------------------------------------------------------------

    def _score_window(self, window, player):
        """
        Bewertet ein Fenster von 4 Feldern heuristisch für `player`.
        Positive Werte = gut für player, negative = gut für Gegner.
        """
        score = 0
        opp = -player

        own = np.count_nonzero(window == player)
        empty = np.count_nonzero(window == 0)
        opp_count = np.count_nonzero(window == opp)

        if own == 4:
            score += 100
        elif own == 3 and empty == 1:
            score += 5
        elif own == 2 and empty == 2:
            score += 2

        if opp_count == 3 and empty == 1:
            score -= 4  # Bedrohung des Gegners blockieren

        return score

    def _heuristic_score(self, player):
        """
        Heuristischer Gesamtscore für `player` aus aktueller Brettposition.
        Wird verwendet wenn depth == 0 und das Spiel noch nicht vorbei ist.
        """
        score = 0
        board = self.board

        # Mittelspalte bevorzugen (strategisch wertvoll)
        center_col = board[:, 3]
        score += int(np.count_nonzero(center_col == player)) * 3

        # Horizontal
        for r in range(6):
            for c in range(4):
                window = board[r, c:c + 4]
                score += self._score_window(window, player)

        # Vertikal
        for c in range(7):
            for r in range(3):
                window = board[r:r + 4, c]
                score += self._score_window(window, player)

        # Diagonal \
        for r in range(3):
            for c in range(4):
                window = np.array([board[r + i][c + i] for i in range(4)])
                score += self._score_window(window, player)

        # Diagonal /
        for r in range(3, 6):
            for c in range(4):
                window = np.array([board[r - i][c + i] for i in range(4)])
                score += self._score_window(window, player)

        return score

    def _minimax_heuristic(self, depth, alpha, beta, ai_player):
        """
        Minimax mit Alpha-Beta-Pruning und Heuristik bei depth == 0.

        Wer maximiert/minimiert wird direkt aus self.currentPlayer abgeleitet –
        kein separates maximizing_player-Flag, das beim Togglen falsch laufen kann.

        Parameters
        ----------
        depth     : verbleibende Suchtiefe
        alpha     : Alpha-Schranke
        beta      : Beta-Schranke
        ai_player : Spielerkennung des KI-Spielers (1 oder -1)

        Returns
        -------
        (score, best_column)
        """
        valid_cols = [c for c in self.COLUMN_ORDER if self.board[0, c] == 0]

        # Terminaler Zustand
        if self.is_terminal():
            winner = self.get_winner()
            if winner == ai_player:
                return (10_000_000 + depth, None)   # Sieg – früher ist besser
            elif winner == -ai_player:
                return (-10_000_000 - depth, None)  # Niederlage – später ist besser
            else:
                return (0, None)                    # Unentschieden

        # Tiefenlimit erreicht → Heuristik immer aus Sicht von ai_player
        if depth == 0:
            return (self._heuristic_score(ai_player), None)

        # Maximiere wenn ai_player am Zug ist, minimiere wenn Gegner dran ist
        if self.currentPlayer == ai_player:
            best_score = -np.inf
            best_col = valid_cols[0]
            for col in valid_cols:
                action = Action(target_column=col, player=self.currentPlayer)
                child = self.take_action(action)
                score, _ = child._minimax_heuristic(depth - 1, alpha, beta, ai_player)
                if score > best_score:
                    best_score, best_col = score, col
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break  # Beta-Cutoff
            return (best_score, best_col)

        else:
            best_score = np.inf
            best_col = valid_cols[0]
            for col in valid_cols:
                action = Action(target_column=col, player=self.currentPlayer)
                child = self.take_action(action)
                score, _ = child._minimax_heuristic(depth - 1, alpha, beta, ai_player)
                if score < best_score:
                    best_score, best_col = score, col
                beta = min(beta, best_score)
                if alpha >= beta:
                    break  # Alpha-Cutoff
            return (best_score, best_col)

    def get_best_move(self, depth=6):
        """
        Gibt die beste Spalte für den aktuellen Spieler zurück (mit Heuristik).

        Parameters
        ----------
        depth : Suchtiefe. Empfehlung:
                  5 → sehr schnell (~ms)
                  6 → guter Kompromiss (Standard)
                  7 → stark, aber merklich langsamer
                  8+ → langsam, ohne Transpositionstabelle nicht empfohlen

        Returns
        -------
        int – beste Spalte (0–6)
        """
        _, best_col = self._minimax_heuristic(
            depth=depth,
            alpha=-np.inf,
            beta=np.inf,
            ai_player=self.currentPlayer
        )
        return best_col


class Action(BaseAction):
    def __init__(self, target_column=None, player=None):
        self.target_column = target_column
        self.player = player

    def __str__(self):
        return str((self.target_column, self.player))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.target_column == other.target_column
                and self.player == other.player)

    def __hash__(self):
        return hash((self.target_column, self.player))
