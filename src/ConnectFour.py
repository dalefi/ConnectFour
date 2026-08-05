import numpy as np
from mcts.base.base import BaseState, BaseAction


class ConnectFour(BaseState):
    """
    A class for a connect four game. Contains a board(state) and who's turn it is.
    Players are 1 and -1. Empty spaces are 0, each space where a player's stone is
    has their number.
    """

    # Spaltenreihenfolge: Mitte zuerst – verbessert Alpha-Beta-Pruning erheblich
    COLUMN_ORDER = sorted(range(7), key=lambda c: abs(c - 3))  # [3,2,4,1,5,0,6]

    # Richtungen fuer die Gewinnpruefung: waagerecht, senkrecht, beide Diagonalen
    WIN_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(self, board=None, currentPlayer=1, last_move=None):
        """
        board: np.array of shape (6,7) with 0 for empty, 1 for Player 1's coin and -1 accordingly
        currentPlayer: +-1 for the current player
        last_move: tuple (row, col) of the last played coin, or None if unknown.
                   Ist er bekannt, genuegt eine lokale Gewinnpruefung um diesen Stein
                   herum; sonst muss das ganze Brett gescannt werden.
        """
        if board is None:
            board = np.zeros((6, 7), dtype=np.int8)

        self.board = board
        self.currentPlayer = currentPlayer
        self.last_move = last_move

        # Caches; None bedeutet "noch nicht berechnet". make_move setzt sie zurueck.
        self._winner = None
        self._terminal = None

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
        if self.get_winner() != 0:
            return -1
        if self.board_is_full():
            return 0
        raise AssertionError("Game hasn't finished but there is supposedly a reward")

    def is_terminal(self):
        if self._terminal is None:
            self._terminal = self.get_winner() != 0 or self.board_is_full()
        return self._terminal

    def take_action(self, action):
        newState = ConnectFour(
            board=self.board.copy(),
            currentPlayer=self.currentPlayer,
            last_move=self.last_move
        )
        newState.make_move(action.target_column, action.player)
        return newState

    def copy(self):
        """
        Eigenstaendige Kopie. Wird u.a. gebraucht, weil der MCTS-Baum eine Stellung
        ueber mehrere Zuege haelt, waehrend der Aufrufer auf seinem Objekt weiterspielt.
        """
        return ConnectFour(
            board=self.board.copy(),
            currentPlayer=self.currentPlayer,
            last_move=self.last_move
        )

    def switch_player(self):
        return (-1) * self.currentPlayer

    def _winner_around_last_move(self):
        """
        Eine Viererreihe kann nur ueber den zuletzt gesetzten Stein entstanden sein.
        Es genuegt also, von diesem Stein aus in vier Richtungen nach aussen zu zaehlen.
        """
        row, column = self.last_move
        board = self.board
        player = board.item(row, column)
        if player == 0:
            return 0

        for delta_row, delta_column in self.WIN_DIRECTIONS:
            in_a_row = 1
            for sign in (1, -1):
                step_row = delta_row * sign
                step_column = delta_column * sign
                current_row = row + step_row
                current_column = column + step_column
                while (0 <= current_row < 6 and 0 <= current_column < 7
                       and board.item(current_row, current_column) == player):
                    in_a_row += 1
                    if in_a_row >= 4:
                        return player
                    current_row += step_row
                    current_column += step_column

        return 0

    def _winner_by_full_scan(self):
        """
        Fallback fuer Bretter, deren letzter Zug unbekannt ist (z.B. aus einem
        Tensor oder aus der Datenbank rekonstruiert).
        """
        board = self.board
        for row in range(6):
            for column in range(7):
                player = board.item(row, column)
                if player == 0:
                    continue
                for delta_row, delta_column in self.WIN_DIRECTIONS:
                    end_row = row + 3 * delta_row
                    end_column = column + 3 * delta_column
                    if not (0 <= end_row < 6 and 0 <= end_column < 7):
                        continue
                    if all(
                        board.item(row + i * delta_row, column + i * delta_column) == player
                        for i in range(1, 4)
                    ):
                        return player
        return 0

    def board_is_full(self):
        # Steine stapeln sich von unten: ist die oberste Zeile voll, ist das Brett voll.
        return 0 not in self.board[0]

    def game_over(self):
        return self.is_terminal()

    def get_winner(self):
        if self._winner is None:
            if self.last_move is None:
                self._winner = self._winner_by_full_scan()
            else:
                self._winner = self._winner_around_last_move()
        return self._winner

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

        # Das Brett hat sich geaendert -> zwischengespeicherte Ergebnisse verwerfen
        self._winner = None
        self._terminal = None

        self.currentPlayer = self.switch_player()
        return played_move

    def random_move(self):
        available_columns = np.where(self.board[0, :] == 0)[0]
        random_column = np.random.choice(available_columns)
        self.make_move(random_column)

    def display_board(self):
        # Lokaler Import: matplotlib wird nur zum Anzeigen gebraucht und soll nicht
        # in jedem Selfplay-Worker-Prozess mitgeladen werden.
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

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
