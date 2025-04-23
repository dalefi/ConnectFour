import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from mcts.base.base import BaseState, BaseAction

class ConnectFour(BaseState):
    """
    A class for a connect four game. Contains a board(state) and who's turn it is.
    Players are 1 and -1. Empty spaces are 0, each space where a player's stone is
    has their number.
    """
    
    def __init__(self, board=None, currentPlayer=None, last_move=None):
        """
        board: np.array of shape (6,7) with 0 for empty, 1 for Player 1's coin and -1 accordingly
        currentPlayer: +-1 for the current player
        last_move: tuple (x,y) for position of last played coin
        """
        self.board = np.zeros(shape=(6,7),dtype=int) # board is empty in the beginning
        self.currentPlayer = 1 # player 1 starts
        self.last_move = (0,0) # easier to check for a winner this way

    def __str__(self):
        return str((self.board, f"Turn: Player {self.currentPlayer}"))

    def get_current_player(self):
        return self.currentPlayer

    def get_possible_actions(self):
        available_columns = np.where(self.board[0,:]==0)[0]
        possibleActions = [Action(target_column=i, player=self.currentPlayer) for i in available_columns]
        
        return possibleActions

    def get_reward(self):
        if self.vertical_check() != 0:
            reward = self.vertical_check()

        elif self.horizontal_check() != 0:
            reward = self.horizontal_check()

        elif self.diagonal_check() != 0:
            reward = self.diagonal_check()

        elif self.board_is_full():
            reward = 0

        else:
            raise AssertionError("There is something fishy here, the game hasnt finished but there is supposedly a reward")

        #print(f"Reward = {reward}")
        return reward

    def is_terminal(self):
        return self.game_over()

    def take_action(self, action):
        newState = deepcopy(self)
        newState.make_move(action.target_column, action.player)
        return newState

    def switch_player(self):
        """
        Switches who's turn it is.
        """
        return (-1)*self.currentPlayer

    def vertical_check(self):
        """
        Checks if the last move leads to a vertical win.
        """

        for col_idx in range(self.board.shape[1]):
            for row_idx in range(self.board.shape[0]-2):
                if self.board[row_idx:row_idx+4,col_idx].sum() == self.currentPlayer*4:
                    return (-1)*self.currentPlayer

        return 0

    def horizontal_check(self):
        """
        Checks if the last move leads to a horizontal win.
        """

        for row_idx in range(self.board.shape[0]):
            for col_idx in range(self.board.shape[1]-2):
                if self.board[row_idx,col_idx:col_idx+4].sum() == self.currentPlayer*4:
                    return (-1)*self.currentPlayer
        return 0

    def diagonal_check(self):
        """
        Checks if the last move leads to a diagonal win.
        """

        # from left-up to right-down
        for row_idx in range(self.board.shape[0]-3):
            for col_idx in range(self.board.shape[1]-3):
                offset = col_idx - row_idx
                if self.board.diagonal(offset=offset)[:4].sum() == self.currentPlayer*4:
                    return (-1)*self.currentPlayer
        return 0

    def board_is_full(self):
        # if there is any 0 left, the board is not full
        if not np.isin(self.board,0).any():
            return True
        return False
    
    # checks if the game is over
    def game_over(self):
        # first check if there is a winner
        if abs(self.vertical_check()) or abs(self.horizontal_check()) or abs(self.diagonal_check()):
            return True

        elif self.board_is_full():
            return True

        else:
            return False
    
    def make_move(self, target_column=None, currentPlayer=None):
        """
        Makes a move given a target column. Returns a tuple with the space where the coin has landed.
        """

        if currentPlayer is None:
            currentPlayer = self.currentPlayer
        
        played_move = None
        
        # check if input column is viable
        if not target_column in range(7):
            raise ValueError("This is not a column you can play in!")
            return played_move

        # check if column is already full
        if self.board[0, target_column] != 0:
            raise ValueError("This column is full!")
            return played_move

        # put the coin at its place
        for i in range(6):
            if self.board[5-i, target_column] != 0:
                i = i+1
            else: 
                self.board[5-i, target_column] = currentPlayer
                played_move = (5-i, target_column)
                self.last_move = played_move
                break

        self.currentPlayer = self.switch_player()

        return played_move

    def random_move(self):
        # available playable columns
        available_columns = np.where(self.board[0,:]==0)[0]
        random_column = np.random.choice(available_columns)

        self.make_move(random_column)

        return

    def display_board(self):
        """
        A matplotlib function to display a board in a nice way.
        """
        
        board = self.board
        rows, cols = board.shape
    
        cell_size = 0.6
        fig, ax = plt.subplots(figsize=(cols * cell_size, rows * cell_size))
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
                # Row inversion only here to make bottom of array = bottom of board
                circle = plt.Circle((c + 0.5, r + 0.5), 0.25, color=color, ec='black')
                ax.add_patch(circle)
    
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()  # Flip Y so that row 0 is at the bottom visually
        plt.grid(False)
        plt.tight_layout()
        plt.show()



class Action(BaseAction):
    def __init__(self, target_column=None, player=None):
        self.target_column = target_column
        self.player = player

    def __str__(self):
        return str((self.target_column, self.player))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.target_column == other.target_column and self.player == other.player and self.player == other.player

    def __hash__(self):
        return hash((self.target_column, self.player))


class ConnectFourGame():
    """
    A class to record a game of Connect Four. All the moves, policies, values and the outcome are stored here.
    """








