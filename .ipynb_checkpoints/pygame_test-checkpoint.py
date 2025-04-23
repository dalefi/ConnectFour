import pygame
import sys
import numpy as np

# Constants
ROWS = 6
COLS = 7
SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5
WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 1) * SQUARESIZE  # Extra row for showing turn
SIZE = (WIDTH, HEIGHT)

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Game state
board = np.zeros((ROWS, COLS), dtype=int)  # 0 = empty, -1 = red, 1 = yellow
current_player = -1  # Start with red

pygame.init()
screen = pygame.display.set_mode(SIZE)
font = pygame.font.SysFont("monospace", 75)

def draw_board():
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, HEIGHT - (r+1)*SQUARESIZE, SQUARESIZE, SQUARESIZE))
            value = board[r][c]
            if value == 0:
                color = BLACK
            elif value == -1:
                color = RED
            else:
                color = YELLOW
            pygame.draw.circle(screen, color, (c*SQUARESIZE + SQUARESIZE//2, HEIGHT - (r * SQUARESIZE + SQUARESIZE//2)), RADIUS)
    pygame.display.update()


def drop_piece(col, player):
    for r in range(ROWS):
        if board[r][col] == 0:
            board[r][col] = player
            return True
    return False

def check_win(player):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c+i] == player for i in range(4)):
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r+i][c] == player for i in range(4)):
                return True
    # Diagonal /
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True
    # Diagonal \
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True
    return False

# Initial draw
draw_board()

# Main loop
game_over = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if game_over:
            continue

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
            x_pos = event.pos[0]
            color = RED if current_player == -1 else YELLOW
            pygame.draw.circle(screen, color, (x_pos, SQUARESIZE//2), RADIUS)
            pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            x_pos = event.pos[0]
            col = x_pos // SQUARESIZE

            if drop_piece(col, current_player):
                if check_win(current_player):
                    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
                    label = font.render("Red wins!" if current_player == -1 else "Yellow wins!", 1, RED if current_player == -1 else YELLOW)
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    game_over = True
                current_player *= -1  # Switch player

                draw_board()
