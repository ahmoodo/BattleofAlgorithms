import numpy as np
import pygame
import sys
import math
from threading import Timer
import random

ROWS = 6
COLS = 7

AI1_TURN = 0
AI2_TURN = 1

AI1_PIECE = 1
AI2_PIECE = 2

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


def create_board():
    board = np.zeros((ROWS, COLS))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == 0:
            return r

def winning_move(board, piece):

    for c in range(COLS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    for c in range(COLS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    for c in range(COLS-3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

    for c in range(3,COLS):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c-1] == piece and board[r-2][c-2] == piece and board[r-3][c-3] == piece:
                return True

def draw_board(board):
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            if board[r][c] == 0:
                pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE/2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)
            elif board[r][c] == 1:
                # RED = AI1 (Minimax)
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE/2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)
            else:
                # YELLOW = AI2 (MCTS)
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE/2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)

    pygame.display.update()


def evaluate_window(window, piece):
    opponent_piece = AI2_PIECE if piece == AI1_PIECE else AI1_PIECE
    score = 0
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opponent_piece) == 3 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, piece):

    score = 0

    center_array = [int(i) for i in list(board[:,COLS//2])]
    center_count = center_array.count(piece)
    score += center_count * 6

    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLS - 3):
            window = row_array[c:c + 4]
            score += evaluate_window(window, piece)

    for c in range(COLS):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROWS-3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    for r in range(3,ROWS):
        for c in range(COLS - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    for r in range(3,ROWS):
        for c in range(3,COLS):
            window = [board[r-i][c-i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    return winning_move(board, AI1_PIECE) or winning_move(board, AI2_PIECE) or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizing_player):

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI1_PIECE):
                return (None, 10000000)
            elif winning_move(board, AI2_PIECE):
                return (None, -10000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI1_PIECE))

    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_locations)

        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI1_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(value, alpha)
            if alpha >= beta:
                break

        return column, value

    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI2_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(value, beta)
            if alpha >= beta:
                break
        return column, value


def get_valid_locations(board):
    valid_locations = []

    for column in range(COLS):
        if is_valid_location(board, column):
            valid_locations.append(column)

    return valid_locations


class MCTSNode:
    def __init__(self, board, piece, parent=None, move=None):
        self.board = board
        self.piece = piece
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = get_valid_locations(board)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c=1.4):
        return max(self.children, key=lambda n: n.wins / n.visits + c * math.sqrt(math.log(self.visits) / n.visits))

    def expand(self):
        col = self.untried_moves.pop()
        next_piece = AI1_PIECE if self.piece == AI2_PIECE else AI2_PIECE
        b_copy = self.board.copy()
        row = get_next_open_row(b_copy, col)
        drop_piece(b_copy, row, col, next_piece)
        child = MCTSNode(b_copy, next_piece, parent=self, move=col)
        self.children.append(child)
        return child

    def rollout(self):
        current_board = self.board.copy()
        current_piece = self.piece
        while not is_terminal_node(current_board):
            valid = get_valid_locations(current_board)
            col = random.choice(valid)
            row = get_next_open_row(current_board, col)
            drop_piece(current_board, row, col, current_piece)
            current_piece = AI1_PIECE if current_piece == AI2_PIECE else AI2_PIECE
        if winning_move(current_board, AI2_PIECE):
            return 1
        elif winning_move(current_board, AI1_PIECE):
            return -1
        return 0

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)


def mcts_move(board, iterations=500):
    root = MCTSNode(board.copy(), AI2_PIECE)
    for _ in range(iterations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        if not is_terminal_node(node.board) and not node.is_fully_expanded():
            node = node.expand()
        result = node.rollout()
        node.backpropagate(result)
    if not root.children:
        return random.choice(get_valid_locations(board))
    best = max(root.children, key=lambda n: n.visits)
    return best.move


def end_game():
    global game_over
    game_over = True
    print(game_over)


board = create_board()

game_over = False
not_over = True
turn = random.randint(AI1_TURN, AI2_TURN)

pygame.init()

SQUARESIZE = 100
width = COLS * SQUARESIZE
height = (ROWS + 1) * SQUARESIZE
circle_radius = int(SQUARESIZE/2 - 5)
size = (width, height)
screen = pygame.display.set_mode(size)

my_font = pygame.font.SysFont("monospace", 75)

pygame.display.set_caption("Connect 4: Minimax vs MCTS")
draw_board(board)
pygame.display.update()
pygame.time.wait(1000)

while not game_over:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    if not_over:
        pygame.time.wait(400)

        if turn == AI1_TURN:
            col, _ = minimax(board, 5, -math.inf, math.inf, True)
            piece = AI1_PIECE
            win_msg = "AI 1 WINS!"
            win_color = RED
        else:
            col = mcts_move(board, iterations=500)
            piece = AI2_PIECE
            win_msg = "AI 2 WINS!"
            win_color = YELLOW

        if col is not None and is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, piece)

            if winning_move(board, piece):
                print(win_msg)
                label = my_font.render(win_msg, 1, win_color)
                screen.blit(label, (40, 10))
                not_over = False
                t = Timer(3.0, end_game)
                t.start()
            elif len(get_valid_locations(board)) == 0:
                print("DRAW!")
                label = my_font.render("DRAW!", 1, BLUE)
                screen.blit(label, (40, 10))
                not_over = False
                t = Timer(3.0, end_game)
                t.start()

            draw_board(board)
            turn = (turn + 1) % 2

        pygame.display.update()
