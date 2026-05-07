import numpy as np
import pygame
import sys
import math
import time
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

def draw_board(board, screen, SQUARESIZE, circle_radius):
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            if board[r][c] == 0:
                pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE/2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)
            elif board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE/2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)
            else:
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
        current_piece = AI1_PIECE if self.piece == AI2_PIECE else AI2_PIECE
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
    root = MCTSNode(board.copy(), AI1_PIECE)
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


def play_game(headless=False):
    """
    Play a single game between AI1 (minimax) and AI2 (MCTS).
    Returns a dict with the game result and per-move decision times.
    """
    board = create_board()
    turn = random.randint(AI1_TURN, AI2_TURN)
    starter = turn 

    ai1_times = []
    ai2_times = []
    winner = None  
   
    screen = None
    SQUARESIZE = 100
    circle_radius = int(SQUARESIZE / 2 - 5)
    my_font = None

    if not headless:
        pygame.init()
        width = COLS * SQUARESIZE
        height = (ROWS + 1) * SQUARESIZE
        size = (width, height)
        screen = pygame.display.set_mode(size)
        my_font = pygame.font.SysFont("monospace", 75)
        pygame.display.set_caption("Connect 4: Minimax vs MCTS")
        draw_board(board, screen, SQUARESIZE, circle_radius)
        pygame.display.update()
        pygame.time.wait(1000)

    game_over = False
    not_over = True

    while not game_over:
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        if not_over:
            if not headless:
                pygame.time.wait(400)

            start = time.perf_counter()
            if turn == AI1_TURN:
                col, _ = minimax(board, 5, -math.inf, math.inf, True)
                piece = AI1_PIECE
                win_msg = "AI 1 WINS!"
                win_color = RED
            else:
                col = mcts_move(board, iterations=500)
                #tesing at:
                # depth - iterations
                # 1 - 500
                # 1 - 2000
                # 3 - 2000
                # 3 - 5000
                # 5 - 5000
                # 5 - 15000

                piece = AI2_PIECE
                win_msg = "AI 2 WINS!"
                win_color = YELLOW
            elapsed = time.perf_counter() - start

            if turn == AI1_TURN:
                ai1_times.append(elapsed)
            else:
                ai2_times.append(elapsed)

            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, piece)

                if winning_move(board, piece):
                    winner = piece
                    if not headless:
                        print(win_msg)
                        label = my_font.render(win_msg, 1, win_color)
                        screen.blit(label, (40, 10))
                    not_over = False
                elif len(get_valid_locations(board)) == 0:
                    winner = 0  
                    if not headless:
                        print("DRAW!")
                        label = my_font.render("DRAW!", 1, BLUE)
                        screen.blit(label, (40, 10))
                    not_over = False

                if not headless:
                    draw_board(board, screen, SQUARESIZE, circle_radius)
                turn = (turn + 1) % 2

            if not headless:
                pygame.display.update()

        if not not_over:
            if headless:
                game_over = True
            else:
                pygame.time.wait(3000)
                game_over = True

    if not headless:
        pygame.quit()

    return {
        "winner": winner,
        "ai1_times": ai1_times,
        "ai2_times": ai2_times,
        "starter": starter,
    }


def run_batch(num_games):
    """Run many games headless and print aggregate stats."""
    ai1_wins = 0
    ai2_wins = 0
    draws = 0
    all_ai1_times = []
    all_ai2_times = []
    ai1_starts = 0
    ai2_starts = 0

    print(f"Running {num_games} games (headless)...")
    start_time = time.perf_counter()

    for i in range(num_games):
        result = play_game(headless=True)
        if result["winner"] == AI1_PIECE:
            ai1_wins += 1
        elif result["winner"] == AI2_PIECE:
            ai2_wins += 1
        else:
            draws += 1

        all_ai1_times.extend(result["ai1_times"])
        all_ai2_times.extend(result["ai2_times"])

        if result["starter"] == AI1_TURN:
            ai1_starts += 1
        else:
            ai2_starts += 1

        print(f"  Game {i+1}/{num_games} done. Winner: "
              f"{'AI1' if result['winner'] == AI1_PIECE else ('AI2' if result['winner'] == AI2_PIECE else 'Draw')}")

    total_time = time.perf_counter() - start_time

    avg_ai1 = sum(all_ai1_times) / len(all_ai1_times) if all_ai1_times else 0.0
    avg_ai2 = sum(all_ai2_times) / len(all_ai2_times) if all_ai2_times else 0.0

    print()
    print("=" * 50)
    print(f"RESULTS OVER {num_games} GAMES")
    print("=" * 50)
    print(f"AI1 (Minimax) wins: {ai1_wins}")
    print(f"AI2 (MCTS)    wins: {ai2_wins}")
    print(f"Draws:              {draws}")
    print()
    print(f"AI1 avg decision time: {avg_ai1:.4f} s  ({len(all_ai1_times)} moves)")
    print(f"AI2 avg decision time: {avg_ai2:.4f} s  ({len(all_ai2_times)} moves)")
    print()
    print(f"AI1 went first: {ai1_starts} times")
    print(f"AI2 went first: {ai2_starts} times")
    print(f"Total batch wall time: {total_time:.2f} s")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [num_games]")
            sys.exit(1)
        run_batch(n)
    else:
        play_game(headless=False)