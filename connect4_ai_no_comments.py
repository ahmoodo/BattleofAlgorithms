import numpy as np
import pygame
import sys
import math
from threading import Timer
import random

ROWS = 6
COLS = 7

PLAYER_TURN = 0
AI_TURN = 1

PLAYER_PIECE = 1
AI_PIECE = 2

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# ==================== MCTS (Monte Carlo Tree Search) Classes ====================

class MCTSNode:
    """Monte Carlo tree node class."""
    
    def __init__(self, parent, board, turn):
        self.q = 0  # sum of rollout outcomes
        self.n = 0  # number of visits
        self.parent = parent
        self.board = board.copy()
        # root is always opponent's turn
        if turn == PLAYER_PIECE:
            self.turn = AI_PIECE
        else:
            self.turn = PLAYER_PIECE
        # no children have been expanded yet
        self.children = []
        self.terminal = self.check_terminal()
        self.expanded = False
    
    def check_terminal(self):
        """Check whether node is a leaf (game ending state)."""
        if mcts_check_win(self.board, PLAYER_PIECE):
            return True
        if mcts_check_win(self.board, AI_PIECE):
            return True
        if mcts_check_tie(self.board):
            return True
        return False
    
    def add_child(self):
        """Add new child to node."""
        if self.expanded:
            return
        # get board of every child
        child_board = []
        for child in self.children:
            child_board.append(child.board)
        # find new child
        for i in range(COLS):
            if self.board[0, i] == 0:
                for j in range(ROWS):
                    if self.board[j, i] == 0:
                        tmp = self.board.copy()
                        if self.turn == PLAYER_PIECE:
                            tmp[j, i] = AI_PIECE
                            if child_board:
                                if not self.compare_children(tmp, child_board):
                                    self.children.append(MCTSNode(self, tmp, PLAYER_PIECE))
                                    return
                            self.children.append(MCTSNode(self, tmp, PLAYER_PIECE))
                            return
                        else:
                            tmp[j, i] = PLAYER_PIECE
                            if child_board:
                                if not self.compare_children(tmp, child_board):
                                    self.children.append(MCTSNode(self, tmp, AI_PIECE))
                                    return
                            self.children.append(MCTSNode(self, tmp, AI_PIECE))
                            return
        self.expanded = True
        return
    
    def compare_children(self, new_child, children):
        """Check if node state is equal to one of children state."""
        for child in children:
            if (new_child == child).all():
                return True
        return False


class MCTS:
    """Monte Carlo Tree search class."""
    
    def __init__(self, symbol, t):
        self.symbol = symbol
        self.t = t
    
    def compute_move(self, node):
        """Compute move using MCTS algorithm."""
        import time
        time0 = time.time()
        while (time.time() - time0) < self.t:
            # selection and expansion
            leaf = self.select(node)
            if leaf is None:
                return -1
            # simulation
            simulation_result = self.rollout(leaf)
            # backpropagation
            self.backpropagate(leaf, simulation_result)
        # from next best state get move coordinates
        selected = self.best_child(node)
        if selected is None:
            return -1
        for j in range(ROWS):
            for i in range(COLS):
                if selected.board[j, i] != node.board[j, i]:
                    return i
        return -1
    
    def select(self, node):
        """Node selection and expansion phase."""
        while self.fully_expanded(node):
            tmp = self.select_uct(node)
            if tmp == node:
                break
            node = tmp
        if node.terminal:
            return node
        node.add_child()
        if node.children:
            return self.pick_unvisited(node.children)
        return node
    
    def select_uct(self, node):
        """Select node with best UCT value."""
        best_uct = -np.inf
        best_node = None
        for child in node.children:
            uct = (child.q / child.n) + 2 * np.sqrt(np.log(node.n) / child.n)
            if uct > best_uct:
                best_uct = uct
                best_node = child
        if best_node is None:
            return node
        return best_node
    
    def fully_expanded(self, node):
        """Check whether a node is fully expanded."""
        visited = True
        if list(node.board[0]).count(0) == len(node.children):
            for child in node.children:
                if child.n == 0:
                    visited = False
            return visited
        return False
    
    def pick_unvisited(self, children):
        """Pick first unexplored child node."""
        for child in children:
            if child.n == 0:
                return child
        return None
    
    def rollout(self, node):
        """Perform a random game simulation."""
        board = node.board.copy()
        turn = node.turn
        if not node.terminal:
            while True:
                turn = PLAYER_PIECE if turn == AI_PIECE else AI_PIECE
                moves = self.get_moves(board, turn)
                if moves:
                    board = random.choice(moves)
                    terminal = self.result(board)
                    if terminal != 0:
                        return terminal
                else:
                    return self.result(board)
        else:
            return self.result(board)
    
    def get_moves(self, board, turn):
        """Get all possible next states."""
        moves = []
        for i in range(COLS):
            if board[0, i] == 0:
                for j in range(ROWS):
                    if board[j, i] == 0:
                        tmp = board.copy()
                        if turn == PLAYER_PIECE:
                            tmp[j, i] = AI_PIECE
                        else:
                            tmp[j, i] = PLAYER_PIECE
                        moves.append(tmp)
                        break
        return moves
    
    def result(self, board):
        """Get game result from terminal board."""
        winner = mcts_check_rows(board)
        if winner is not None:
            return winner
        winner = mcts_check_cols(board)
        if winner is not None:
            return winner
        winner = mcts_check_diag(board)
        if winner is not None:
            return winner
        return None
    
    def backpropagate(self, node, winner):
        """Update recursively node visits and scores from leaf to root."""
        if node.turn == winner:
            node.q += 1
        node.n += 1
        if node.parent is None:
            return
        self.backpropagate(node.parent, winner)
    
    def best_child(self, node):
        """Get child node with largest number of visits."""
        max_visit = 0
        best_node = None
        for child in node.children:
            if child.n > max_visit:
                max_visit = child.n
                best_node = child
        return best_node


# MCTS Helper Functions
def mcts_get_valid_locations(board):
    """Get valid column locations for MCTS."""
    valid_locations = []
    for column in range(COLS):
        if is_valid_location(board, column):
            valid_locations.append(column)
    return valid_locations

def mcts_check_win(board, piece):
    """Check for winning move (MCTS version)."""
    return winning_move(board, piece)

def mcts_check_tie(board):
    """Check if board is a tie."""
    return np.all(board != 0)

def mcts_check_rows(board):
    """Check for winner in rows."""
    for y in range(ROWS):
        row = list(board[y, :])
        for x in range(COLS - 3):
            if row[x:x + 4].count(PLAYER_PIECE) == 4:
                return PLAYER_PIECE
            if row[x:x + 4].count(AI_PIECE) == 4:
                return AI_PIECE
    return None

def mcts_check_cols(board):
    """Check for winner in columns."""
    for x in range(COLS):
        col = list(board[:, x])
        for y in range(ROWS - 3):
            if col[y:y + 4].count(PLAYER_PIECE) == 4:
                return PLAYER_PIECE
            if col[y:y + 4].count(AI_PIECE) == 4:
                return AI_PIECE
    return None

def mcts_check_diag(board):
    """Check for winner in diagonals."""
    # Right diagonal
    for point in [(3, 0), (4, 0), (3, 1), (5, 0), (4, 1), (3, 2), (5, 1), (4, 2), (3, 3), (5, 2), (4, 3), (5, 3)]:
        diag = []
        for k in range(4):
            diag.append(board[point[0] - k, point[1] + k])
        if diag.count(PLAYER_PIECE) == 4:
            return PLAYER_PIECE
        if diag.count(AI_PIECE) == 4:
            return AI_PIECE
    # Left diagonal
    for point in [(5, 3), (5, 4), (4, 3), (5, 5), (4, 4), (3, 3), (5, 6), (4, 5), (3, 4), (4, 6), (3, 5), (3, 6)]:
        diag = []
        for k in range(4):
            diag.append(board[point[0] - k, point[1] - k])
        if diag.count(PLAYER_PIECE) == 4:
            return PLAYER_PIECE
        if diag.count(AI_PIECE) == 4:
            return AI_PIECE
    return None


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
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE ))
            if board[r][c] == 0:
                pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE/2), int(r* SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)
            elif board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE/2), int(r* SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)
            else :
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE/2), int(r* SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), circle_radius)

    pygame.display.update()


def evaluate_window(window, piece):
    opponent_piece = PLAYER_PIECE

    if piece == PLAYER_PIECE:
        opponent_piece = AI_PIECE
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

# end montecarlo
def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizing_player):

    valid_locations = get_valid_locations(board)

    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal: 
            if winning_move(board, AI_PIECE):
                return (None, 10000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000)
            else:
                return (None, 0)
        else: 
            return (None, score_position(board, AI_PIECE))

    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_locations)

        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
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
            drop_piece(b_copy, row, col, PLAYER_PIECE)
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


def end_game():
    global game_over
    game_over = True
    print(game_over)


board = create_board()

game_over = False
not_over = True
turn = random.randint(PLAYER_TURN, AI_TURN)

# ==================== AI Selection ====================
print("=" * 50)
print("CONNECT 4 - BATTLE OF THE ALGORITHMS")
print("=" * 50)
print("Select AI opponent:")
print("1. Minimax (depth-based with alpha-beta pruning)")
print("2. Monte Carlo Tree Search (MCTS)")
print("=" * 50)

while True:
    try:
        ai_choice = int(input("Enter your choice (1 or 2): "))
        if ai_choice in [1, 2]:
            break
        print("Please enter 1 or 2.")
    except ValueError:
        print("Please enter 1 or 2.")

AI_TYPE = "minimax" if ai_choice == 1 else "mcts"
print(f"You selected: {'Minimax' if AI_TYPE == 'minimax' else 'MCTS'}")
print("=" * 50)

pygame.init()

SQUARESIZE = 100
width = COLS * SQUARESIZE
height = (ROWS + 1) * SQUARESIZE
circle_radius = int(SQUARESIZE/2 - 5)
size = (width, height)
screen = pygame.display.set_mode(size)

my_font = pygame.font.SysFont("monospace", 75)

draw_board(board)
pygame.display.update()


while not game_over:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION and not_over:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            xpos = pygame.mouse.get_pos()[0]
            if turn == PLAYER_TURN:
                pygame.draw.circle(screen, RED, (xpos, int(SQUARESIZE/2)), circle_radius )

        if event.type == pygame.MOUSEBUTTONDOWN and not_over:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))

            if turn == PLAYER_TURN:

                xpos = event.pos[0] 
                col = int(math.floor(xpos/SQUARESIZE)) 

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)
                    if winning_move(board, PLAYER_PIECE):
                        print("PLAYER 1 WINS!")
                        label = my_font.render("PLAYER 1 WINS!", 1, RED)
                        screen.blit(label, (40, 10))
                        not_over = False
                        t = Timer(3.0, end_game)
                        t.start()
                
                draw_board(board) 

                turn += 1
                turn = turn % 2 

        pygame.display.update()
      
    if turn == AI_TURN and not game_over and not_over:

        # Use selected AI algorithm
        if AI_TYPE == "minimax":
            col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
        else:  # MCTS
            root = MCTSNode(None, board, AI_PIECE)
            mcts = MCTS(AI_PIECE, 3.0)  # 3 second time budget
            col = mcts.compute_move(root)

        if is_valid_location(board, col):
            pygame.time.wait(500)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)
            if winning_move(board, AI_PIECE):
                print("PLAYER 2 WINS!")
                label = my_font.render("PLAYER 2 WINS!", 1, YELLOW)
                screen.blit(label, (40, 10))
                not_over = False
                t = Timer(3.0, end_game)
                t.start()
        draw_board(board)    

        turn += 1
        turn = turn % 2
