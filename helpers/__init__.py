from game import GameController, Board
import numpy as np
import math
import os


def get_zig_zag(width: int, height: int, starting_value: int = None):
    board = Board(width, height)
    value = board.length if starting_value == None else starting_value
    x = 0
    y = 0
    x_increment = 1
    while value > 0:
        idx = board.coord2idx(y, x)
        board.squares[idx].value = value
        value -= 1
        x += x_increment
        if x == width or x < 0:
            y += 1
            x_increment *= -1
            x += x_increment
    return board.flat()


def normalize_board(board: list[int], base_value: float = None):
    base = max(board) if base_value == None else base_value
    return [float(x) / base for x in board]


def center_of_mass(board: list[float], width: int, height: int):
    matrix = np.reshape(board, (height, width))
    rows = height
    cols = width
    total_mass = 0
    x_mass = 0
    y_mass = 0
    for i in range(rows):
        for j in range(cols):
            total_mass += matrix[i][j]
            x_mass += j * matrix[i][j]
            y_mass += i * matrix[i][j]
    x_center = x_mass / total_mass
    y_center = y_mass / total_mass
    return (x_center / width, y_center / height)


distance_to_top_left = lambda x, y: math.sqrt((0 - x) ** 2 + (0 - y) ** 2)
distance_to_top_right = lambda x, y: math.sqrt((0 - x) ** 2 + (1 - y) ** 2)
distance_to_bottom_left = lambda x, y: math.sqrt((1 - x) ** 2 + (0 - y) ** 2)
distance_to_bottom_right = lambda x, y: math.sqrt((1 - x) ** 2 + (1 - y) ** 2)


def distance_to_closest_corner(x, y):
    closest_distance = min(
        distance_to_top_left(x, y),
        distance_to_top_right(x, y),
        distance_to_bottom_left(x, y),
        distance_to_bottom_right(x, y),
    )
    return closest_distance / math.sqrt(2)


def get_variations(board: list[int], width: int, height: int):
    matrix = np.reshape(board, (height, width))
    rows = height
    columns = width
    horizontal_variation = 0
    vertical_variation = 0
    for i in range(rows):
        for j in range(columns - 1):
            horizontal_variation += abs(matrix[i][j] - matrix[i][j + 1])
    for j in range(columns):
        for i in range(rows - 1):
            vertical_variation += abs(matrix[i][j] - matrix[i + 1][j])
    return horizontal_variation, vertical_variation


def play_randomly(game: GameController, max_moves=0, print_play= False, play_title:str=None):
    moves = 0
    show_game = print_game if print_play else lambda game, name:None
    can_move = lambda moves: moves <= max_moves if max_moves > 0 else lambda moves: True
    if game.is_winner() or game.is_game_over():
        game.start()
        show_game(game, play_title)
        
    while can_move(moves) and not game.is_winner() and not game.is_game_over():
        game.move(game.get_random_move())
        moves += 1
        show_game(game, play_title)
    
    return moves

def print_game(game: GameController, name: str = None):
    os.system("cls")
    if name != None:
        print(name)
    print(game)