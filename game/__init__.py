from .game_controller import GameController
from .square import Square, VALUE_MAP
from .board import Board
from .move_direction import MoveDirection
import os

def print_game(game: GameController, name: str = None):
    os.system("cls")
    if name != None:
        print(name)
    print(game)


def play_randomly(
    game: GameController, max_moves=0, print_play=False, play_title: str = None
):
    moves = 0
    show_game = print_game if print_play else lambda game, name: None
    can_move = lambda moves: moves <= max_moves if max_moves > 0 else lambda moves: True
    if game.is_winner() or game.is_game_over():
        game.start()
        show_game(game, play_title)

    while can_move(moves) and not game.is_winner() and not game.is_game_over():
        game.move(game.get_random_move())
        moves += 1
        show_game(game, play_title)

    highest_block = game.board.highest_block()
    return moves, game.score, highest_block