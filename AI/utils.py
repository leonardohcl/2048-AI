from game import GameController, MoveDirection, Board
import torch
from matplotlib import pyplot as plt
import os
import numpy as np


def print_game(game: GameController, name: str = None):
    os.system("cls")
    if name != None:
        print(name)
    print(game)


def play_randomly(game: GameController, max_moves=0):
    moves = 0
    can_move = lambda moves: moves <= max_moves if max_moves > 0 else lambda moves: True
    while can_move(moves) and not game.is_winner() and not game.is_game_over():
        game.move(game.get_random_move())
        moves += 1


def output2move(output: torch.tensor, game: GameController):
    invalid_moves = list(
        filter(
            lambda x: x not in game.can_move,
            [
                MoveDirection.UP,
                MoveDirection.RIGHT,
                MoveDirection.LEFT,
                MoveDirection.DOWN,
            ],
        )
    )
    for move in invalid_moves:
        output[move.value] = 0.0
    idx = output.argmax()
    return MoveDirection(int(idx))


def plot_training(
    scores: list[int],
    mean_scores: list[float],
    errors: list[float],
    mean_errors: list[float],
    moves:list[float],
    mean_moves:list[float],
    highest_blocks: list[int],
    mean_highest_blocks: list[float]
):
    
    figure, axis = plt.subplots(2, 2, sharex=True)
    
    axis[0,0].set_title("Score Variation")
    axis[0,0].plot(scores)
    axis[0,0].plot(mean_scores, 'r:')
    axis[0,0].set_ylim(0)
    
    axis[0,1].set_title("Error Variation")
    axis[0,1].plot(errors)
    axis[0,1].plot(mean_errors, 'r:')
    axis[0,1].set_ylim(0,max(errors) * 1.1)
    
    axis[1,0].set_title("Moves made")
    axis[1,0].plot(moves)
    axis[1,0].plot(mean_moves, 'r:')
    axis[1,0].set_ylim(0)
    
    axis[1,1].set_title("Highest Block")
    axis[1,1].plot(highest_blocks)
    axis[1,1].plot(mean_highest_blocks, 'r:')
    axis[1,1].set_ylim(0,16)
     
    plt.show()        