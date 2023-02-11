from game import GameController, MoveDirection
from AI import Agent2048
import torch
import time
import math
import numpy as np

MAX_GAMES = 100
MAX_ERROR = 0.001
RANDOM_PROBABILITY = 0.98
RANDOM_PROBABILITY_DECAY = 0.999
GAMES_EXPLORATION_PERCENT = 0.15
LEARNING_RATE = 0.005
GAME_WIDTH = 3
GAME_HEIGHT = 3
WINNING_BLOCK = 8

LAYERS = [512,128]

GAME_LENGTH = GAME_WIDTH * GAME_HEIGHT


def get_state(game: GameController):
    board = game.board.flat()

    available_moves = [MoveDirection(i) in game.can_move for i in range(4)]

    return board + available_moves


def get_reward(
    prev_state: list[int],
    next_state: list[int],
    points: int,
    blocks_moved: int,
    merges: int,
    win: bool,
    game_over: bool,
):
    prev_best = max(prev_state)
    next_best = max(next_state)
    empty_square_bonus = sum([1 if x == 0 else -1 for x in next_state])/GAME_LENGTH if win else 0
    rewards = [
        1 if win else 0,
        empty_square_bonus,
        next_best if next_best > prev_best else 0,
        float(merges) / (0.5 * GAME_LENGTH),
        # float(-blocks_moved)/GAME_LENGTH
        -1 if game_over else 0,
    ]
    return sum(rewards)


game = GameController(GAME_WIDTH, GAME_HEIGHT, WINNING_BLOCK)
game.start()
INPUT_SIZE = len(get_state(game))

agent = Agent2048(
    GameController(GAME_WIDTH, GAME_HEIGHT, WINNING_BLOCK),
    get_state=get_state,
    get_reward=get_reward,
    state_size=INPUT_SIZE,
    brain_structure=LAYERS,
    learning_rate=LEARNING_RATE,
    gamma=0.9,
    optmizer=torch.optim.SGD,
    criterion=torch.nn.MSELoss,
    epsillon=RANDOM_PROBABILITY,
    epsillon_decay=RANDOM_PROBABILITY_DECAY
)

start = time.time()
agent.train(max_games=MAX_GAMES, max_error=MAX_ERROR)
elapsed = time.time() - start

agent.model.save()
m = elapsed // 60
s = elapsed % 60

print(f"trained for {m:.0f} minutes and {s:.0f} seconds")
