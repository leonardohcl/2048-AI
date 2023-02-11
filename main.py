from game import GameController, MoveDirection
from AI import Agent2048
from utils import normalize_board, get_zig_zag
import torch
import time
import math
import numpy as np

MAX_GAMES = 50
MAX_ERROR = 0.001
GAMES_EXPLORATION_PERCENT = 0.15
RANDOM_MOVE_PROBABILITY = 0.3
LEARNING_RATE = 0.005
GAME_WIDTH = 4
GAME_HEIGHT = 4
WINNING_BLOCK = 11

LAYERS = [32]

GAME_LENGTH = GAME_WIDTH * GAME_HEIGHT


def get_state(game: GameController):
    board = normalize_board(game.board.flat(), WINNING_BLOCK)

    available_moves = [MoveDirection(i) in game.can_move for i in range(4)]

    return board + available_moves


TARGET = normalize_board(get_zig_zag(GAME_WIDTH, GAME_HEIGHT))


def get_reward(
    prev_state: list[int],
    next_state: list[int],
    points: int,
    blocks_moved: int,
    merges: int,
    win: bool,
    game_over: bool,
):
    board = normalize_board(next_state, WINNING_BLOCK)
    diff = [
        1
        if board[i] == TARGET[i]
        else TARGET[i]
        if board[i] > TARGET[i]
        else -TARGET[i]
        for i in range(GAME_LENGTH)
    ]
    return sum(diff)


game = GameController()
game.start()
INPUT_SIZE = len(get_state(game))

agent = Agent2048(
    GameController(GAME_WIDTH, GAME_HEIGHT, WINNING_BLOCK),
    get_state=get_state,
    get_reward=get_reward,
    state_size=INPUT_SIZE,
    brain_structure=LAYERS,
    learning_rate=LEARNING_RATE,
    gamma=0.85,
    optmizer=torch.optim.SGD,
    criterion=torch.nn.MSELoss,
    exploration_heavy_epochs=math.floor(MAX_GAMES * GAMES_EXPLORATION_PERCENT),
    random_exploration_probability=RANDOM_MOVE_PROBABILITY,
    batch_size=1000,
)

start = time.time()
agent.train(max_games=MAX_GAMES, max_error=MAX_ERROR)
elapsed = time.time() - start

m = elapsed // 60
s = elapsed % 60

print(f"trained for {m:.0f} minutes and {s:.0f} seconds")
