from game import GameController, MoveDirection
from AI import Agent2048
from utils import normalize_board
import torch
import time
import math
import numpy as np

MAX_GAMES = 1000
MAX_ERROR = 0.001
GAMES_EXPLORATION_PERCENT = 0.3
RANDOM_MOVE_PROBABILITY = 0.4
LEARNING_RATE = 0.01
GAME_WIDTH = 4
GAME_HEIGHT = 4
WINNING_BLOCK = 11

LAYERS = [256, 32]

GAME_LENGTH = GAME_WIDTH * GAME_HEIGHT


def get_state(game: GameController):
    board = normalize_board(game.board.flat(), WINNING_BLOCK)

    future_sight = []
    for i in range(4):
        move = MoveDirection(i)
        if move in game.can_move:
            (
                future_board,
                future_points,
                future_blocks_moved,
                future_merges,
            ) = game.get_next_board(move)

            future_flat_board = normalize_board(future_board.flat(), WINNING_BLOCK)

            future_sight += future_flat_board + [1]
        else:
            future_sight += list(np.zeros(GAME_LENGTH + 1))

    return board + future_sight


def get_reward(
    prev_state: list[int],
    next_state: list[int],
    points: int,
    blocks_moved: int,
    merges: int,
    win: bool,
    game_over: bool,
):
    previous_best = max(prev_state)
    next_best = max(next_state)
    highest_increase_points = (
        next_state / GAME_LENGTH if next_best > previous_best else 0
    )
    merge_points = float(merges) / (0.5 * GAME_LENGTH)
    game_over_penalty = -1 if game_over else 0
    win_reward = 1 if win else 0

    return highest_increase_points + merge_points + win_reward + game_over_penalty


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
