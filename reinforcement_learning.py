from game import GameController
from AI import Agent2048, get_state, get_state_size, get_model_name
import torch
import time

MAX_GAMES = 250
MAX_ERROR = 0.001
RANDOM_PROBABILITY = 0.8
RANDOM_PROBABILITY_DECAY = 0.999
LEARNING_RATE = 0.005
GAME_WIDTH = 5
GAME_HEIGHT = 3
WINNING_BLOCK = 11
INPUT_SIZE = get_state_size(GAME_WIDTH, GAME_HEIGHT)

LAYERS = [512,128]

GAME_LENGTH = GAME_WIDTH * GAME_HEIGHT


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

m = elapsed // 60
s = elapsed % 60

print(f"trained for {m:.0f} minutes and {s:.0f} seconds")
