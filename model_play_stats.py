from game import GameController, VALUE_MAP
from AI import Linear_2048Qnet, play_with_model, get_state_size, get_model_name
import numpy as np
import torch
import statistics
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

total_block_reached = np.zeros(len(VALUE_MAP))
wins = 0
times = []
moves = []
scores = []


GAME_RUNS = 10_000
GAME_WIDTH = 4
GAME_HEIGHT = 4
WINNING_BLOCK = 12

MODEL_HIDDEN_LAYERS = [512,128]
STATE_SIZE= get_state_size(GAME_WIDTH,GAME_HEIGHT)
MODEL = Linear_2048Qnet(input_size=STATE_SIZE, hidden_layers=MODEL_HIDDEN_LAYERS)
MODEL.load_state_dict(torch.load(f'model/{get_model_name(GAME_WIDTH,GAME_HEIGHT,MODEL_HIDDEN_LAYERS)}'))

num_cores = multiprocessing.cpu_count()
output = Parallel(n_jobs=num_cores)(
    delayed(play_with_model)(MODEL, GameController(GAME_WIDTH, GAME_HEIGHT, WINNING_BLOCK))
    for _ in tqdm(range(GAME_RUNS), desc=f"playing {GAME_WIDTH}x{GAME_HEIGHT}")
)

moves, scores, highest_blocks = zip(*output)

for block in highest_blocks:
    for i in range(block + 1):
        total_block_reached[i] += 1

print(f"model play statistics ({GAME_RUNS} games played)\n")


move_avg = statistics.mean(moves)
move_std_dev = statistics.stdev(moves)

score_avg = statistics.mean(scores)
score_std_dev = statistics.stdev(scores)


print(f"average moves/play: {(move_avg):.2f} ± {move_std_dev:.2f}\n")
print(f"average score/play: {(score_avg):.2f} ± {score_std_dev:.2f}\n")

print("BLOCKS REACHED (% games)")
for i in range(len(VALUE_MAP)):
    print(f"{VALUE_MAP[i]}: {(100*total_block_reached[i]/GAME_RUNS):.2f}%")
