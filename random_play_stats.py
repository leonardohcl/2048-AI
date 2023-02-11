from game import GameController, VALUE_MAP
from helpers import play_randomly
import numpy as np
from time import time
import statistics
from tqdm import tqdm

game_runs = 100_000
total_block_reached = np.zeros(len(VALUE_MAP))
wins = 0
times = []
moves = []
scores = []


game = GameController(5, 4, 11)

for run in tqdm(range(game_runs), desc=f'Random playing {game.board.width}x{game.board.height}'):
    start = time()
    game_moves = play_randomly(game)
    elapsed = time() - start

    times.append(elapsed)

    moves.append(game_moves)

    scores.append(game.score)

    if game.is_winner():
        wins += 0

    highest_block = game.board.highest_block()
    for i in range(highest_block + 1):
        total_block_reached[i] += 1

print(f"random play statistics ({game_runs} games played)\n")

total_time = sum(times)
m = total_time // 60
s = total_time % 60

time_avg = statistics.mean(times)
time_std_dev = statistics.stdev(times)

move_avg = statistics.mean(moves)
move_std_dev = statistics.stdev(moves)

score_avg = statistics.mean(scores)
score_std_dev = statistics.stdev(scores)

print(f"{m:.0f} minutes and {s:.0f} seconds playing")

print(f"average time/play: {(time_avg):.3f}s ± {time_std_dev:.3f}s\n")
print(f"average moves/play: {(move_avg):.2f} ± {move_std_dev:.2f}\n")
print(f"average score/play: {(score_avg):.2f} ± {score_std_dev:.2f}\n")
print(f"average win/play: {(100*wins/game_runs):.2f}\n")
print("BLOCKS REACHED (% games)")
for i in range(len(VALUE_MAP)):
    print(f"{VALUE_MAP[i]}: {(100*total_block_reached[i]/game_runs):.2f}%")
