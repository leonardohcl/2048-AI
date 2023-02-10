from game import GameController, MoveDirection, Board
from AI import MonteCarloSearchTree
from time import time
import math

game = GameController(4, 4, 11)

start = time()
MonteCarloSearchTree.play(game, num_simulations=10, look_ahead=4, print_gameplay=True)
total = time() - start
minutes = math.floor(total / 60)
secs = total % 60
print(f"play duration {minutes} minutes and {secs:.2f} seconds")