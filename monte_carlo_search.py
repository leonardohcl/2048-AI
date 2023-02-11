from game import GameController, MoveDirection, Board
from AI import MonteCarloSearchTree
from time import time

game = GameController(4, 4, 11)

start = time()
MonteCarloSearchTree.play(game, num_simulations=200, look_ahead=0, print_gameplay=True)
total = time() - start
minutes = total // 60
secs = total % 60
print(f"play duration {minutes:.0f} minutes and {secs:.0f} seconds")
