from game import GameController, MoveDirection, Board
from AI import MonteCarloSearchTree
from time import time
import math

game = GameController(4, 4, 11)

start = time()
MonteCarloSearchTree.play(game, 100, 200, print_gameplay=True)
total = time() - start
minutes = math.floor(total / 60)
secs = total % 60
