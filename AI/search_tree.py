from game import GameController
from joblib import Parallel, delayed
import math
import multiprocessing
import os


class MonteCarloSearchTree:
    @staticmethod
    def play(game: GameController, num_simulations: int, look_ahead = 100, print_gameplay = False):
        
        if(print_gameplay):
            def output_gameplay():
                os.system('cls')
                print(game)
        else:
            def output_gameplay(): None
        game.start()
        moves = 0
        while not game.is_game_over() and not game.is_winner():
            output_gameplay()
            next_move = MonteCarloSearchTree.get_next_move(game, num_simulations, look_ahead)
            game.move(next_move)
            moves+=1
        output_gameplay()

    @staticmethod
    def get_next_move(game: GameController, num_simulations: int, look_ahead = 100):
        possible_moves = game.can_move
        dir_simulations = math.ceil(num_simulations / len(possible_moves))
        best_dir = None
        best_score = 0

        for dir in possible_moves:
            def play_simulation():
                scenario = GameController(game.width, game.height, game.winning_block)
                scenario.load_preset(game.board.flat())
                scenario.move(dir)
                moves = 0
                while (
                    moves < look_ahead
                    and not scenario.is_winner()
                    and not scenario.is_game_over()
                ):
                    scenario.move(scenario.get_random_move())
                    moves += 1
                return scenario.score

            num_cores = multiprocessing.cpu_count()
            scores = Parallel(n_jobs=num_cores)(
                delayed(play_simulation)() for i in range(dir_simulations)
            )
            dir_score = sum(scores) / len(scores)
            if dir_score > best_score:
                best_score = dir_score
                best_dir = dir

        return best_dir
