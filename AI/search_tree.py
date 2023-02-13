from game import GameController, MoveDirection, play_randomly, print_game
from joblib import Parallel, delayed
import math
import multiprocessing


class MonteCarloSearchTree:
    @staticmethod
    def play(
        game: GameController,
        num_simulations=0,
        look_ahead=10,
        print_gameplay=False,
    ):
        output_gameplay = print_game if print_gameplay else lambda game: None

        game.start()
        moves = 0
        while not game.is_game_over() and not game.is_winner():
            output_gameplay(game)
            next_move = MonteCarloSearchTree.search_next_move(
                game, num_simulations, look_ahead
            )
            game.move(next_move)
            moves += 1
        output_gameplay(game)

    @staticmethod
    def search_next_move(
        game: GameController,
        num_simulations: int,
        look_ahead: int,
    ):
        possible_moves = game.can_move
        dir_simulations = math.ceil(num_simulations / len(possible_moves))
        best_dir = None
        best_score = 0

        for dir in possible_moves:
            num_cores = multiprocessing.cpu_count()
            scores = Parallel(n_jobs=num_cores)(
                delayed(MonteCarloSearchTree.play_simulation)(
                    game,
                    dir,
                    look_ahead,
                )
                for i in range(dir_simulations)
            )
            dir_score = sum(scores) / len(scores)
            if dir_score > best_score:
                best_score = dir_score
                best_dir = dir

        return best_dir

    @staticmethod
    def play_simulation(
        game: GameController,
        starting_direction: MoveDirection,
        look_ahead: int,
    ):
        scenario = GameController(game.width, game.height, game.winning_block)
        scenario.load_preset(game.board.flat())
        scenario.move(starting_direction)
        play_randomly(scenario, look_ahead)

        return scenario.score
