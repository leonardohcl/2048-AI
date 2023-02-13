from game import GameController, MoveDirection, print_game
import torch
from matplotlib import pyplot as plt


def output2move(output: torch.tensor, game: GameController):
    invalid_moves = list(
        filter(
            lambda x: x not in game.can_move,
            [
                MoveDirection.UP,
                MoveDirection.RIGHT,
                MoveDirection.LEFT,
                MoveDirection.DOWN,
            ],
        )
    )
    for move in invalid_moves:
        output[move.value] = 0.0
    idx = output.argmax()
    return MoveDirection(int(idx))


def plot_training(
    scores: list[int],
    mean_scores: list[float],
    errors: list[float],
    mean_errors: list[float],
    moves: list[float],
    mean_moves: list[float],
    highest_blocks: list[int],
    mean_highest_blocks: list[float],
):
    figure, axis = plt.subplots(2, 2, sharex=True)

    axis[0, 0].set_title("Score Variation")
    axis[0, 0].plot(scores)
    axis[0, 0].plot(mean_scores, "r:")
    axis[0, 0].set_ylim(0)

    axis[0, 1].set_title("Error Variation")
    axis[0, 1].plot(errors)
    axis[0, 1].plot(mean_errors, "r:")
    axis[0, 1].set_ylim(0, max(errors) * 1.1)

    axis[1, 0].set_title("Moves made")
    axis[1, 0].plot(moves)
    axis[1, 0].plot(mean_moves, "r:")
    axis[1, 0].set_ylim(0)

    axis[1, 1].set_title("Highest Block")
    axis[1, 1].plot(highest_blocks)
    axis[1, 1].plot(mean_highest_blocks, "r:")
    axis[1, 1].set_ylim(0, 16)

    plt.show()


def get_state(game: GameController):
    board = game.board.flat()

    available_moves = [MoveDirection(i) in game.can_move for i in range(4)]

    return board + available_moves


def get_state_size(width, height):
    game = GameController(width, height)
    return len(get_state(game))


def play_with_model(
    model,
    game: GameController,
    max_moves=0,
    print_play=False,
    play_title: str = None,
):
    moves = 0
    show_game = print_game if print_play else lambda game, name: None
    can_move = lambda moves: moves <= max_moves if max_moves > 0 else lambda moves: True
    if game.is_winner() or game.is_game_over():
        game.start()
        show_game(game, play_title)

    while can_move(moves) and not game.is_winner() and not game.is_game_over():
        state = get_state(game)
        move = model.predict(state, game)
        game.move(move)
        moves += 1
        show_game(game, play_title)

    highest_block = game.board.highest_block()
    return moves, game.score, highest_block

def get_model_name(width, height, hidden_layers:list):
    return f"{width}x{height}_{'_'.join([str(x) for x in hidden_layers])}"
