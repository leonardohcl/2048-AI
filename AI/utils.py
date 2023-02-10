from game import GameController


def play_randomly(game: GameController, max_moves=0):
    moves = 0
    can_move = (
        lambda moves: moves <= max_moves if max_moves > 0 else lambda moves: True
    )
    while (
        can_move(moves) and not game.is_winner() and not game.is_game_over()
    ):
        game.move(game.get_random_move())
        moves += 1    
    