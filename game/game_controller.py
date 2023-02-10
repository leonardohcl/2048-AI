from .move_direction import MoveDirection
from .board import Board
from random import choice, choices
import re

SORTING_CONFIG = {
    MoveDirection.UP: ["row", "asc"],
    MoveDirection.DOWN: ["row", "desc"],
    MoveDirection.LEFT: ["col", "asc"],
    MoveDirection.RIGHT: ["col", "desc"],
}


class GameController:
    score: int
    can_move: list[MoveDirection]
    winning_block: int
    board: Board

    @property
    def width(self):
        return self.board.width

    @property
    def height(self):
        return self.board.height

    def _update_can_move(self, dir: MoveDirection):
        for sqr in self.board.squares:
            if sqr.value == 0:
                continue

            neighbor = self.board.get_square_neighbor(sqr.row, sqr.col, dir)
            if neighbor == None:
                continue

            if self.board.can_square_move(sqr, neighbor):
                self.can_move.append(dir)
                return

    def _update_valid_moves(self):
        self.can_move = []
        for dir in MoveDirection:
            self._update_can_move(dir)

    def _spawn(self, board: Board):
        empty = board.empty_squares()

        if len(empty) == 0:
            return

        selected = choice(empty)
        selected.value = choices([1, 2], weights=[0.1, 0.9], k=1)[0]

    def is_winner(self):
        return self.board.highest_block() >= self.winning_block

    def is_game_over(self):
        return len(self.can_move) == 0

    def _clear_board(self):
        self.board = Board(self.width, self.height)

    def load_preset(self, preset: list[int]):
        self.board.load_preset(preset)
        self._update_state()

    def _update_state(self):
        self._update_valid_moves()

    def start(self):
        self.score = 0
        self._clear_board()
        self._spawn(self.board)
        self._spawn(self.board)
        self._update_valid_moves()

    def get_next_board(self, dir: MoveDirection):
        next_board = Board(self.width, self.height)
        points = 0

        field, sorting_order = SORTING_CONFIG[dir]

        row_range = (
            range(self.height - 1, -1, -1)
            if field == "row" and sorting_order == "desc"
            else range(self.height)
        )
        col_range = (
            range(self.width - 1, -1, -1)
            if field == "col" and sorting_order == "desc"
            else range(self.width)
        )

        for row in row_range:
            for col in col_range:
                sqr = self.board.get_square(row, col)
                if sqr == None or sqr.value == 0:
                    continue

                next_board.move_square_value(sqr.row, sqr.col, sqr.value)
                move_sqr = next_board.get_square_valid_movement(sqr.row, sqr.col, dir)

                if move_sqr == None:
                    continue

                next_board.move_square_value(sqr.row, sqr.col, 0)
                points += next_board.move_square_value(
                    move_sqr.row, move_sqr.col, sqr.value
                )

        return next_board, points

    def move(self, dir: MoveDirection, spawn_after=True):
        if dir not in self.can_move:
            return

        next_board, points = self.get_next_board(dir)
        
        self.score += points

        if spawn_after:
            self._spawn(next_board)
        self.board = next_board
        self._update_state()

    def get_random_move(self):
        return choice(self.can_move)

    def __init__(self, width=4, height=4, winning_block=2048):
        self.winning_block = winning_block
        self.score = 0
        self.can_move = []
        self.board = Board(height, width)

    def __str__(self) -> str:
        return f"score:{self.score} | win: {self.is_winner()} | game over: {self.is_game_over()} \n {self.board}"