import numpy as np
import math
from .square import Square
from .move_direction import MoveDirection


class Board:
    width = 0
    height = 0
    squares: list[Square] = []

    @property
    def length(self):
        return self.width * self.height

    def empty_squares(self):
        return [sqr for sqr in self.squares if sqr.value == 0]

    def filled_squares(self):
        return [sqr for sqr in self.squares if sqr.value > 0]

    def flat(self):
        return [sqr.value for sqr in self.squares]

    def highest_block(self):
        return max(self.flat())

    def load_preset(self, preset: list[int]):
        for idx in range(len(preset)):
            self.squares[idx].value = preset[idx]

    def _is_valid_coords(self, row: int, col: int):
        if row < 0 or col < 0:
            return False
        if row >= self.height:
            return False
        if col >= self.width:
            return False
        return True

    def _is_valid_idx(self, idx: int):
        return idx >= 0 and idx < self.length

    def idx2coord(self, idx: int):
        if not self._is_valid_idx(idx):
            return -1, -1
        row = math.floor(idx / self.width) % self.height
        col = idx % self.width
        return row, col

    def coord2idx(self, row: int, col: int):
        if not self._is_valid_coords(row, col):
            return -1
        row_component = row * self.width
        col_component = col % self.width
        return row_component + col_component

    def get_square(self, row: int, col: int):
        idx = self.coord2idx(row, col)
        if idx < 0:
            return None
        return self.squares[idx]

    def get_square_neighbor(self, row: int, col: int, dir: MoveDirection):
        if not self._is_valid_coords(row, col):
            return None
        if dir == MoveDirection.UP:
            return self.get_square(row - 1, col)
        if dir == MoveDirection.DOWN:
            return self.get_square(row + 1, col)
        if dir == MoveDirection.LEFT:
            return self.get_square(row, col - 1)
        if dir == MoveDirection.RIGHT:
            return self.get_square(row, col + 1)

    def _can_squares_merge(self, sqr: Square, target: Square):
        return not target.merged and sqr.value == target.value

    def can_square_move(self, sqr: Square, target: Square):
        if target.value == 0:
            return True
        return self._can_squares_merge(sqr, target)

    def get_square_valid_movement(self, row: int, col: int, dir: MoveDirection):
        sqr = self.get_square(row, col)
        if sqr == None or sqr.value == 0 or sqr.is_wall:
            return None

        neighbor = self.get_square_neighbor(row, col, dir)
        selectedNeighbor = None

        while neighbor != None:
            if self.can_square_move(sqr, neighbor):
                selectedNeighbor = neighbor
            else:
                break
                
            neighbor = self.get_square_neighbor(neighbor.row, neighbor.col, dir)

        return selectedNeighbor

    def move_square_value(self, row: int, col: int, value: int):
        sqr = self.get_square(row, col)
        if sqr == None:
            return 0
        if sqr.value == value:
            sqr.value += 1
            sqr.merged = True
            return sqr.value
        sqr.value = value
        return 0

    def __init__(self, width, height, preset: list[int] = None) -> None:
        self.width = width
        self.height = height
        self.squares = [Square(0, 0, 0) for i in range(width * height)]
        for idx, sqr in enumerate(self.squares):
            row, col = self.idx2coord(idx)
            sqr.row = row
            sqr.col = col

        if preset != None:
            self.load_preset(preset)

    def __str__(self) -> str:
        arr = np.array([sqr.base_2 for sqr in self.squares])
        return (
            str(arr.reshape((self.height, self.width)))
            .replace("-1", "x")
            .replace("[", "")
            .replace("]", "")
        )
