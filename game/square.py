VALUE_MAP = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


class Square:
    value: int
    row: int
    col: int
    merged: bool
    
    @property
    def is_wall(self):
        return self.value < 0

    @property
    def base_2(self):
        return -1 if self.is_wall else VALUE_MAP[self.value]

    def __init__(self, row, col, value=0, merged=False):
        self.row = row
        self.col = col
        self.value = value
        self.merged = merged

    def __str__(self) -> str:
        return self.base_2
