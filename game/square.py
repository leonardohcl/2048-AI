VALUE_MAP = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


class Square:
    value = 0
    row = 0
    col = 0
    merged = False
    
    @property
    def base_2(self):
        return VALUE_MAP[self.value]

    def __init__(self, row, col, value=0):
        self.row = row
        self.col = col
        self.value = value
        
    def __str__(self) -> str:
        return str(self.base_2)
