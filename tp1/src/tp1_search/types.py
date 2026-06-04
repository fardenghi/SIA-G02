from enum import Enum
from typing import NamedTuple


class Position(NamedTuple):
    row: int
    col: int

    def __add__(self, other: tuple) -> "Position":  # type: ignore[override]
        return Position(self.row + other[0], self.col + other[1])


class Direction(Enum):
    """Direcciones de movimiento del jugador."""

    UP = Position(-1, 0)
    DOWN = Position(1, 0)
    LEFT = Position(0, -1)
    RIGHT = Position(0, 1)

    @property
    def delta(self) -> Position:
        return self.value
