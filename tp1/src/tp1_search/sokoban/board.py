from dataclasses import dataclass

from tp1_search.types import Position


@dataclass(frozen=True)
class Board:
    rows: int
    cols: int
    walls: frozenset[Position]
    goals: frozenset[Position]

    def is_wall(self, pos: Position) -> bool:
        return pos in self.walls

    def is_goal(self, pos: Position) -> bool:
        return pos in self.goals

    def in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.row < self.rows and 0 <= pos.col < self.cols

    def is_free(self, pos: Position) -> bool:
        """True si la posición está dentro del tablero y no es pared."""
        return self.in_bounds(pos) and not self.is_wall(pos)
