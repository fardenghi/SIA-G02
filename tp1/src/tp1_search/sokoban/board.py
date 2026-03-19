from dataclasses import dataclass

import numpy as np

from tp1_search.types import Position


@dataclass(frozen=True)
class Board:
    rows: int
    cols: int
    walls: frozenset[Position]
    goals: frozenset[Position]

    def __post_init__(self) -> None:
        # Construir arrays numpy para lookups O(1)
        walls_np = np.zeros((self.rows, self.cols), dtype=bool)
        goals_np = np.zeros((self.rows, self.cols), dtype=bool)

        for pos in self.walls:
            walls_np[pos.row, pos.col] = True
        for pos in self.goals:
            goals_np[pos.row, pos.col] = True

        # frozen=True → usar object.__setattr__ para asignar atributos calculados
        object.__setattr__(self, "_walls_np", walls_np)
        object.__setattr__(self, "_goals_np", goals_np)

    # ------------------------------------------------------------------
    # Lookups usando numpy (O(1) array indexing)
    # ------------------------------------------------------------------

    def is_wall(self, pos: Position) -> bool:
        return bool(self._walls_np[pos.row, pos.col])  # type: ignore[attr-defined]

    def is_goal(self, pos: Position) -> bool:
        return bool(self._goals_np[pos.row, pos.col])  # type: ignore[attr-defined]

    def in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.row < self.rows and 0 <= pos.col < self.cols

    def is_free(self, pos: Position) -> bool:
        """True si la posición está dentro del tablero y no es pared."""
        return self.in_bounds(pos) and not self.is_wall(pos)

    def _is_blocked(self, pos: Position) -> bool:
        return not self.in_bounds(pos) or self.is_wall(pos)

    def is_dead_square(self, pos: Position) -> bool:
        """True si la celda es una esquina no objetivo formada por paredes/bordes."""
        if not self.in_bounds(pos) or self.is_wall(pos) or self.is_goal(pos):
            return False

        up = self._is_blocked(Position(pos.row - 1, pos.col))
        down = self._is_blocked(Position(pos.row + 1, pos.col))
        left = self._is_blocked(Position(pos.row, pos.col - 1))
        right = self._is_blocked(Position(pos.row, pos.col + 1))

        return (up and left) or (up and right) or (down and left) or (down and right)
