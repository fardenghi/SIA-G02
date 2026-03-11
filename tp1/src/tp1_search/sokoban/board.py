from collections import deque
from dataclasses import dataclass

import numpy as np

from tp1_search.types import Position

# @TODO hablar en la presentación de esto
def _compute_dead_squares(
    rows: int,
    cols: int,
    walls_np: "np.ndarray",
    goals_np: "np.ndarray",
) -> "np.ndarray":
    """Backward BFS desde los goals para identificar celdas dead.

    Una celda es "live" si una caja ubicada ahí puede eventualmente llegar a
    algún goal.  Hacemos BFS hacia atrás: goal → origen del empuje.

    Para que una caja pueda moverse de S → T (en dirección dr,dc):
      - T no es pared, S no es pared
      - La posición del jugador al empujar es P = S - (dr,dc), que tampoco puede ser pared
    """
    live = goals_np.copy().astype(bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if goals_np[r, c]:
                q.append((r, c))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        tr, tc = q.popleft()
        for dr, dc in dirs:
            # La caja estaba en S antes de ser empujada a (tr, tc)
            sr, sc = tr - dr, tc - dc
            # El jugador estaba en P para empujar
            pr, pc = sr - dr, sc - dc

            if (
                0 <= sr < rows
                and 0 <= sc < cols
                and not walls_np[sr, sc]
                and 0 <= pr < rows
                and 0 <= pc < cols
                and not walls_np[pr, pc]
                and not live[sr, sc]
            ):
                live[sr, sc] = True
                q.append((sr, sc))

    # dead = celda que no es live ni pared
    return ~live & ~walls_np


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

        dead_sq = _compute_dead_squares(self.rows, self.cols, walls_np, goals_np)

        # frozen=True → usar object.__setattr__ para asignar atributos calculados
        object.__setattr__(self, "_walls_np", walls_np)
        object.__setattr__(self, "_goals_np", goals_np)
        object.__setattr__(self, "_dead_sq", dead_sq)

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

    def is_dead_square(self, pos: Position) -> bool:
        """True si una caja en esta posición nunca puede llegar a ningún goal."""
        return bool(self._dead_sq[pos.row, pos.col])  # type: ignore[attr-defined]
