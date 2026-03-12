"""Heurísticas para búsqueda informada en Sokoban."""

from __future__ import annotations

import math

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState


# ---------------------------------------------------------------------------
# Tipo de una función heurística: (Board, SokobanState) → float
# ---------------------------------------------------------------------------
HeuristicFn = "Callable[[Board, SokobanState], float]"  # solo documentación


def dead_square_heuristic(board: Board, state: SokobanState) -> float:
    """Devuelve inf si alguna caja está en una celda muerta, 0 en caso contrario.

    Una celda es "muerta" si una caja ubicada ahí nunca puede llegar a ningún
    goal (precomputado por backward BFS en Board.__post_init__).

    Es admisible: si el estado es irresolvible el costo real también es inf;
    si no, devolver 0 nunca sobreestima.

    Para métodos informados (Greedy, A*) esta heurística reemplaza la poda
    por dead squares que se usa en métodos no informados (BFS, DFS).
    Se puede combinar (sumar) con otras heurísticas de distancia.
    """
    for box in state.boxes:
        if board.is_dead_square(box):
            return math.inf
    return 0.0
