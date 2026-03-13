"""Heurísticas para búsqueda informada en Sokoban."""

from __future__ import annotations

import math
from typing import Callable

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState


# ---------------------------------------------------------------------------
# Tipo de una función heurística: (Board, SokobanState) → float
# ---------------------------------------------------------------------------
HeuristicFn = Callable[[Board, SokobanState], float]


def dead_square_heuristic(board: Board, state: SokobanState) -> float:
    """Devuelve inf si alguna caja está en una celda muerta, 0 en caso contrario.

    Una celda es "muerta" si una caja ubicada ahí nunca puede llegar a ningún
    goal (precomputado por backward BFS en Board.__post_init__).

    Es admisible: si el estado es irresolvible el costo real también es inf;
    si no, devolver 0 nunca sobreestima.
    """
    for box in state.boxes:
        if board.is_dead_square(box):
            return math.inf
    return 0.0


def manhattan_heuristic(board: Board, state: SokobanState) -> float:
    """Suma de distancias Manhattan de cada caja al goal más cercano.

    Para cada caja, calcula la distancia Manhattan (|dr| + |dc|) al goal
    más cercano y suma todas. Es admisible porque cada caja debe moverse
    al menos esa cantidad de pasos para llegar a algún goal, y la suma
    de mínimos es un lower bound del costo total real.
    """
    total = 0.0
    goals = board.goals
    for box in state.boxes:
        min_dist = min(abs(box.row - g.row) + abs(box.col - g.col) for g in goals)
        total += min_dist
    return total


def euclidean_heuristic(board: Board, state: SokobanState) -> float:
    """Suma de distancias euclidianas de cada caja al goal más cercano.

    Para cada caja, calcula la distancia euclidiana al goal más cercano
    y suma todas. Es admisible porque la distancia euclidiana es siempre
    <= la distancia Manhattan, que ya es un lower bound.
    """
    total = 0.0
    goals = board.goals
    for box in state.boxes:
        min_dist = min(
            math.sqrt((box.row - g.row) ** 2 + (box.col - g.col) ** 2) for g in goals
        )
        total += min_dist
    return total


# ---------------------------------------------------------------------------
# Mapeo de nombres (usados en TOML) a funciones heurísticas
# ---------------------------------------------------------------------------
HEURISTICS: dict[str, HeuristicFn] = {
    "manhattan": manhattan_heuristic,
    "euclidean": euclidean_heuristic,
    "dead_square": dead_square_heuristic,
}
