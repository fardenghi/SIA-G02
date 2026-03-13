"""Heurísticas para búsqueda informada en Sokoban."""

from __future__ import annotations

import math
from typing import Callable, Sequence

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from scipy.optimize import linear_sum_assignment #hungarian heuristic


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


def hungarian_heuristic(board: Board, state: SokobanState) -> float:
    """Suma de distancias Manhattan usando asignación óptima (Algoritmo Húngaro).
    
    Empareja cada caja con un goal único de forma tal que la suma total
    de las distancias sea la mínima posible. Retorna infinito si detecta
    una caja en un dead square para podar la rama tempranamente.
    """
    boxes = state.boxes
    goals = board.goals
    
    # 1. Poda temprana: Si hay cajas en celdas muertas, el costo es infinito.
    for box in boxes:
        if board.is_dead_square(box):
            return math.inf
            
    # Casos borde (aunque en Sokoban estándar siempre hay cajas y goals)
    if not boxes or not goals:
        return 0.0

    # 2. Construir la matriz de costos
    # cost_matrix[i][j] será la distancia de la caja i al goal j
    cost_matrix = []
    for box in boxes:
        row_costs = []
        for goal in goals:
            # Calculamos la distancia Manhattan
            dist = abs(box.row - goal.row) + abs(box.col - goal.col)
            row_costs.append(dist)
        cost_matrix.append(row_costs)

    # 3. Aplicar el algoritmo Húngaro (Jonker-Volgenant)
    # row_ind contiene los índices de las cajas, col_ind los índices de los goals asignados
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Sumar los costos de la asignación óptima
    total_cost = 0.0
    for i in range(len(row_ind)):
        # Accedemos a la matriz original con los índices óptimos que nos devolvió scipy
        total_cost += cost_matrix[row_ind[i]][col_ind[i]]

    return total_cost


# ---------------------------------------------------------------------------
# Mapeo de nombres (usados en TOML) a funciones heurísticas
# ---------------------------------------------------------------------------
HEURISTICS: dict[str, HeuristicFn] = {
    "manhattan": manhattan_heuristic,
    "euclidean": euclidean_heuristic,
    "dead_square": dead_square_heuristic,
    "hungarian": hungarian_heuristic
}


def combine_heuristics_max(heuristics: Sequence[HeuristicFn]) -> HeuristicFn:
    """Combina múltiples heurísticas usando max(h1, ..., hn).

    Esta estrategia mantiene el lower bound más informativo de todas.
    """
    if not heuristics:
        raise ValueError("Se requiere al menos una heurística para combinar")

    if len(heuristics) == 1:
        return heuristics[0]

    def _combined(board: Board, state: SokobanState) -> float:
        return max(h(board, state) for h in heuristics)

    return _combined
