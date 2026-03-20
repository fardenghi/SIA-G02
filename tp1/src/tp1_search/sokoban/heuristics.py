"""Heurísticas para búsqueda informada en Sokoban."""

from __future__ import annotations

import math
from typing import Callable, Sequence

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from scipy.optimize import linear_sum_assignment  # hungarian heuristic


# ---------------------------------------------------------------------------
# Tipo de una función heurística: (Board, SokobanState) → float
# ---------------------------------------------------------------------------
HeuristicFn = Callable[[Board, SokobanState], float]

WEIGHTED_HUNGARIAN_FACTOR = 1.5
PLAYER_DISTANCE_FACTOR = 0.5


def _manhattan_distance(a, b) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def _hungarian_assignment_cost(
    boxes: Sequence,
    goals: Sequence,
) -> float:
    if not boxes or not goals:
        return 0.0

    cost_matrix = []
    for box in boxes:
        row_costs = []
        for goal in goals:
            row_costs.append(_manhattan_distance(box, goal))
        cost_matrix.append(row_costs)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return float(sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind, strict=False)))


def _player_to_nearest_unsolved_box(board: Board, state: SokobanState) -> float:
    unsolved_boxes = [box for box in state.boxes if not board.is_goal(box)]
    if not unsolved_boxes:
        return 0.0
    return float(min(_manhattan_distance(state.player, box) for box in unsolved_boxes))


def dead_square_heuristic(board: Board, state: SokobanState) -> float:
    """Devuelve inf si alguna caja no-goal está atrapada en una esquina.


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
    de las distancias sea la mínima posible.
    """

    return _hungarian_assignment_cost(tuple(state.boxes), tuple(board.goals))


def weighted_hungarian_heuristic(board: Board, state: SokobanState) -> float:
    """Heurística no admisible basada en Hungarian inflada + término del jugador.

    Multiplica el matching óptimo caja-goal por un factor > 1 y agrega la
    distancia Manhattan del jugador a la caja no resuelta más cercana.
    Esto la vuelve más agresiva para guiar Greedy/A*, pero deja de garantizar
    optimalidad.
    """
    base_cost = _hungarian_assignment_cost(tuple(state.boxes), tuple(board.goals))
    player_term = _player_to_nearest_unsolved_box(board, state)
    return WEIGHTED_HUNGARIAN_FACTOR * base_cost + PLAYER_DISTANCE_FACTOR * player_term


# ---------------------------------------------------------------------------
# Mapeo de nombres (usados en TOML) a funciones heurísticas
# ---------------------------------------------------------------------------
HEURISTICS: dict[str, HeuristicFn] = {
    "manhattan": manhattan_heuristic,
    "euclidean": euclidean_heuristic,
    "dead_square": dead_square_heuristic,
    "hungarian": hungarian_heuristic,
    "weighted_hungarian": weighted_hungarian_heuristic,
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
