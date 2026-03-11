from tp1_search.types import Direction
from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.actions import apply_action


def get_successors(
    board: Board, state: SokobanState
) -> list[tuple[SokobanState, Direction, int]]:
    """Genera todos los estados sucesores válidos.

    Prueba las 4 direcciones y retorna las que producen un movimiento válido.

    Returns:
        Lista de tuplas (nuevo_estado, dirección, costo).
        El costo de cada movimiento es 1.
    """
    successors: list[tuple[SokobanState, Direction, int]] = []

    for direction in Direction:
        new_state = apply_action(board, state, direction)
        if new_state is not None:
            successors.append((new_state, direction, 1))

    return successors
