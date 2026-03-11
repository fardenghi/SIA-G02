from tp1_search.types import Direction, Position
from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState


def get_successors(
    board: Board, state: SokobanState
) -> list[tuple[SokobanState, Direction, int]]:
    """Genera todos los estados sucesores válidos con poda de dead squares.

    Prueba las 4 direcciones y retorna las que producen un movimiento válido
    y no llevan una caja a una celda dead (donde nunca puede llegar a un goal).

    Returns:
        Lista de tuplas (nuevo_estado, dirección, costo).
        El costo de cada movimiento es 1.
    """
    successors: list[tuple[SokobanState, Direction, int]] = []
    rows, cols = board.rows, board.cols

    player = state.player
    pr, pc = player.row, player.col

    for direction in Direction:
        dr, dc = direction.delta.row, direction.delta.col

        # Posición destino del jugador
        nr, nc = pr + dr, pc + dc

        # Bounds check
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue

        # Wall check (numpy O(1))
        if board._walls_np[nr, nc]:  # type: ignore[attr-defined]
            continue

        new_player = Position(nr, nc)

        # ¿Hay caja en el destino?
        if state.is_box(new_player):
            # Calcular destino de la caja
            br, bc = nr + dr, nc + dc

            # Bounds check para la caja
            if not (0 <= br < rows and 0 <= bc < cols):
                continue

            # La caja no puede ir a pared ni a otra caja
            if board._walls_np[br, bc] or state.is_box(Position(br, bc)):  # type: ignore[attr-defined]
                continue

            new_box = Position(br, bc)

            # Dead square pruning: descartar si la caja acaba en celda muerta
            if board._dead_sq[br, bc]:  # type: ignore[attr-defined]
                continue

            new_boxes = (state.boxes - {new_player}) | {new_box}
            new_state = SokobanState(player=new_player, boxes=new_boxes)
        else:
            new_state = SokobanState(player=new_player, boxes=state.boxes)

        successors.append((new_state, direction, 1))

    return successors
