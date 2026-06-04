from tp1_search.types import Direction, Position
from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState


def apply_action(
    board: Board, state: SokobanState, direction: Direction
) -> SokobanState | None:
    """Intenta mover al jugador en la dirección dada.

    Reglas de Sokoban:
      1. El jugador no puede moverse a una pared.
      2. Si hay una caja en el destino, el jugador la empuja.
      3. Una caja no se puede empujar a una pared ni a otra caja.

    Returns:
        Nuevo SokobanState si el movimiento es válido, None si no lo es.
    """
    new_player: Position = state.player + direction.delta

    # Regla 1: no se puede mover a una pared
    if not board.is_free(new_player):
        return None

    # Si no hay caja en el destino, es un movimiento simple
    if not state.is_box(new_player):
        return SokobanState(player=new_player, boxes=state.boxes)

    # Hay una caja: calcular a dónde se empuja
    new_box: Position = new_player + direction.delta

    # Regla 3: la caja no se puede empujar a pared ni a otra caja
    if not board.is_free(new_box) or state.is_box(new_box):
        return None

    # Movimiento válido con empuje de caja
    new_boxes = (state.boxes - {new_player}) | {new_box}
    return SokobanState(player=new_player, boxes=new_boxes)
