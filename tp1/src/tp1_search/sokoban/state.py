from dataclasses import dataclass

from tp1_search.types import Position


@dataclass(frozen=True)
class SokobanState:
    """Parte dinámica del tablero de Sokoban.

    Contiene la posición del jugador y las posiciones de las cajas.
    Es inmutable y hasheable para poder usarse en el conjunto de visitados.
    """

    player: Position
    boxes: frozenset[Position]

    def is_box(self, pos: Position) -> bool:
        return pos in self.boxes
