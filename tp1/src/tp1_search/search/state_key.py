"""Utilidad compartida para generar claves compactas de estado.

Todos los algoritmos de búsqueda usan esta función para el visited set,
logrando hashing eficiente con bytes en vez de objetos Python.
"""

import struct

from tp1_search.sokoban.state import SokobanState


def state_key(state: SokobanState, cols: int) -> bytes:
    """Convierte un estado a bytes para hashing eficiente en el visited set.

    Codifica la posición del jugador y las cajas como índices lineales
    (row * cols + col) empaquetados en unsigned shorts (2 bytes cada uno).
    Las cajas se ordenan para que el orden no afecte la igualdad.
    """
    player_idx = state.player.row * cols + state.player.col
    box_idxs = sorted(b.row * cols + b.col for b in state.boxes)
    n = len(box_idxs)
    return struct.pack(f"{1 + n}H", player_idx, *box_idxs)
