from pathlib import Path

from tp1_search.types import Position
from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState

# Símbolos del formato de tablero
WALL = "#"
BOX = "$"
BOX_ON_GOAL = "*"
GOAL = "."
PLAYER = "@"
PLAYER_ON_GOAL = "+"
FLOOR = " "

VALID_SYMBOLS = {WALL, BOX, BOX_ON_GOAL, GOAL, PLAYER, PLAYER_ON_GOAL, FLOOR}


def parse_board(path: str | Path) -> tuple[Board, SokobanState]:
    """Lee un archivo .txt de Sokoban y retorna (Board, SokobanState).

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el tablero es inválido (sin jugador, sin cajas, etc.).
    """
    text = Path(path).read_text()
    return parse_board_string(text)


def parse_board_string(text: str) -> tuple[Board, SokobanState]:
    """Parsea un string con el contenido de un tablero de Sokoban."""
    lines = text.rstrip("\n").split("\n")

    walls: set[Position] = set()
    goals: set[Position] = set()
    boxes: set[Position] = set()
    player: Position | None = None

    rows = len(lines)
    cols = max(len(line) for line in lines) if lines else 0

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            pos = Position(r, c)

            if ch == WALL:
                walls.add(pos)
            elif ch == BOX:
                boxes.add(pos)
            elif ch == BOX_ON_GOAL:
                boxes.add(pos)
                goals.add(pos)
            elif ch == GOAL:
                goals.add(pos)
            elif ch == PLAYER:
                if player is not None:
                    raise ValueError(f"Múltiples jugadores: {player} y {pos}")
                player = pos
            elif ch == PLAYER_ON_GOAL:
                if player is not None:
                    raise ValueError(f"Múltiples jugadores: {player} y {pos}")
                player = pos
                goals.add(pos)
            elif ch != FLOOR:
                raise ValueError(f"Símbolo desconocido '{ch}' en fila {r}, col {c}")

    # Validaciones
    if player is None:
        raise ValueError("No se encontró jugador (@) en el tablero")
    if len(boxes) == 0:
        raise ValueError("No se encontraron cajas ($) en el tablero")
    if len(goals) == 0:
        raise ValueError("No se encontraron objetivos (.) en el tablero")
    if len(boxes) != len(goals):
        raise ValueError(
            f"Cantidad de cajas ({len(boxes)}) != cantidad de objetivos ({len(goals)})"
        )

    board = Board(
        rows=rows,
        cols=cols,
        walls=frozenset(walls),
        goals=frozenset(goals),
    )
    state = SokobanState(player=player, boxes=frozenset(boxes))

    return board, state
