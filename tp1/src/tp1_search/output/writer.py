import json
from datetime import datetime
from pathlib import Path

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.actions import apply_action
from tp1_search.metrics.result import SearchResult


def _build_frames(
    board: Board,
    initial_state: SokobanState,
    result: SearchResult,
) -> list[dict]:
    """Genera la lista de frames a partir del estado inicial y el path."""
    frames = []
    state = initial_state

    # Frame inicial
    frames.append(
        {
            "player": [state.player.row, state.player.col],
            "boxes": sorted([b.row, b.col] for b in state.boxes),
        }
    )

    # Un frame por cada movimiento
    for direction in result.path:
        next_state = apply_action(board, state, direction)
        assert next_state is not None, f"Movimiento inválido en replay: {direction}"
        state = next_state
        frames.append(
            {
                "player": [state.player.row, state.player.col],
                "boxes": sorted([b.row, b.col] for b in state.boxes),
            }
        )

    return frames


def write_replay(
    result: SearchResult,
    board: Board,
    initial_state: SokobanState,
    board_path: str,
    algorithm: str,
    output_dir: str | Path = "results/raw",
) -> Path:
    """Serializa la solución a un archivo JSON de replay.

    El archivo contiene:
      - metadata: algoritmo, tablero, métricas
      - board: estructura estática (paredes, objetivos, dimensiones)
      - frames: lista de estados (jugador + cajas) frame por frame
      - moves: secuencia de direcciones

    Returns:
        Path al archivo JSON generado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    board_name = Path(board_path).stem
    filename = f"{algorithm}_{board_name}_{timestamp}.json"
    output_path = output_dir / filename

    frames = (
        _build_frames(board, initial_state, result)
        if result.success
        else [
            {
                "player": [initial_state.player.row, initial_state.player.col],
                "boxes": sorted([b.row, b.col] for b in initial_state.boxes),
            }
        ]
    )

    replay = {
        "metadata": {
            "algorithm": algorithm,
            "board_path": board_path,
            "success": result.success,
            "cost": result.cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "time_elapsed": round(result.time_elapsed, 6),
        },
        "board": {
            "rows": board.rows,
            "cols": board.cols,
            "walls": sorted([p.row, p.col] for p in board.walls),
            "goals": sorted([p.row, p.col] for p in board.goals),
        },
        "moves": [d.name for d in result.path],
        "frames": frames,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(replay, f, indent=2)

    return output_path
