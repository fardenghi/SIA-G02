import argparse
import sys

from tp1_search.config import load_config, SearchConfig
from tp1_search.sokoban.parser import parse_board
from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.search.bfs import bfs
from tp1_search.metrics.result import SearchResult
from tp1_search.output.writer import write_replay


# Mapeo de nombre de algoritmo a función de búsqueda
ALGORITHMS = {
    "bfs": bfs,
}


def render_board(board: Board, state: SokobanState, show_dead: bool = False) -> str:
    """Renderiza el tablero como texto.

    Leyenda:
      #   pared
      .   objetivo
      @   jugador
      +   jugador sobre objetivo
      $   caja
      *   caja sobre objetivo
      X   dead square (solo si show_dead=True)
          celda libre
    """
    lines = []
    for r in range(board.rows):
        row = ""
        for c in range(board.cols):
            pos = (r, c)
            is_wall = board._walls_np[r, c]  # type: ignore[attr-defined]
            is_goal = board._goals_np[r, c]  # type: ignore[attr-defined]
            is_dead = board._dead_sq[r, c]  # type: ignore[attr-defined]
            is_player = state.player.row == r and state.player.col == c
            is_box = any(b.row == r and b.col == c for b in state.boxes)

            if is_wall:
                row += "#"
            elif is_player and is_goal:
                row += "+"
            elif is_player:
                row += "@"
            elif is_box and is_goal:
                row += "*"
            elif is_box:
                row += "$"
            elif is_goal:
                row += "."
            elif show_dead and is_dead:
                row += "X"
            else:
                row += " "
        lines.append(row)
    return "\n".join(lines)


def print_dead_squares(board: Board, state: SokobanState) -> None:
    """Imprime el tablero sin y con dead squares marcados lado a lado."""
    normal = render_board(board, state, show_dead=False).splitlines()
    with_dead = render_board(board, state, show_dead=True).splitlines()

    dead_count = int(board._dead_sq.sum())  # type: ignore[attr-defined]
    total_free = int((~board._walls_np).sum())  # type: ignore[attr-defined]

    sep = "     "
    title_l = "Tablero normal"
    title_r = f"Dead squares (X = celda muerta)"
    border = "─" * board.cols

    print(f"{title_l}{sep}{title_r}")
    print(f"{border}{sep}{border}")
    for l, r in zip(normal, with_dead):
        print(f"{l}{sep}{r}")
    print(f"{border}{sep}{border}")
    print(
        f"\nCeldas dead: {dead_count} / {total_free} libres "
        f"({100 * dead_count / total_free:.1f}% del espacio podado)\n"
    )


def run_search(config: SearchConfig):
    """Ejecuta la búsqueda según la configuración dada. Retorna (result, board, initial_state)."""
    search_fn = ALGORITHMS.get(config.algorithm)
    if search_fn is None:
        raise NotImplementedError(
            f"Algoritmo '{config.algorithm}' aún no implementado. "
            f"Disponibles: {', '.join(sorted(ALGORITHMS))}"
        )

    board, initial_state = parse_board(config.board_path)
    result = search_fn(board, initial_state)
    return result, board, initial_state


def print_result(result: SearchResult) -> None:
    """Imprime el resultado de la búsqueda."""
    print("=" * 40)
    print(f"Resultado:          {'EXITO' if result.success else 'FRACASO'}")
    print(f"Costo (pasos):      {result.cost}")
    print(f"Nodos expandidos:   {result.expanded_nodes}")
    print(f"Nodos en frontera:  {result.frontier_nodes}")
    print(f"Tiempo:             {result.time_elapsed:.4f}s")

    if result.success:
        moves = " -> ".join(d.name for d in result.path)
        print(f"Camino ({len(result.path)} pasos): {moves}")

    print("=" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Motor de búsqueda para Sokoban — TP1 SIA"
    )
    parser.add_argument(
        "config",
        help="Ruta al archivo de configuración .toml",
    )
    parser.add_argument(
        "--save-replay",
        action="store_true",
        help="Guarda un archivo JSON de replay en results/raw/ para animar con tp1-animate",
    )
    parser.add_argument(
        "--show-dead-squares",
        action="store_true",
        help="Muestra el tablero con las celdas dead marcadas con X antes de buscar",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error en configuración: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Algoritmo: {config.algorithm}")
    print(f"Tablero:   {config.board_path}")
    if config.heuristic:
        print(f"Heurística: {config.heuristic}")
    print()

    board, initial_state = parse_board(config.board_path)

    if args.show_dead_squares:
        print_dead_squares(board, initial_state)

    search_fn = ALGORITHMS[config.algorithm]
    result = search_fn(board, initial_state)

    print_result(result)

    if args.save_replay:
        replay_path = write_replay(
            result=result,
            board=board,
            initial_state=initial_state,
            board_path=config.board_path,
            algorithm=config.algorithm,
        )
        print(f"\nReplay guardado en: {replay_path}")
        print(f"Animar con: uv run tp1-animate {replay_path}")


if __name__ == "__main__":
    main()
