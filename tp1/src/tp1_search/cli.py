import argparse
import sys

from tp1_search.config import load_config, SearchConfig
from tp1_search.sokoban.parser import parse_board
from tp1_search.search.bfs import bfs
from tp1_search.metrics.result import SearchResult
from tp1_search.output.writer import write_replay


# Mapeo de nombre de algoritmo a función de búsqueda
ALGORITHMS = {
    "bfs": bfs,
}


def run_search(config: SearchConfig):
    """Ejecuta la búsqueda según la configuración dada. Retorna (result, board, initial_state)."""
    search_fn = ALGORITHMS.get(config.algorithm)
    if search_fn is None:
        raise NotImplementedError(
            f"Algoritmo '{config.algorithm}' aún no implementado. "
            f"Disponibles: {', '.join(sorted(ALGORITHMS))}"
        )

    board, initial_state = parse_board(config.board_path)
    result = search_fn(
        board, initial_state, dead_square_pruning=config.dead_square_pruning
    )
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
    print(f"Dead square pruning: {'sí' if config.dead_square_pruning else 'no'}")
    print()

    result, board, initial_state = run_search(config)
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
