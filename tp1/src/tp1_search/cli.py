import argparse
import sys

from tp1_search.config import load_config, SearchConfig
from tp1_search.sokoban.parser import parse_board
from tp1_search.search.bfs import bfs
from tp1_search.metrics.result import SearchResult


# Mapeo de nombre de algoritmo a función de búsqueda
ALGORITHMS = {
    "bfs": bfs,
}


def run_search(config: SearchConfig) -> SearchResult:
    """Ejecuta la búsqueda según la configuración dada."""
    search_fn = ALGORITHMS.get(config.algorithm)
    if search_fn is None:
        raise NotImplementedError(
            f"Algoritmo '{config.algorithm}' aún no implementado. "
            f"Disponibles: {', '.join(sorted(ALGORITHMS))}"
        )

    board, initial_state = parse_board(config.board_path)
    return search_fn(board, initial_state)


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

    result = run_search(config)
    print_result(result)


if __name__ == "__main__":
    main()
