import argparse
import sys

from tp1_search.config import load_config, SearchConfig
from tp1_search.sokoban.parser import parse_board
from tp1_search.sokoban.heuristics import (
    HEURISTICS,
    HeuristicFn,
    combine_heuristics_max,
)
from tp1_search.search.bfs import bfs
from tp1_search.search.dfs import dfs
from tp1_search.search.iddfs import iddfs
from tp1_search.search.greedy import greedy
from tp1_search.search.astar import astar
from tp1_search.metrics.result import SearchResult
from tp1_search.output.writer import write_replay


# Algoritmos no informados: (board, state) -> SearchResult
UNINFORMED_ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "iddfs": iddfs,
}

# Algoritmos informados: (board, state, heuristic) -> SearchResult
INFORMED_ALGORITHMS = {
    "greedy": greedy,
    "astar": astar,
}


def _format_heuristics(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return f"max({', '.join(names)})"


def _build_heuristic_fn(config: SearchConfig) -> tuple[HeuristicFn, str]:
    heuristic_names = config.heuristics
    heuristic_fns = [HEURISTICS[name] for name in heuristic_names]
    heuristic_fn = combine_heuristics_max(heuristic_fns)
    return heuristic_fn, _format_heuristics(heuristic_names)

def run_search(config: SearchConfig):
    """Ejecuta la búsqueda según la configuración dada. Retorna (result, board, initial_state)."""
    all_algorithms = set(UNINFORMED_ALGORITHMS) | set(INFORMED_ALGORITHMS)
    if config.algorithm not in all_algorithms:
        raise NotImplementedError(
            f"Algoritmo '{config.algorithm}' aún no implementado. "
            f"Disponibles: {', '.join(sorted(all_algorithms))}"
        )

    board, initial_state = parse_board(config.board_path)

    if config.algorithm in INFORMED_ALGORITHMS:
        heuristic_fn, _ = _build_heuristic_fn(config)
        result = INFORMED_ALGORITHMS[config.algorithm](
            board, initial_state, heuristic_fn
        )
    else:
        result = UNINFORMED_ALGORITHMS[config.algorithm](board, initial_state)

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
    if config.heuristics:
        print(f"Heurística: {_format_heuristics(config.heuristics)}")
    print()

    board, initial_state = parse_board(config.board_path)

    if config.algorithm in INFORMED_ALGORITHMS:
        heuristic_fn, _ = _build_heuristic_fn(config)
        result = INFORMED_ALGORITHMS[config.algorithm](
            board, initial_state, heuristic_fn
        )
    else:
        result = UNINFORMED_ALGORITHMS[config.algorithm](board, initial_state)

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
