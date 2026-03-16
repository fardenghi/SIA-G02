#!/usr/bin/env python3
"""Benchmark runner: prueba todas las combinaciones algoritmo × heurística × tablero.

Uso:
    uv run python scripts/run_batch.py [--timeout SECONDS] [--output PATH]

Genera un CSV con columnas:
    board, algorithm, heuristic, success, cost, expanded_nodes, frontier_nodes, time_elapsed, timed_out
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Asegurar que el paquete se pueda importar al correr como script
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.sokoban.parser import parse_board  # noqa: E402
from tp1_search.sokoban.heuristics import HEURISTICS, HeuristicFn  # noqa: E402
from tp1_search.search.bfs import bfs  # noqa: E402
from tp1_search.search.dfs import dfs  # noqa: E402
from tp1_search.search.iddfs import iddfs  # noqa: E402
from tp1_search.search.greedy import greedy  # noqa: E402
from tp1_search.search.astar import astar  # noqa: E402

# ---------------------------------------------------------------------------
# Combinaciones a ejecutar
# ---------------------------------------------------------------------------
BOARDS = sorted(Path(ROOT / "boards" / "sokoban").glob("level_*.txt"))

UNINFORMED = {"bfs": bfs, "dfs": dfs, "iddfs": iddfs}
INFORMED = {"greedy": greedy, "astar": astar}
HEURISTIC_NAMES = list(HEURISTICS.keys())  # manhattan, euclidean, dead_square

CSV_COLUMNS = [
    "board",
    "algorithm",
    "heuristic",
    "success",
    "cost",
    "expanded_nodes",
    "frontier_nodes",
    "time_elapsed",
    "timed_out",
]


def _run_single(
    board_path: str, algo_name: str, heuristic_name: str | None, queue: mp.Queue
) -> None:
    """Worker function executed in a child process."""
    import io, contextlib

    # Suppress search debug prints
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        board, state = parse_board(board_path)
        if heuristic_name:
            heuristic_fn = HEURISTICS[heuristic_name]
            algo_fn = INFORMED[algo_name]
            result = algo_fn(board, state, heuristic_fn)
        else:
            algo_fn = UNINFORMED[algo_name]
            result = algo_fn(board, state)

    queue.put(
        {
            "success": result.success,
            "cost": result.cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "time_elapsed": result.time_elapsed,
        }
    )


def run_with_timeout(
    board_path: str,
    algo_name: str,
    heuristic_name: str | None,
    timeout: float,
) -> dict:
    """Run a single search with a timeout. Returns a dict for CSV row."""
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_run_single, args=(board_path, algo_name, heuristic_name, queue)
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "success": False,
            "cost": 0,
            "expanded_nodes": -1,
            "frontier_nodes": -1,
            "time_elapsed": timeout,
            "timed_out": True,
        }

    if not queue.empty():
        data = queue.get_nowait()
        data["timed_out"] = False
        return data

    # Process finished but no result (crash)
    return {
        "success": False,
        "cost": 0,
        "expanded_nodes": -1,
        "frontier_nodes": -1,
        "time_elapsed": 0.0,
        "timed_out": False,
    }


def build_configs() -> list[tuple[str, str, str | None]]:
    """Return all (board_path, algorithm, heuristic_or_None) combos."""
    configs = []
    for board in BOARDS:
        board_str = str(board)
        for algo in UNINFORMED:
            configs.append((board_str, algo, None))
        for algo in INFORMED:
            for h in HEURISTIC_NAMES:
                configs.append((board_str, algo, h))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark de algoritmos de búsqueda en Sokoban"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout por ejecución en segundos (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "results" / "benchmark" / "benchmark.csv"),
        help="Ruta del CSV de salida",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    total = len(configs)

    print(f"=== Benchmark Sokoban ===")
    print(f"Tableros:       {len(BOARDS)}")
    print(f"Combinaciones:  {total}")
    print(f"Timeout:        {args.timeout}s por ejecución")
    print(f"Output:         {output_path}")
    print()

    rows: list[dict] = []
    t_start = time.time()

    for i, (board_path, algo, heuristic) in enumerate(configs, 1):
        board_name = Path(board_path).stem
        h_label = heuristic or "-"
        tag = f"[{i:3d}/{total}] {board_name:10s} | {algo:6s} | {h_label:12s}"

        print(f"{tag} ...", end=" ", flush=True)

        data = run_with_timeout(board_path, algo, heuristic, args.timeout)

        status = (
            "TIMEOUT" if data["timed_out"] else ("OK" if data["success"] else "FAIL")
        )
        exp = data["expanded_nodes"]
        t = data["time_elapsed"]
        print(f"{status:7s}  expanded={exp}  time={t:.3f}s")

        rows.append(
            {
                "board": board_name,
                "algorithm": algo,
                "heuristic": heuristic or "-",
                **data,
            }
        )

    elapsed = time.time() - t_start
    print(f"\n=== Benchmark completado en {elapsed:.1f}s ===")

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados guardados en: {output_path}")

    # Quick summary
    solved = sum(1 for r in rows if r["success"])
    timeouts = sum(1 for r in rows if r["timed_out"])
    print(f"  Resueltos: {solved}/{total}  |  Timeouts: {timeouts}/{total}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
