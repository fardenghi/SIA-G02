#!/usr/bin/env python3
"""Compara BFS vs DFS vs IDDFS en un único tablero de Sokoban, ejecutándolos en paralelo.

Corre 5 veces y guarda cada resultado en una carpeta con timestamp.

Uso:
    uv run python scripts/bfs_vs_dfs.py --map boards/sokoban/level_01.txt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.sokoban.parser import parse_board  # noqa: E402
from tp1_search.search.bfs import bfs  # noqa: E402
from tp1_search.search.dfs import dfs  # noqa: E402
from tp1_search.search.iddfs import iddfs  # noqa: E402

N_RUNS = 5
ALGOS = {"bfs": bfs, "dfs": dfs, "iddfs": iddfs}


def _worker(algo_name: str, board_path: str, queue: mp.Queue) -> None:
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        board, state = parse_board(board_path)
        result = ALGOS[algo_name](board, state)

    queue.put(
        {
            "success": result.success,
            "cost": result.cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "max_frontier_nodes": result.max_frontier_nodes,
            "time_elapsed": result.time_elapsed,
        }
    )


def run_once(board_path: str) -> dict:
    queues = {algo: mp.Queue() for algo in ALGOS}
    procs = {
        algo: mp.Process(target=_worker, args=(algo, board_path, queues[algo]))
        for algo in ALGOS
    }

    for p in procs.values():
        p.start()
    for p in procs.values():
        p.join()

    return {
        algo: queues[algo].get_nowait() if not queues[algo].empty() else {}
        for algo in ALGOS
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara BFS vs DFS vs IDDFS en paralelo sobre un tablero de Sokoban"
    )
    parser.add_argument("--map", required=True, help="Ruta al archivo de tablero")
    args = parser.parse_args()

    board_path = Path(args.map)
    if not board_path.is_absolute():
        board_path = ROOT / board_path
    if not board_path.exists():
        sys.exit(f"Error: tablero no encontrado: {board_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "bfs_vs_dfs_vs_iddfs" / f"{board_path.stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ejecutando {N_RUNS} corridas de BFS / DFS / IDDFS sobre {board_path.name} ...\n")

    for i in range(1, N_RUNS + 1):
        result = run_once(str(board_path))

        out_file = out_dir / f"run_{i}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        parts = "  |  ".join(
            f"{algo.upper()}: cost={result[algo].get('cost')}  "
            f"expanded={result[algo].get('expanded_nodes')}  "
            f"time={result[algo].get('time_elapsed', 0):.4f}s"
            for algo in ALGOS
        )
        print(f"  [{i}/{N_RUNS}]  {parts}")

    print(f"\nResultados guardados en: {out_dir}")
    print(f"\nPara visualizar los resultados ejecutá:")
    print(f"  uv run python scripts/bfs_vs_dfs_vs_iddfs_plots.py --result {out_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
