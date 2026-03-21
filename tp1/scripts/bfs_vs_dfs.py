#!/usr/bin/env python3
"""Compara BFS vs DFS en un único tablero de Sokoban, ejecutándolos en paralelo.

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

N_RUNS = 5


def _worker(algo_name: str, board_path: str, queue: mp.Queue) -> None:
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        board, state = parse_board(board_path)
        algo_fn = bfs if algo_name == "bfs" else dfs
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


def run_once(board_path: str) -> dict:
    bfs_queue: mp.Queue = mp.Queue()
    dfs_queue: mp.Queue = mp.Queue()

    bfs_proc = mp.Process(target=_worker, args=("bfs", board_path, bfs_queue))
    dfs_proc = mp.Process(target=_worker, args=("dfs", board_path, dfs_queue))

    bfs_proc.start()
    dfs_proc.start()
    bfs_proc.join()
    dfs_proc.join()

    return {
        "bfs": bfs_queue.get_nowait() if not bfs_queue.empty() else {},
        "dfs": dfs_queue.get_nowait() if not dfs_queue.empty() else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara BFS vs DFS en paralelo sobre un tablero de Sokoban"
    )
    parser.add_argument("--map", required=True, help="Ruta al archivo de tablero")
    args = parser.parse_args()

    board_path = Path(args.map)
    if not board_path.is_absolute():
        board_path = ROOT / board_path
    if not board_path.exists():
        sys.exit(f"Error: tablero no encontrado: {board_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "bfs_vs_dfs" / f"{board_path.stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ejecutando {N_RUNS} corridas de BFS y DFS sobre {board_path.name} ...\n")

    for i in range(1, N_RUNS + 1):
        result = run_once(str(board_path))

        out_file = out_dir / f"run_{i}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        bfs_r = result["bfs"]
        dfs_r = result["dfs"]
        print(f"  [{i}/{N_RUNS}]  "
              f"BFS: cost={bfs_r.get('cost')}  expanded={bfs_r.get('expanded_nodes')}  time={bfs_r.get('time_elapsed', 0):.4f}s  |  "
              f"DFS: cost={dfs_r.get('cost')}  expanded={dfs_r.get('expanded_nodes')}  time={dfs_r.get('time_elapsed', 0):.4f}s")

    print(f"\nResultados guardados en: {out_dir}")
    print(f"\nPara visualizar los resultados ejecutá:")
    print(f"  uv run python scripts/bfs_vs_dfs_plots.py --result {out_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
