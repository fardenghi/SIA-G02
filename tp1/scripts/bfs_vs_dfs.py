#!/usr/bin/env python3
"""Compara BFS vs DFS en un único tablero de Sokoban, ejecutándolos en paralelo.

Uso:
    uv run python scripts/bfs_vs_dfs.py --map boards/sokoban/level_01.txt

Genera un JSON en results/bfs_vs_dfs/<stem>_<YYYYMMDD_HHMMSS>.json con los
resultados de ambos algoritmos.
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

# ---------------------------------------------------------------------------
# Asegurar que el paquete se pueda importar al correr como script
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.sokoban.parser import parse_board  # noqa: E402
from tp1_search.search.bfs import bfs  # noqa: E402
from tp1_search.search.dfs import dfs  # noqa: E402


def _worker(algo_name: str, board_path: str, queue: mp.Queue) -> None:
    """Worker function executed in a child process."""
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

    bfs_queue: mp.Queue = mp.Queue()
    dfs_queue: mp.Queue = mp.Queue()

    bfs_proc = mp.Process(target=_worker, args=("bfs", str(board_path), bfs_queue))
    dfs_proc = mp.Process(target=_worker, args=("dfs", str(board_path), dfs_queue))

    print(f"Ejecutando BFS y DFS sobre {board_path.name} ...")
    bfs_proc.start()
    dfs_proc.start()
    bfs_proc.join()
    dfs_proc.join()

    bfs_result = bfs_queue.get_nowait() if not bfs_queue.empty() else None
    dfs_result = dfs_queue.get_nowait() if not dfs_queue.empty() else None

    output: dict = {
        "bfs": bfs_result or {},
        "dfs": dfs_result or {},
    }

    out_dir = ROOT / "results" / "bfs_vs_dfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{board_path.stem}_{timestamp}.json"

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Console summary
    print()
    for algo, res in output.items():
        if not res:
            print(f"  {algo.upper()}: sin resultado (proceso fallido)")
            continue
        status = "OK" if res["success"] else "FAIL"
        print(
            f"  {algo.upper()}: {status}  cost={res['cost']}  "
            f"expanded={res['expanded_nodes']}  time={res['time_elapsed']:.3f}s"
        )

    print(f"\nResultados guardados en: {out_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
