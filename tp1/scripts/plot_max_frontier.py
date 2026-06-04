#!/usr/bin/env python3
"""Comparación del tamaño de frontera final vs máxima a lo largo de la ejecución.

Lee los JSONs de una carpeta de corridas (generada por bfs_vs_dfs_vs_iddfs.py),
promedia las métricas entre corridas y genera un PNG comparativo.

Uso:
    uv run python scripts/plot_max_frontier.py --result results/bfs_vs_dfs_vs_iddfs/level_01_20260320_093344/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

ALGOS = ["bfs", "dfs", "iddfs"]
COLORS = {"frontier": "#4C72B0", "max_frontier": "#DD8452"}
LABELS = {"bfs": "BFS", "dfs": "DFS", "iddfs": "IDDFS"}

def load_runs(folder: Path) -> list[dict]:
    files = sorted(folder.glob("run_*.json"))
    if not files:
        print(f"Error: no se encontraron archivos run_*.json en {folder}", file=sys.stderr)
        sys.exit(1)
    runs = []
    for f in files:
        with open(f) as fp:
            runs.append(json.load(fp))
    return runs

def compute_stats(runs: list[dict], algo: str, metric: str) -> tuple[float, float]:
    vals = [r[algo][metric] for r in runs if r.get(algo, {}).get(metric) is not None]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))

def _plain_fmt(x, _pos):
    if x == 0:
        return "0"
    if x >= 1:
        return f"{x:.0f}" if x == int(x) else f"{x:g}"
    return f"{x:.10f}".rstrip("0").rstrip(".")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera gráfico comparativo de Frontera Final vs Máxima Frontera"
    )
    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="Carpeta con los run_*.json generados por bfs_vs_dfs_vs_iddfs.py",
    )
    args = parser.parse_args()

    result_dir = Path(args.result)
    if not result_dir.is_dir():
        print(f"Error: carpeta no encontrada: {result_dir}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(result_dir)
    print(f"Corridas cargadas: {len(runs)}  ({result_dir.name})")

    output_path = result_dir / "max_frontier_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    
    means_frontier = [compute_stats(runs, a, "frontier_nodes")[0] for a in ALGOS]
    errs_frontier = [compute_stats(runs, a, "frontier_nodes")[1] for a in ALGOS]
    
    means_max = [compute_stats(runs, a, "max_frontier_nodes")[0] for a in ALGOS]
    errs_max = [compute_stats(runs, a, "max_frontier_nodes")[1] for a in ALGOS]

    x = np.arange(len(ALGOS))
    width = 0.35

    ax.bar(x - width/2, means_frontier, width, yerr=errs_frontier, label="Frontera Final", color=COLORS["frontier"], capsize=5)
    ax.bar(x + width/2, means_max, width, yerr=errs_max, label="Máxima Frontera", color=COLORS["max_frontier"], capsize=5)

    ax.set_title(f"Frontera Final vs Máxima Frontera - {result_dir.name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cantidad de Nodos")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=11)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_plain_fmt))
    
    # Agregar recuadro a estilo
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico guardado en: {output_path}")

if __name__ == "__main__":
    main()
