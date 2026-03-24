#!/usr/bin/env python3
"""Visualización comparativa BFS vs DFS.

Lee los JSONs de una carpeta de corridas (generada por bfs_vs_dfs.py),
promedia las métricas entre corridas y genera un PNG con barras de error.

Uso:
    uv run python scripts/bfs_vs_dfs_plots.py --result results/bfs_vs_dfs/level_01_20260320_093344/
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

BFS_COLOR = "#4C72B0"
DFS_COLOR = "#DD8452"


# ── data loading ──────────────────────────────────────────────────────────


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
    """Returns (mean, std) for the given algo+metric across all runs."""
    vals = [r[algo][metric] for r in runs if r.get(algo, {}).get(metric) is not None]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def success_rate(runs: list[dict], algo: str) -> float:
    vals = [r[algo].get("success", False) for r in runs if r.get(algo)]
    return sum(vals) / len(vals) if vals else 0.0


# ── plotting helpers ──────────────────────────────────────────────────────


def _plain_fmt(x, _pos):
    if x == 0:
        return "0"
    if x >= 1:
        return f"{x:.0f}" if x == int(x) else f"{x:g}"
    return f"{x:.10f}".rstrip("0").rstrip(".")


def _bar_group(ax, bfs_mean, dfs_mean, bfs_err, dfs_err, ylabel, title, error_bars=False):
    w = 0.5

    # Normalize to BFS: BFS = 1, DFS = dfs/bfs
    scale = bfs_mean if bfs_mean > 0 else 1.0
    bfs_norm = bfs_mean / scale          # always 1.0
    dfs_norm = dfs_mean / scale
    bfs_err_norm = bfs_err / scale if error_bars else None
    dfs_err_norm = dfs_err / scale if error_bars else None

    err_kw = dict(capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "black"})

    ax.bar(0, bfs_norm, w, yerr=bfs_err_norm, label="BFS", color=BFS_COLOR,
           **(err_kw if error_bars else {}))
    ax.bar(1, dfs_norm, w, yerr=dfs_err_norm, label="DFS", color=DFS_COLOR,
           **(err_kw if error_bars else {}))

    # Annotate actual values above each bar
    ax.text(0, bfs_norm + 0.05, _plain_fmt(bfs_mean, None), ha="center", va="bottom", fontsize=9)
    ax.text(1, dfs_norm + 0.05, _plain_fmt(dfs_mean, None), ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{ylabel}  (× BFS)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BFS", "DFS"], fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))


# ── individual plot functions ─────────────────────────────────────────────


def _plot_expanded(ax, runs):
    bfs_m, bfs_e = compute_stats(runs, "bfs", "expanded_nodes")
    dfs_m, dfs_e = compute_stats(runs, "dfs", "expanded_nodes")
    _bar_group(ax, bfs_m, dfs_m, bfs_e, dfs_e,
               ylabel="Nodos expandidos", title="Nodos expandidos")


def _plot_frontier(ax, runs):
    bfs_m, bfs_e = compute_stats(runs, "bfs", "frontier_nodes")
    dfs_m, dfs_e = compute_stats(runs, "dfs", "frontier_nodes")
    _bar_group(ax, bfs_m, dfs_m, bfs_e, dfs_e,
               ylabel="Nodos frontera", title="Nodos frontera")


def _plot_time(ax, runs):
    bfs_m, bfs_e = compute_stats(runs, "bfs", "time_elapsed")
    dfs_m, dfs_e = compute_stats(runs, "dfs", "time_elapsed")
    _bar_group(ax, bfs_m, dfs_m, bfs_e, dfs_e,
               ylabel="Tiempo (s)", title="Tiempo de ejecución", error_bars=True)


def _plot_cost(ax, runs):
    bfs_m, bfs_e = compute_stats(runs, "bfs", "cost")
    dfs_m, dfs_e = compute_stats(runs, "dfs", "cost")
    _bar_group(ax, bfs_m, dfs_m, bfs_e, dfs_e,
               ylabel="Costo (pasos)", title="Costo de la solución")



# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera gráfico comparativo BFS vs DFS desde una carpeta de corridas"
    )
    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="Carpeta con los run_*.json generados por bfs_vs_dfs.py",
    )
    args = parser.parse_args()

    result_dir = Path(args.result)
    if not result_dir.is_dir():
        print(f"Error: carpeta no encontrada: {result_dir}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(result_dir)
    print(f"Corridas cargadas: {len(runs)}  ({result_dir.name})")

    output_path = result_dir / "comparison.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"BFS vs DFS — {result_dir.name}  (n={len(runs)})",
        fontsize=13, fontweight="bold",
    )

    _plot_expanded(axes[0, 0], runs)
    _plot_frontier(axes[0, 1], runs)
    _plot_time(axes[1, 0], runs)
    _plot_cost(axes[1, 1], runs)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    main()
