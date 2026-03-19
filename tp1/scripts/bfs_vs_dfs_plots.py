#!/usr/bin/env python3
"""Visualización comparativa BFS vs DFS.

Uso:
    uv run python scripts/bfs_vs_dfs_plots.py [--input DIR] [--output PATH]

Lee los JSON de results/bfs_vs_dfs/ (generados por bfs_vs_dfs.py) y produce
un PNG con cuatro subgráficos: nodos expandidos, tiempo, costo y status.
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
OK_MARKER = "✓"
FAIL_MARKER = "✗"


# ── data loading ─────────────────────────────────────────────────────────


# ── plotting helpers ──────────────────────────────────────────────────────


def _extract(results: dict, algo: str, metric: str) -> list:
    return [results[b].get(algo, {}).get(metric, None) for b in results]


def _annotate_status(ax, x_positions, values, statuses, offsets):
    """Draw ✓/✗ above each bar according to success status."""
    for x, v, ok, off in zip(x_positions, values, statuses, offsets):
        if v is None:
            continue
        label = OK_MARKER if ok else FAIL_MARKER
        color = "#2ca02c" if ok else "#d62728"
        ax.text(
            x + off,
            v * 1.05,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            color=color,
        )


def _bar_group(ax, boards, bfs_vals, dfs_vals, ylabel, title, log_scale=False):
    """Grouped bar chart (BFS | DFS) for a single metric."""
    n = len(boards)
    x = np.arange(n)
    w = 0.35

    bfs_plot = [v if v is not None else 0 for v in bfs_vals]
    dfs_plot = [v if v is not None else 0 for v in dfs_vals]

    ax.bar(x - w / 2, bfs_plot, w, label="BFS", color=BFS_COLOR)
    ax.bar(x + w / 2, dfs_plot, w, label="DFS", color=DFS_COLOR)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(boards))
    ax.legend(fontsize=9)

    if log_scale:
        ax.set_yscale("log")

    # Force plain decimal notation on both major and minor ticks
    def _plain_fmt(x, _pos):
        if x == 0:
            return "0"
        if x >= 1:
            return f"{x:.0f}" if x == int(x) else f"{x:g}"
        # Small decimals: strip trailing zeros
        return f"{x:.10f}".rstrip("0").rstrip(".")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_plain_fmt))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(_plain_fmt))

    return x, w


# ── individual plot functions ─────────────────────────────────────────────


def _plot_expanded(ax, boards, results):
    bfs_vals = _extract(results, "bfs", "expanded_nodes")
    dfs_vals = _extract(results, "dfs", "expanded_nodes")
    bfs_ok = _extract(results, "bfs", "success")
    dfs_ok = _extract(results, "dfs", "success")

    x, w = _bar_group(
        ax, boards, bfs_vals, dfs_vals,
        ylabel="Nodos expandidos",
        title="Nodos expandidos",
        log_scale=True,
    )
    _annotate_status(ax, x, [v or 1 for v in bfs_vals], bfs_ok, [-w / 2] * len(x))
    _annotate_status(ax, x, [v or 1 for v in dfs_vals], dfs_ok, [+w / 2] * len(x))


def _plot_time(ax, boards, results):
    bfs_vals = _extract(results, "bfs", "time_elapsed")
    dfs_vals = _extract(results, "dfs", "time_elapsed")
    bfs_ok = _extract(results, "bfs", "success")
    dfs_ok = _extract(results, "dfs", "success")

    x, w = _bar_group(
        ax, boards, bfs_vals, dfs_vals,
        ylabel="Tiempo (s)",
        title="Tiempo de ejecución",
        log_scale=True,
    )
    _annotate_status(ax, x, [v or 1e-9 for v in bfs_vals], bfs_ok, [-w / 2] * len(x))
    _annotate_status(ax, x, [v or 1e-9 for v in dfs_vals], dfs_ok, [+w / 2] * len(x))


def _plot_cost(ax, boards, results):
    bfs_vals = _extract(results, "bfs", "cost")
    dfs_vals = _extract(results, "dfs", "cost")
    bfs_ok = _extract(results, "bfs", "success")
    dfs_ok = _extract(results, "dfs", "success")

    x, w = _bar_group(
        ax, boards, bfs_vals, dfs_vals,
        ylabel="Costo (pasos)",
        title="Costo de la solución",
        log_scale=False,
    )
    _annotate_status(ax, x, [max(v or 0, 0.1) for v in bfs_vals], bfs_ok, [-w / 2] * len(x))
    _annotate_status(ax, x, [max(v or 0, 0.1) for v in dfs_vals], dfs_ok, [+w / 2] * len(x))


def _plot_frontier(ax, boards, results):
    bfs_vals = _extract(results, "bfs", "frontier_nodes")
    dfs_vals = _extract(results, "dfs", "frontier_nodes")
    bfs_ok = _extract(results, "bfs", "success")
    dfs_ok = _extract(results, "dfs", "success")

    x, w = _bar_group(
        ax, boards, bfs_vals, dfs_vals,
        ylabel="Nodos frontera",
        title="Nodos frontera",
        log_scale=False,
    )
    _annotate_status(ax, x, [v or 1 for v in bfs_vals], bfs_ok, [-w / 2] * len(x))
    _annotate_status(ax, x, [v or 1 for v in dfs_vals], dfs_ok, [+w / 2] * len(x))


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera gráfico comparativo BFS vs DFS desde un JSON de resultados"
    )
    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="Ruta al JSON generado por bfs_vs_dfs.py",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"Error: archivo no encontrado: {result_path}", file=sys.stderr)
        sys.exit(1)

    output_path = result_path.with_suffix(".png")

    with open(result_path) as fp:
        data = json.load(fp)
    board_name = result_path.stem.rsplit("_", 2)[0]
    results = {board_name: data}
    boards = [board_name]

    print(f"Tableros encontrados: {', '.join(boards)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("BFS vs DFS — Comparación de métricas", fontsize=14, fontweight="bold")

    _plot_expanded(axes[0, 0], boards, results)
    _plot_frontier(axes[0, 1], boards, results)
    _plot_time(axes[1, 0], boards, results)
    _plot_cost(axes[1, 1], boards, results)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    main()
