#!/usr/bin/env python3
"""Visualización comparativa BFS vs DFS vs IDDFS.

Lee los JSONs de una carpeta de corridas (generada por bfs_vs_dfs.py),
promedia las métricas entre corridas y genera un PNG con barras de error en tiempo.

Uso:
    uv run python scripts/bfs_vs_dfs_plots.py --result results/bfs_vs_dfs_vs_iddfs/level_01_20260320_093344/
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
COLORS = {"bfs": "#4C72B0", "dfs": "#DD8452", "iddfs": "#55A868"}
LABELS = {"bfs": "BFS", "dfs": "DFS", "iddfs": "IDDFS"}


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
    vals = [r[algo][metric] for r in runs if r.get(algo, {}).get(metric) is not None]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


# ── plotting helpers ──────────────────────────────────────────────────────


def _plain_fmt(x, _pos):
    if x == 0:
        return "0"
    if x >= 1:
        return f"{x:.0f}" if x == int(x) else f"{x:g}"
    return f"{x:.10f}".rstrip("0").rstrip(".")


def _bar_group(ax, means, errs, ylabel, title, error_bars=False):
    x = np.arange(len(ALGOS))
    w = 0.5

    for i, algo in enumerate(ALGOS):
        yerr = errs[algo] if error_bars else None
        err_kw = dict(capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "black"}) if error_bars else {}
        ax.bar(i, means[algo], w, yerr=yerr, label=LABELS[algo], color=COLORS[algo], **err_kw)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_plain_fmt))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(_plain_fmt))


# ── individual plot functions ─────────────────────────────────────────────


def _plot_expanded(ax, runs):
    means = {a: compute_stats(runs, a, "expanded_nodes")[0] for a in ALGOS}
    errs  = {a: compute_stats(runs, a, "expanded_nodes")[1] for a in ALGOS}
    _bar_group(ax, means, errs, ylabel="Nodos expandidos", title="Nodos expandidos")


def _plot_frontier(ax, runs):
    means = {a: compute_stats(runs, a, "frontier_nodes")[0] for a in ALGOS}
    errs  = {a: compute_stats(runs, a, "frontier_nodes")[1] for a in ALGOS}
    _bar_group(ax, means, errs, ylabel="Nodos frontera", title="Nodos frontera")


def _plot_time(ax, runs):
    means = {a: compute_stats(runs, a, "time_elapsed")[0] for a in ALGOS}
    errs  = {a: compute_stats(runs, a, "time_elapsed")[1] for a in ALGOS}
    _bar_group(ax, means, errs, ylabel="Tiempo (s)", title="Tiempo de ejecución", error_bars=True)


def _plot_cost(ax, runs):
    means = {a: compute_stats(runs, a, "cost")[0] for a in ALGOS}
    errs  = {a: compute_stats(runs, a, "cost")[1] for a in ALGOS}
    _bar_group(ax, means, errs, ylabel="Costo (pasos)", title="Costo de la solución")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera gráfico comparativo BFS vs DFS vs IDDFS desde una carpeta de corridas"
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

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"BFS vs DFS vs IDDFS — {result_dir.name}  (n={len(runs)})",
        fontsize=13, fontweight="bold",
    )

    _plot_expanded(axes[0], runs)
    _plot_frontier(axes[1], runs)
    _plot_time(axes[2], runs)
    _plot_cost(axes[3], runs)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    main()
