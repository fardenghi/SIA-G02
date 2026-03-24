#!/usr/bin/env python3
"""Visualización comparativa BFS vs DFS vs IDDFS.

Lee los JSONs de una carpeta de corridas (generada por bfs_vs_dfs_vs_iddfs.py),
promedia las métricas entre corridas y genera un PNG con barras de error en tiempo.

Uso:
    uv run python scripts/bfs_vs_dfs_vs_iddfs_plots.py --result results/bfs_vs_dfs_vs_iddfs/level_01_20260320_093344/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import ConnectionPatch

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


def _bar_group(ax, means, errs, ylabel, title, error_bars=False, algos=None):
    algos = algos or ALGOS
    x = np.arange(len(algos))
    w = 0.5

    for i, algo in enumerate(algos):
        yerr = errs[algo] if error_bars else None
        err_kw = dict(capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "black"}) if error_bars else {}
        ax.bar(i, means[algo], w, yerr=yerr, label=LABELS[algo], color=COLORS[algo], **err_kw)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in algos], fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_plain_fmt))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(_plain_fmt))


def _add_zoom_inset(ax_orig, ax_zoom, means, errs, ylabel, title, error_bars=False):
    """Dibuja el panel de zoom con recuadro punteado y líneas de conexión al original."""
    bfs_top = means["bfs"] + errs["bfs"]
    dfs_top = means["dfs"] + errs["dfs"]
    ylim = max(bfs_top, dfs_top) * 1.3 or 1.0

    # Solo mostrar los algoritmos que entran en el rango del zoom
    visible_algos = [a for a in ALGOS if means[a] <= ylim]
    _bar_group(ax_zoom, means, errs, "", "", error_bars=error_bars, algos=visible_algos)
    ax_zoom.set_ylim(0, ylim)


    # Recuadro punteado alrededor del zoom axes
    for spine in ax_zoom.spines.values():
        spine.set_linestyle("dashed")
        spine.set_linewidth(1.5)
        spine.set_color("gray")

    # Líneas de conexión desde las barras de BFS/DFS en el original hacia el zoom
    fig = ax_orig.figure
    for xy_orig, xy_zoom in [
        ((-0.4, 0), (0, 1)),   # esquina izquierda
        ((1.4,  0), (1, 1)),   # esquina derecha
    ]:
        con = ConnectionPatch(
            xyA=xy_orig, coordsA=ax_orig.transData,
            xyB=xy_zoom, coordsB=ax_zoom.transAxes,
            color="gray", linestyle="dashed", linewidth=1,
            axesA=ax_orig, axesB=ax_zoom,
        )
        fig.add_artist(con)


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
        help="Carpeta con los run_*.json generados por bfs_vs_dfs_vs_iddfs.py",
    )
    parser.add_argument("--zoom-expanded-nodes", action="store_true", default=False,
                        help="Agrega panel de zoom para nodos expandidos")
    parser.add_argument("--zoom-frontier-nodes", action="store_true", default=False,
                        help="Agrega panel de zoom para nodos frontera")
    parser.add_argument("--zoom-execution-time", action="store_true", default=False,
                        help="Agrega panel de zoom para tiempo de ejecución")
    parser.add_argument("--zoom-solution-cost", action="store_true", default=False,
                        help="Agrega panel de zoom para costo de la solución")
    args = parser.parse_args()

    result_dir = Path(args.result)
    if not result_dir.is_dir():
        print(f"Error: carpeta no encontrada: {result_dir}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(result_dir)
    print(f"Corridas cargadas: {len(runs)}  ({result_dir.name})")

    output_path = result_dir / "comparison.png"

    zoom_flags = [
        args.zoom_expanded_nodes,
        args.zoom_frontier_nodes,
        args.zoom_execution_time,
        args.zoom_solution_cost,
    ]
    any_zoom = any(zoom_flags)

    if any_zoom:
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        top = axes[0]
        bot = axes[1]
    else:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        top = axes
        bot = None

    fig.suptitle(
        f"BFS vs DFS vs IDDFS — {result_dir.name}  (n={len(runs)})",
        fontsize=13, fontweight="bold",
    )

    _plot_expanded(top[0], runs)
    _plot_frontier(top[1], runs)
    _plot_time(top[2], runs)
    _plot_cost(top[3], runs)

    if any_zoom:
        zoom_configs = [
            (0, args.zoom_expanded_nodes, "expanded_nodes", "Nodos expandidos", "Nodos expandidos", False),
            (1, args.zoom_frontier_nodes, "frontier_nodes", "Nodos frontera",   "Nodos frontera",   False),
            (2, args.zoom_execution_time, "time_elapsed",   "Tiempo (s)",       "Tiempo de ejecución", True),
            (3, args.zoom_solution_cost,  "cost",           "Costo (pasos)",    "Costo de la solución", False),
        ]
        for col, enabled, metric, ylabel, title, error_bars in zoom_configs:
            if enabled:
                means = {a: compute_stats(runs, a, metric)[0] for a in ALGOS}
                errs  = {a: compute_stats(runs, a, metric)[1] for a in ALGOS}
                _add_zoom_inset(top[col], bot[col], means, errs, ylabel, title, error_bars=error_bars)
            else:
                bot[col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    main()
