#!/usr/bin/env python3
"""Grafico puntual para mostrar el trade-off de weighted_hungarian.

Uso:
    uv run python scripts/plot_weighted_tradeoff.py
    uv run python scripts/plot_weighted_tradeoff.py --board boards/sokoban/weighted_hungarian_counterexample.txt
    uv run python scripts/plot_weighted_tradeoff.py --out results/plots/mi_grafico.png
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stderr
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.search.astar import astar  # noqa: E402
from tp1_search.sokoban.heuristics import (  # noqa: E402
    manhattan_heuristic,
    weighted_hungarian_heuristic,
)
from tp1_search.sokoban.parser import parse_board  # noqa: E402


def _run_case(board_path: Path) -> dict[str, dict[str, float]]:
    board, state = parse_board(board_path)

    with redirect_stderr(io.StringIO()):
        manhattan = astar(board, state, manhattan_heuristic)
    with redirect_stderr(io.StringIO()):
        weighted = astar(board, state, weighted_hungarian_heuristic)

    return {
        "manhattan": {
            "cost": float(manhattan.cost),
            "expanded_nodes": float(manhattan.expanded_nodes),
            "time_elapsed": float(manhattan.time_elapsed),
        },
        "weighted_hungarian": {
            "cost": float(weighted.cost),
            "expanded_nodes": float(weighted.expanded_nodes),
            "time_elapsed": float(weighted.time_elapsed),
        },
    }


def _format_value(metric: str, value: float) -> str:
    if metric == "time_elapsed":
        return f"{value:.2f} s"
    return f"{int(value):,}"


def build_plot(board_path: Path, outpath: Path) -> Path:
    data = _run_case(board_path)

    labels = ["Manhattan", "Weighted\nHungarian"]
    colors = ["#2A9D8F", "#E76F51"]
    metrics = [
        ("cost", "Costo de solucion", False),
        ("expanded_nodes", "Nodos expandidos", True),
        ("time_elapsed", "Tiempo de ejecucion (s)", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6))
    fig.patch.set_facecolor("white")

    for ax, (metric, title, log_scale) in zip(axes, metrics, strict=False):
        values = [data["manhattan"][metric], data["weighted_hungarian"][metric]]
        bars = ax.bar(labels, values, color=colors, width=0.58)

        if log_scale:
            ax.set_yscale("log")
            positive = [v for v in values if v > 0]
            ax.set_ylim(min(positive) * 0.6, max(values) * 1.8)
        else:
            ax.set_ylim(0, max(values) * 1.25)

        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.set_axisbelow(True)

        for bar, value in zip(bars, values, strict=False):
            y = value * (1.08 if value > 0 else 1.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                _format_value(metric, value),
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

    man = data["manhattan"]
    weighted = data["weighted_hungarian"]
    node_ratio = man["expanded_nodes"] / weighted["expanded_nodes"]
    time_ratio = man["time_elapsed"] / weighted["time_elapsed"]
    extra_steps = int(weighted["cost"] - man["cost"])


    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.1, 0.98, 0.88))
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def build_plot_v2(board_path: Path, outpath: Path) -> Path:
    data = _run_case(board_path)
    man = data["manhattan"]
    weighted = data["weighted_hungarian"]

    node_ratio = man["expanded_nodes"] / weighted["expanded_nodes"]
    time_ratio = man["time_elapsed"] / weighted["time_elapsed"]
    extra_steps = int(weighted["cost"] - man["cost"])

    fig = plt.figure(figsize=(13, 7.2), facecolor="#FBF8F3")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.93,
        "Weighted Hungarian: mas rapido, menos optimo",
        fontsize=25,
        weight="bold",
        color="#1F1F1F",
    )
    ax.text(
        0.05,
        0.885,
        f"A* sobre {board_path.name}",
        fontsize=13,
        color="#5B5B5B",
    )

    card_specs = [
        (0.05, 0.24, 0.27, 0.56, "#FFFFFF"),
        (0.365, 0.24, 0.27, 0.56, "#FFFFFF"),
        (0.68, 0.24, 0.27, 0.56, "#FFFFFF"),
    ]
    for x, y, w, h, face in card_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.03",
                facecolor=face,
                edgecolor="#E4DCCF",
                linewidth=1.4,
            )
        )

    green = "#2A9D8F"
    orange = "#E76F51"

    def pill(x: float, y: float, text: str, color: str) -> None:
        ax.text(
            x,
            y,
            text,
            fontsize=10,
            color="white",
            weight="bold",
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": color},
        )

    # Card 1: quality
    ax.text(
        0.075, 0.745, "Calidad de solucion", fontsize=16, weight="bold", color="#232323"
    )
    pill(0.075, 0.688, "MANHATTAN", green)
    ax.text(
        0.075, 0.608, f"{int(man['cost'])}", fontsize=36, weight="bold", color="#111111"
    )
    ax.text(0.075, 0.57, "pasos", fontsize=13, color="#666666")
    ax.text(
        0.075,
        0.522,
        "mejor solucion encontrada",
        fontsize=11,
        color=green,
        weight="bold",
    )

    pill(0.075, 0.44, "WEIGHTED HUNGARIAN", orange)
    ax.text(
        0.075,
        0.36,
        f"{int(weighted['cost'])}",
        fontsize=36,
        weight="bold",
        color="#111111",
    )
    ax.text(0.075, 0.322, "pasos", fontsize=13, color="#666666")
    ax.text(
        0.075,
        0.274,
        f"{extra_steps:+d} pasos respecto a Manhattan",
        fontsize=11,
        color=orange,
        weight="bold",
    )

    # Card 2: nodes
    ax.text(
        0.39, 0.745, "Esfuerzo de busqueda", fontsize=16, weight="bold", color="#232323"
    )
    ax.text(0.39, 0.655, f"{node_ratio:.1f}x", fontsize=38, weight="bold", color=orange)
    ax.text(0.39, 0.61, "menos nodos expandidos", fontsize=13, color="#666666")
    ax.text(
        0.39,
        0.52,
        f"Manhattan: {int(man['expanded_nodes']):,}",
        fontsize=16,
        color="#1F1F1F",
    )
    ax.text(
        0.39,
        0.455,
        f"Weighted Hungarian: {int(weighted['expanded_nodes']):,}",
        fontsize=16,
        color="#1F1F1F",
    )
    ax.text(
        0.39,
        0.33,
        "Explora mucho menos el espacio de estados,\npor eso converge bastante antes.",
        fontsize=12,
        color="#666666",
        linespacing=1.35,
    )

    # Card 3: time
    ax.text(
        0.705, 0.745, "Tiempo de ejecucion", fontsize=16, weight="bold", color="#232323"
    )
    ax.text(
        0.705, 0.655, f"{time_ratio:.1f}x", fontsize=38, weight="bold", color=orange
    )
    ax.text(0.705, 0.61, "mas rapido", fontsize=13, color="#666666")
    ax.text(
        0.705,
        0.52,
        f"Manhattan: {man['time_elapsed']:.2f} s",
        fontsize=16,
        color="#1F1F1F",
    )
    ax.text(
        0.705,
        0.455,
        f"Weighted Hungarian: {weighted['time_elapsed']:.2f} s",
        fontsize=16,
        color="#1F1F1F",
    )
    ax.text(
        0.705,
        0.33,
        "Sacrifica optimalidad para priorizar\nestados prometedores mucho antes.",
        fontsize=12,
        color="#666666",
        linespacing=1.35,
    )

    ax.text(
        0.05,
        0.12,
        "Idea fuerza: heuristicas no admisibles pueden perder optimalidad, pero mejorar mucho el rendimiento.",
        fontsize=15,
        color="#1E1E1E",
        weight="bold",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grafico puntual para el caso Manhattan vs weighted_hungarian"
    )
    parser.add_argument(
        "--board",
        type=Path,
        default=ROOT / "boards" / "sokoban" / "weighted_hungarian_counterexample.txt",
        help="Tablero a evaluar",
    )
    parser.add_argument(
        "--style",
        choices=["classic", "ppt"],
        default="classic",
        help="Estilo del grafico: barras clasicas o version mas limpia para PPT",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ruta del PNG de salida",
    )
    args = parser.parse_args()

    if args.out is None:
        filename = (
            "weighted_hungarian_counterexample2_tradeoff.png"
            if args.style == "classic"
            else "weighted_hungarian_counterexample2_tradeoff_v2.png"
        )
        outpath = ROOT / "results" / "plots" / filename
    else:
        outpath = args.out

    builder = build_plot if args.style == "classic" else build_plot_v2
    output = builder(args.board, outpath)
    print(output)


if __name__ == "__main__":
    main()
