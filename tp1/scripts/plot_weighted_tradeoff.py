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

    fig.text(
        0.5,
        0.03,
        f"Tablero usado: {board_path.name}",
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.98))
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
        "--out",
        type=Path,
        default=ROOT
        / "results"
        / "plots"
        / "weighted_hungarian_counterexample_tradeoff.png",
        help="Ruta del PNG de salida",
    )
    args = parser.parse_args()

    output = build_plot(args.board, args.out)
    print(output)


if __name__ == "__main__":
    main()
