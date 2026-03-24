#!/usr/bin/env python3
"""Grafico puntual para mostrar el trade-off de weighted_hungarian.

Uso:
    uv run python scripts/plot_weighted_tradeoff.py
    uv run python scripts/plot_weighted_tradeoff.py --board boards/sokoban/weighted_hungarian_counterexample.txt
    uv run python scripts/plot_weighted_tradeoff.py --out results/plots/mi_grafico.png
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
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
    return _run_case_n(board_path, runs=1)


def _run_case_n(board_path: Path, runs: int) -> dict[str, dict[str, float]]:
    board, state = parse_board(board_path)

    raw: dict[str, list[dict[str, float]]] = {
        "manhattan": [],
        "weighted_hungarian": [],
    }
    for _ in range(runs):
        with redirect_stderr(io.StringIO()):
            manhattan = astar(board, state, manhattan_heuristic)
        with redirect_stderr(io.StringIO()):
            weighted = astar(board, state, weighted_hungarian_heuristic)
        raw["manhattan"].append(
            {
                "cost": float(manhattan.cost),
                "expanded_nodes": float(manhattan.expanded_nodes),
                "time_elapsed": float(manhattan.time_elapsed),
                "frontier_nodes": float(manhattan.frontier_nodes),
            }
        )
        raw["weighted_hungarian"].append(
            {
                "cost": float(weighted.cost),
                "expanded_nodes": float(weighted.expanded_nodes),
                "time_elapsed": float(weighted.time_elapsed),
                "frontier_nodes": float(weighted.frontier_nodes),
            }
        )

    return {
        "manhattan": {
            "cost": statistics.fmean(item["cost"] for item in raw["manhattan"]),
            "expanded_nodes": statistics.fmean(
                item["expanded_nodes"] for item in raw["manhattan"]
            ),
            "time_elapsed": statistics.fmean(
                item["time_elapsed"] for item in raw["manhattan"]
            ),
            "time_std": statistics.pstdev(
                item["time_elapsed"] for item in raw["manhattan"]
            )
            if len(raw["manhattan"]) > 1
            else 0.0,
            "frontier_nodes": statistics.fmean(
                item["frontier_nodes"] for item in raw["manhattan"]
            ),
        },
        "weighted_hungarian": {
            "cost": statistics.fmean(
                item["cost"] for item in raw["weighted_hungarian"]
            ),
            "expanded_nodes": statistics.fmean(
                item["expanded_nodes"] for item in raw["weighted_hungarian"]
            ),
            "time_elapsed": statistics.fmean(
                item["time_elapsed"] for item in raw["weighted_hungarian"]
            ),
            "time_std": statistics.pstdev(
                item["time_elapsed"] for item in raw["weighted_hungarian"]
            )
            if len(raw["weighted_hungarian"]) > 1
            else 0.0,
            "frontier_nodes": statistics.fmean(
                item["frontier_nodes"] for item in raw["weighted_hungarian"]
            ),
        },
    }


def write_csv(
    board_path: Path,
    runs: int,
    outpath: Path,
) -> Path:
    board, state = parse_board(board_path)
    rows: list[dict[str, float | int | str]] = []
    for run in range(1, runs + 1):
        with redirect_stderr(io.StringIO()):
            manhattan = astar(board, state, manhattan_heuristic)
        with redirect_stderr(io.StringIO()):
            weighted = astar(board, state, weighted_hungarian_heuristic)
        rows.append(
            {
                "run": run,
                "heuristic": "manhattan",
                "cost": manhattan.cost,
                "expanded_nodes": manhattan.expanded_nodes,
                "time_elapsed": manhattan.time_elapsed,
                "frontier_nodes": manhattan.frontier_nodes,
            }
        )
        rows.append(
            {
                "run": run,
                "heuristic": "weighted_hungarian",
                "cost": weighted.cost,
                "expanded_nodes": weighted.expanded_nodes,
                "time_elapsed": weighted.time_elapsed,
                "frontier_nodes": weighted.frontier_nodes,
            }
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "heuristic", "cost", "expanded_nodes", "time_elapsed", "frontier_nodes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return outpath


def _format_value(metric: str, value: float) -> str:
    if metric == "time_elapsed":
        return f"{value:.2f} s"
    return f"{int(value):,}"


def build_plot(board_path: Path, outpath: Path, runs: int = 5) -> Path:
    data = _run_case_n(board_path, runs=runs)

    labels = ["Manhattan", "Weighted\nHungarian"]
    colors = ["#2A9D8F", "#E76F51"]
    metrics = [
        ("cost", "Costo de solucion", False),
        ("expanded_nodes", "Nodos expandidos", True),
        ("time_elapsed", "Tiempo de ejecucion (s)", True),
        ("frontier_nodes", "Nodos frontera (max)", True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.6))
    fig.patch.set_facecolor("white")

    for ax, (metric, title, log_scale) in zip(axes, metrics, strict=False):
        values = [data["manhattan"][metric], data["weighted_hungarian"][metric]]
        error = None
        error_kw = None
        if metric == "time_elapsed":
            error = [
                data["manhattan"]["time_std"],
                data["weighted_hungarian"]["time_std"],
            ]
            error_kw = {"elinewidth": 1.8, "capthick": 1.8}
        bars = ax.bar(
            labels,
            values,
            color=colors,
            width=0.58,
            yerr=error,
            ecolor="#202020",
            capsize=8 if error is not None else 0,
            error_kw=error_kw,
        )

        if log_scale:
            ax.set_yscale("log")
            positive = [v for v in values if v > 0]
            ax.set_ylim(min(positive) * 0.6, max(values) * 1.8)
        else:
            ax.set_ylim(0, max(values) * 1.25)

        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.set_axisbelow(True)

        for idx, (bar, value) in enumerate(zip(bars, values, strict=False)):
            extra = error[idx] if error is not None else 0.0
            y = (value + extra) * (1.08 if value > 0 else 1.0)
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
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT
        / "results"
        / "benchmark"
        / "weighted_hungarian_counterexample_tradeoff.csv",
        help="CSV de salida con las metricas por corrida",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Cantidad de corridas para estimar media y desvio del tiempo (default: 5)",
    )
    args = parser.parse_args()

    write_csv(args.board, args.runs, args.csv)
    output = build_plot(args.board, args.out, runs=args.runs)
    print(output)


if __name__ == "__main__":
    main()
