#!/usr/bin/env python3
"""Comparativa puntual para mostrar el efecto de dead_square.

Genera una figura con:
  - dos tableros showcase con dead squares marcados
  - comparacion de nodos expandidos
  - comparacion de tiempo

Uso:
    uv run python scripts/dead_square_showcase.py
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import multiprocessing as mp
import statistics
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import patches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.search.astar import astar  # noqa: E402
from tp1_search.sokoban.heuristics import (  # noqa: E402
    combine_heuristics_max,
    dead_square_heuristic,
    manhattan_heuristic,
)
from tp1_search.sokoban.parser import parse_board  # noqa: E402
from tp1_search.types import Position  # noqa: E402

DEFAULT_BOARD_PATHS = [
    ROOT / "boards" / "sokoban" / "dead_square_corner_showcase.txt",
    ROOT / "boards" / "sokoban" / "four_corners_showcase.txt",
]

BOARD_TITLES = {
    "dead_square_corner_showcase.txt": "Esquinas predominantes",
    "four_corners_showcase.txt": "Esquinas no predominantes",
}

HEURISTICS = {
    "manhattan": manhattan_heuristic,
    "manhattan+dead_square": combine_heuristics_max(
        [manhattan_heuristic, dead_square_heuristic]
    ),
}

LABELS = {
    "manhattan": "Manhattan",
    "manhattan+dead_square": "Manhattan + Dead Square",
}

COLORS = {
    "manhattan": "#2A9D8F",
    "manhattan+dead_square": "#E76F51",
}


class RunData(TypedDict):
    board: str
    run: int
    heuristic: str
    success: bool
    cost: int
    expanded_nodes: int
    frontier_nodes: int
    time_elapsed: float


def _run_single(board_path: str, heuristic_name: str, queue: mp.Queue) -> None:
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        board, state = parse_board(board_path)
        result = astar(board, state, HEURISTICS[heuristic_name])
    queue.put(
        {
            "heuristic": heuristic_name,
            "success": result.success,
            "cost": result.cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "time_elapsed": result.time_elapsed,
        }
    )


def run_case_n(board_paths: list[Path], runs: int) -> list[RunData]:
    rows: list[RunData] = []
    for run_idx in range(1, runs + 1):
        for board_path in board_paths:
            for heuristic in ["manhattan", "manhattan+dead_square"]:
                queue: mp.Queue = mp.Queue()
                proc = mp.Process(
                    target=_run_single, args=(str(board_path), heuristic, queue)
                )
                proc.start()
                proc.join(timeout=20)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    raise RuntimeError(
                        f"Timeout inesperado para {board_path.name} con {heuristic}"
                    )
                data = queue.get_nowait()
                data["board"] = board_path.name
                data["run"] = run_idx
                rows.append(data)
    return rows


def write_csv(rows: list[RunData], outpath: Path) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "board",
                "heuristic",
                "run",
                "success",
                "cost",
                "expanded_nodes",
                "frontier_nodes",
                "time_elapsed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return outpath


def summarize_rows(rows: list[RunData]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    board_names = sorted({row["board"] for row in rows})

    for board_name in board_names:
        by_heuristic: dict[str, dict[str, float]] = {}
        for heuristic in ["manhattan", "manhattan+dead_square"]:
            subset = [
                row
                for row in rows
                if row["board"] == board_name and row["heuristic"] == heuristic
            ]
            times = [float(row["time_elapsed"]) for row in subset]
            nodes = [float(row["expanded_nodes"]) for row in subset]
            costs = [float(row["cost"]) for row in subset]
            by_heuristic[heuristic] = {
                "time_mean": statistics.fmean(times),
                "time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "nodes_mean": statistics.fmean(nodes),
                "cost_mean": statistics.fmean(costs),
            }
        summary[board_name] = by_heuristic

    return summary


def _board_display_title(board_path: Path) -> str:
    return BOARD_TITLES.get(board_path.name, board_path.stem.replace("_", " ").title())


def _dead_square_stats(board_path: Path) -> tuple[int, int, float]:
    board, _ = parse_board(board_path)
    dead_count = sum(
        1
        for r in range(board.rows)
        for c in range(board.cols)
        if board.is_dead_square(Position(r, c))
    )
    free_count = board.rows * board.cols - len(board.walls)
    ratio = dead_count / free_count if free_count else 0.0
    return dead_count, free_count, ratio


def _set_log_limits(ax: Axes, values: list[float], min_floor: float) -> None:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return

    lower = max(min(positive_values) * 0.92, min_floor)
    upper = max(positive_values) * 1.12
    if math.isclose(lower, upper):
        lower *= 0.9
        upper *= 1.1
    ax.set_ylim(lower, upper)


def _draw_board(ax: Axes, board_path: Path) -> None:
    board, state = parse_board(board_path)
    dead_count, free_count, ratio = _dead_square_stats(board_path)
    ax.set_xlim(0, board.cols)
    ax.set_ylim(board.rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(board.rows):
        for c in range(board.cols):
            pos = Position(r, c)
            if pos in board.walls:
                face = "#4A4A4A"
                edge = "#6A6A6A"
            elif board.is_dead_square(pos):
                face = "#F6D5CF"
                edge = "#D66A54"
            else:
                face = "#D8A15D"
                edge = "#B77B37"

            ax.add_patch(
                patches.Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.9,
                )
            )

    for goal in board.goals:
        ax.add_patch(
            patches.Circle(
                (goal.col + 0.5, goal.row + 0.5),
                0.18,
                facecolor="#F1EAC5",
                edgecolor="#9D8B45",
                linewidth=1.0,
            )
        )

    for box in state.boxes:
        ax.add_patch(
            patches.FancyBboxPatch(
                (box.col + 0.2, box.row + 0.18),
                0.6,
                0.64,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor="#3B7A57",
                edgecolor="#1E4D35",
                linewidth=1.2,
            )
        )

    ax.add_patch(
        patches.Circle(
            (state.player.col + 0.5, state.player.row + 0.5),
            0.22,
            facecolor="#4C78A8",
            edgecolor="#2F4B6C",
            linewidth=1.2,
        )
    )

    ax.set_title(_board_display_title(board_path), fontsize=14, weight="bold", pad=12)
    ax.text(
        0.5,
        -0.08,
        f"dead squares: {dead_count}/{free_count} libres ({ratio:.1%})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
    )


def build_plot(rows: list[RunData], board_paths: list[Path], outpath: Path) -> Path:
    by_name = summarize_rows(rows)
    labels = ["Manhattan", "Manhattan\n+ Dead Square"]

    fig = plt.figure(figsize=(12.5, 5.2 * len(board_paths)), facecolor="white")
    gs = fig.add_gridspec(
        len(board_paths), 3, width_ratios=[1.15, 1.0, 1.0], hspace=0.5, wspace=0.32
    )

    for row_idx, board_path in enumerate(board_paths):
        board_summary = by_name[board_path.name]

        ax_board = fig.add_subplot(gs[row_idx, 0])
        _draw_board(ax_board, board_path)

        ax_nodes = fig.add_subplot(gs[row_idx, 1])
        node_values = [
            board_summary["manhattan"]["nodes_mean"],
            board_summary["manhattan+dead_square"]["nodes_mean"],
        ]
        node_bars = ax_nodes.bar(
            labels,
            node_values,
            color=[COLORS["manhattan"], COLORS["manhattan+dead_square"]],
            width=0.6,
        )
        ax_nodes.set_title("Nodos expandidos", fontsize=14, weight="bold")
        ax_nodes.set_yscale("log")
        ax_nodes.grid(axis="y", linestyle="--", alpha=0.25)
        ax_nodes.tick_params(axis="x", labelsize=11)
        _set_log_limits(ax_nodes, node_values, min_floor=1.0)

        ax_time = fig.add_subplot(gs[row_idx, 2])
        time_values = [
            board_summary["manhattan"]["time_mean"],
            board_summary["manhattan+dead_square"]["time_mean"],
        ]
        time_stds = [
            board_summary["manhattan"]["time_std"],
            board_summary["manhattan+dead_square"]["time_std"],
        ]
        time_bars = ax_time.bar(
            labels,
            time_values,
            color=[COLORS["manhattan"], COLORS["manhattan+dead_square"]],
            width=0.6,
            yerr=time_stds,
            ecolor="#202020",
            capsize=8,
            error_kw={"elinewidth": 1.8, "capthick": 1.8},
        )
        ax_time.set_title("Tiempo de ejecucion", fontsize=14, weight="bold")
        ax_time.set_yscale("log")
        ax_time.grid(axis="y", linestyle="--", alpha=0.25)
        ax_time.tick_params(axis="x", labelsize=11)
        _set_log_limits(
            ax_time,
            time_values
            + [value + std for value, std in zip(time_values, time_stds, strict=False)],
            min_floor=1e-4,
        )

        node_labels = [f"{int(round(v)):,}" for v in node_values]
        for bar, text in zip(node_bars, node_labels, strict=False):
            ax_nodes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.025,
                text,
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

        time_labels = [f"{v:.4f}s" for v in time_values]
        for bar, text, std in zip(time_bars, time_labels, time_stds, strict=False):
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                (bar.get_height() + std) * 1.02,
                text,
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

    fig.suptitle("Efecto de Dead Square sobre A*", fontsize=18, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.03,
        "Comparativa entre un tablero sesgado a esquinas y otro donde no predominan",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, wspace=0.32)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparativa puntual de dead_square")
    parser.add_argument(
        "--board",
        type=Path,
        action="append",
        help=(
            "Tablero showcase para incluir en la comparativa. "
            "Puede repetirse; si se omite, usa ambos casos por defecto"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "benchmark" / "dead_square_showcase.csv",
        help="CSV de salida con las metricas",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "results" / "plots" / "dead_square_showcase.png",
        help="PNG de salida para la figura",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Cantidad de corridas para estimar media y desvio del tiempo (default: 5)",
    )
    args = parser.parse_args()

    board_paths = args.board or list(DEFAULT_BOARD_PATHS)

    rows = run_case_n(board_paths, args.runs)
    csv_path = write_csv(rows, args.csv)
    plot_path = build_plot(rows, board_paths, args.plot)
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
