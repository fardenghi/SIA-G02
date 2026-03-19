#!/usr/bin/env python3
"""Estudio de heuristicas en funcion de la cantidad de cajas.

Genera una familia controlada de tableros (ya versionados en boards/sokoban/box_count_eval),
ejecuta A* con todas las heuristicas y produce:

  - CSV con metricas por tablero / heuristica
  - grafico compuesto listo para diapositiva

Uso:
    uv run python scripts/box_count_heuristic_study.py
    uv run python scripts/box_count_heuristic_study.py --timeout 12
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import multiprocessing as mp
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import patches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.search.astar import astar  # noqa: E402
from tp1_search.sokoban.heuristics import HEURISTICS  # noqa: E402
from tp1_search.sokoban.parser import parse_board  # noqa: E402

BOARD_DIR = ROOT / "boards" / "sokoban" / "box_count_eval"
BOARDS = sorted(BOARD_DIR.glob("boxes_*.txt"))

HEURISTIC_ORDER = [
    "manhattan",
    "euclidean",
    "dead_square",
    "hungarian",
    "weighted_hungarian",
]

HEURISTIC_LABELS = {
    "manhattan": "Manhattan",
    "euclidean": "Euclidean",
    "dead_square": "Dead Square",
    "hungarian": "Hungarian",
    "weighted_hungarian": "Weighted Hungarian",
}

LEGEND_LABELS = {
    "manhattan": "Manhattan",
    "euclidean": "Euclidean",
    "dead_square": "Dead Sq.",
    "hungarian": "Hungarian",
    "weighted_hungarian": "Weighted Hung.",
}

HEURISTIC_COLORS = {
    "manhattan": "#2A9D8F",
    "euclidean": "#4C78A8",
    "dead_square": "#7A7A7A",
    "hungarian": "#E9C46A",
    "weighted_hungarian": "#E76F51",
}

CSV_COLUMNS = [
    "board",
    "box_count",
    "heuristic",
    "success",
    "cost",
    "expanded_nodes",
    "frontier_nodes",
    "time_elapsed",
    "timed_out",
]


class RunData(TypedDict):
    success: bool
    cost: int
    expanded_nodes: int
    frontier_nodes: int
    time_elapsed: float
    timed_out: bool


class RowData(RunData):
    board: str
    box_count: int
    heuristic: str


def _run_single(board_path: str, heuristic_name: str, queue: mp.Queue) -> None:
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        board, state = parse_board(board_path)
        result = astar(board, state, HEURISTICS[heuristic_name])

    queue.put(
        {
            "success": result.success,
            "cost": result.cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "time_elapsed": result.time_elapsed,
        }
    )


def run_with_timeout(board_path: Path, heuristic_name: str, timeout: float) -> RunData:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_single, args=(str(board_path), heuristic_name, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "success": False,
            "cost": 0,
            "expanded_nodes": -1,
            "frontier_nodes": -1,
            "time_elapsed": timeout,
            "timed_out": True,
        }

    if not queue.empty():
        data = queue.get_nowait()
        data["timed_out"] = False
        return data

    return {
        "success": False,
        "cost": 0,
        "expanded_nodes": -1,
        "frontier_nodes": -1,
        "time_elapsed": 0.0,
        "timed_out": False,
    }


def collect_rows(timeout: float) -> list[RowData]:
    rows: list[RowData] = []

    for board_path in BOARDS:
        board, state = parse_board(board_path)
        box_count = len(state.boxes)

        for heuristic in HEURISTIC_ORDER:
            print(
                f"[{board_path.stem}] A* + {heuristic:18s} ...",
                end=" ",
                flush=True,
            )
            data = run_with_timeout(board_path, heuristic, timeout)
            status = (
                "TIMEOUT"
                if data["timed_out"]
                else ("OK" if data["success"] else "FAIL")
            )
            print(
                f"{status:7s} cost={data['cost']:>3} expanded={data['expanded_nodes']:>8} time={float(data['time_elapsed']):.3f}s"
            )
            rows.append(
                {
                    "board": board_path.stem,
                    "box_count": box_count,
                    "heuristic": heuristic,
                    **data,
                }
            )

    return rows


def write_csv(rows: list[RowData], outpath: Path) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return outpath


def _draw_board(ax: Axes, board_path: Path) -> None:
    board, state = parse_board(board_path)
    ax.set_xlim(0, board.cols)
    ax.set_ylim(board.rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(board.rows):
        for c in range(board.cols):
            is_wall = any(w.row == r and w.col == c for w in board.walls)
            face = "#444444" if is_wall else "#D8A15D"
            ax.add_patch(
                patches.Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#6B6B6B" if is_wall else "#B77B37",
                    linewidth=0.8,
                )
            )

    for goal in board.goals:
        ax.add_patch(
            patches.Circle(
                (goal.col + 0.5, goal.row + 0.5),
                0.18,
                facecolor="#F2E9C9",
                edgecolor="#BCA96A",
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
    ax.set_title(
        f"{len(state.boxes)} caja{'s' if len(state.boxes) != 1 else ''}",
        fontsize=11,
        pad=6,
    )


def _rows_for_heuristic(rows: list[RowData], heuristic: str) -> list[RowData]:
    return sorted(
        [row for row in rows if row["heuristic"] == heuristic],
        key=lambda row: int(row["box_count"]),
    )


def build_plot(rows: list[RowData], timeout: float, outpath: Path) -> Path:
    box_counts = [len(parse_board(board)[1].boxes) for board in BOARDS]

    fig = plt.figure(figsize=(15, 9), facecolor="white")
    gs = fig.add_gridspec(
        3, 12, height_ratios=[1.05, 1.65, 1.1], hspace=0.48, wspace=0.7
    )

    board_axes = [
        fig.add_subplot(gs[0, i * 12 // len(BOARDS) : (i + 1) * 12 // len(BOARDS)])
        for i in range(len(BOARDS))
    ]
    for ax, board_path in zip(board_axes, BOARDS, strict=False):
        _draw_board(ax, board_path)

    ax_nodes = fig.add_subplot(gs[1, :6])
    ax_time = fig.add_subplot(gs[1, 6:])
    ax_status = fig.add_subplot(gs[2, :])

    for heuristic in HEURISTIC_ORDER:
        series = _rows_for_heuristic(rows, heuristic)
        xs = [int(row["box_count"]) for row in series]

        node_values = [
            float(row["expanded_nodes"])
            if not bool(row["timed_out"]) and float(row["expanded_nodes"]) > 0
            else math.nan
            for row in series
        ]
        time_values = [
            float(row["time_elapsed"]) if not bool(row["timed_out"]) else math.nan
            for row in series
        ]

        color = HEURISTIC_COLORS[heuristic]
        label = LEGEND_LABELS[heuristic]

        ax_nodes.plot(
            xs, node_values, marker="o", linewidth=2.4, color=color, label=label
        )
        ax_time.plot(
            xs, time_values, marker="o", linewidth=2.4, color=color, label=label
        )

        timeout_x = [int(row["box_count"]) for row in series if bool(row["timed_out"])]
        if timeout_x:
            ax_time.scatter(
                timeout_x,
                [timeout] * len(timeout_x),
                marker="X",
                s=90,
                color=color,
                zorder=4,
            )
            for x in timeout_x:
                ax_time.text(
                    x,
                    timeout * 1.05,
                    "T/O",
                    color=color,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    weight="bold",
                )

    ax_nodes.set_title(
        "A*: nodos expandidos vs cantidad de cajas", fontsize=13, weight="bold"
    )
    ax_nodes.set_xlabel("Cantidad de cajas")
    ax_nodes.set_ylabel("Nodos expandidos")
    ax_nodes.set_yscale("log")
    ax_nodes.set_xticks(box_counts)
    ax_nodes.grid(axis="y", linestyle="--", alpha=0.25)

    ax_time.set_title("A*: tiempo vs cantidad de cajas", fontsize=13, weight="bold")
    ax_time.set_xlabel("Cantidad de cajas")
    ax_time.set_ylabel("Tiempo (s)")
    ax_time.set_yscale("log")
    ax_time.set_xticks(box_counts)
    ax_time.grid(axis="y", linestyle="--", alpha=0.25)
    ax_time.axhline(timeout, color="#999999", linestyle=":", linewidth=1.2)
    ax_time.text(
        box_counts[-1] - 0.1,
        timeout * 1.14,
        f"timeout = {timeout:.0f}s",
        color="#777777",
        fontsize=9,
    )

    status_values = []
    for heuristic in HEURISTIC_ORDER:
        row_vals = []
        for board_path in BOARDS:
            box_count = len(parse_board(board_path)[1].boxes)
            match = next(
                row
                for row in rows
                if row["heuristic"] == heuristic and int(row["box_count"]) == box_count
            )
            if bool(match["timed_out"]):
                row_vals.append(0)
            elif bool(match["success"]):
                row_vals.append(1)
            else:
                row_vals.append(-1)
        status_values.append(row_vals)

    cmap = mcolors.ListedColormap(["#D95D5D", "#EFE6DA", "#4CAF50"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax_status.imshow(status_values, cmap=cmap, norm=norm, aspect="auto")
    ax_status.set_title(
        "Estado por heuristica y cantidad de cajas", fontsize=13, weight="bold"
    )
    ax_status.set_xticks(range(len(BOARDS)))
    ax_status.set_xticklabels([str(count) for count in box_counts])
    ax_status.set_yticks(range(len(HEURISTIC_ORDER)))
    ax_status.set_yticklabels([HEURISTIC_LABELS[h] for h in HEURISTIC_ORDER])
    ax_status.set_xlabel("Cantidad de cajas")

    for i, heuristic in enumerate(HEURISTIC_ORDER):
        for j, board_path in enumerate(BOARDS):
            box_count = len(parse_board(board_path)[1].boxes)
            match = next(
                row
                for row in rows
                if row["heuristic"] == heuristic and int(row["box_count"]) == box_count
            )
            text = (
                "OK"
                if bool(match["success"]) and not bool(match["timed_out"])
                else ("T/O" if bool(match["timed_out"]) else "FAIL")
            )
            ax_status.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#1F1F1F",
            )

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.89),
    )
    fig.suptitle(
        "Heuristicas vs cantidad de cajas (A*)", fontsize=18, weight="bold", y=0.98
    )
    fig.text(
        0.5,
        0.945,
        "Familia controlada de tableros abiertos: mismo patron, mas cajas y metas a medida que crece el problema.",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.text(
        0.5,
        0.02,
        "Dead Square se evalua como heuristica standalone (0 / inf), por eso actua mas como poda que como estimador de costo.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estudio de heuristicas segun cantidad de cajas"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Timeout por corrida en segundos (default: 12)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "benchmark" / "heuristics_by_boxcount_astar.csv",
        help="CSV de salida para las metricas",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "results" / "plots" / "heuristics_by_boxcount_astar.png",
        help="PNG de salida para el grafico compuesto",
    )
    args = parser.parse_args()

    rows = collect_rows(args.timeout)
    csv_path = write_csv(rows, args.csv)
    plot_path = build_plot(rows, args.timeout, args.plot)
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
