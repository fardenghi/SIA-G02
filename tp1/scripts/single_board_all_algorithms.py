#!/usr/bin/env python3
"""Benchmark de un solo tablero para todos los algoritmos.

Corre el mismo tablero multiples veces para comparar:
  - BFS, DFS e IDDFS
  - Greedy y A* con heuristicas individuales
  - combinaciones utiles con dead_square usando max(h, dead_square)

Uso:
    uv run python scripts/single_board_all_algorithms.py
    uv run python scripts/single_board_all_algorithms.py --board boards/sokoban/level_04.txt
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
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.search.astar import astar  # noqa: E402
from tp1_search.search.bfs import bfs  # noqa: E402
from tp1_search.search.dfs import dfs  # noqa: E402
from tp1_search.search.greedy import greedy  # noqa: E402
from tp1_search.search.iddfs import iddfs  # noqa: E402
from tp1_search.sokoban.heuristics import (  # noqa: E402
    combine_heuristics_max,
    dead_square_heuristic,
    euclidean_heuristic,
    hungarian_heuristic,
    manhattan_heuristic,
    weighted_hungarian_heuristic,
)
from tp1_search.sokoban.parser import parse_board  # noqa: E402

DEFAULT_BOARD = ROOT / "boards" / "sokoban" / "level_02.txt"

UNINFORMED = {
    "bfs": bfs,
    "dfs": dfs,
    "iddfs": iddfs,
}

INFORMED = {
    "greedy": greedy,
    "astar": astar,
}

# Solo se incluyen combinaciones que aportan algo real:
# - dead_square con manhattan, euclidean y weighted_hungarian
# - no se incluye manhattan+euclidean porque max(manhattan, euclidean) colapsa a manhattan
# - no se incluye hungarian+dead_square porque hungarian ya poda dead squares internamente
HEURISTIC_VARIANTS = {
    "manhattan": manhattan_heuristic,
    "manhattan+dead_square": combine_heuristics_max(
        [manhattan_heuristic, dead_square_heuristic]
    ),
    "euclidean": euclidean_heuristic,
    "euclidean+dead_square": combine_heuristics_max(
        [euclidean_heuristic, dead_square_heuristic]
    ),
    "hungarian": hungarian_heuristic,
    "weighted_hungarian": weighted_hungarian_heuristic,
    "weighted_hungarian+dead_square": combine_heuristics_max(
        [weighted_hungarian_heuristic, dead_square_heuristic]
    ),
}

HEURISTIC_ORDER = [
    "manhattan",
    "manhattan+dead_square",
    "euclidean",
    "euclidean+dead_square",
    "hungarian",
    "weighted_hungarian",
    "weighted_hungarian+dead_square",
]

HEURISTIC_LABELS = {
    "manhattan": "Manhattan",
    "manhattan+dead_square": "Manhattan + DS",
    "euclidean": "Euclidean",
    "euclidean+dead_square": "Euclidean + DS",
    "hungarian": "Hungarian",
    "weighted_hungarian": "Weighted Hung.",
    "weighted_hungarian+dead_square": "Weighted Hung. + DS",
}

HEURISTIC_COLORS = {
    "manhattan": "#2A9D8F",
    "manhattan+dead_square": "#2A9D8F",
    "euclidean": "#4C78A8",
    "euclidean+dead_square": "#4C78A8",
    "hungarian": "#E9C46A",
    "weighted_hungarian": "#E76F51",
    "weighted_hungarian+dead_square": "#E76F51",
}

HEURISTIC_HATCHES = {
    "manhattan": "",
    "manhattan+dead_square": "//",
    "euclidean": "",
    "euclidean+dead_square": "//",
    "hungarian": "",
    "weighted_hungarian": "",
    "weighted_hungarian+dead_square": "//",
}

UNINFORMED_COLORS = {
    "bfs": "#5C677D",
    "dfs": "#7D8597",
    "iddfs": "#2B2D42",
}

ALGORITHM_EDGE_COLORS = {
    "bfs": "#1F1F1F",
    "dfs": "#1F1F1F",
    "iddfs": "#1F1F1F",
    "greedy": "#1F5C57",
    "astar": "#8C3B30",
}

CSV_COLUMNS = [
    "board",
    "run",
    "algorithm",
    "heuristic",
    "label",
    "success",
    "cost",
    "expanded_nodes",
    "frontier_nodes",
    "time_elapsed",
    "timed_out",
]


@dataclass(frozen=True)
class ExperimentSpec:
    algorithm: str
    heuristic: str | None
    key: str
    console_label: str
    plot_label: str


class RunData(TypedDict):
    success: bool
    cost: int
    expanded_nodes: int
    frontier_nodes: int
    time_elapsed: float
    timed_out: bool


class RowData(RunData):
    board: str
    run: int
    algorithm: str
    heuristic: str
    label: str


class SummaryData(TypedDict):
    success_rate: float
    timeout_rate: float
    cost_mean: float
    cost_std: float
    nodes_mean: float
    nodes_std: float
    time_mean: float
    time_std: float


def _build_experiments() -> list[ExperimentSpec]:
    experiments = [
        ExperimentSpec(
            algorithm="bfs",
            heuristic=None,
            key="bfs",
            console_label="BFS",
            plot_label="BFS",
        ),
        ExperimentSpec(
            algorithm="dfs",
            heuristic=None,
            key="dfs",
            console_label="DFS",
            plot_label="DFS",
        ),
        ExperimentSpec(
            algorithm="iddfs",
            heuristic=None,
            key="iddfs",
            console_label="IDDFS",
            plot_label="IDDFS",
        ),
    ]

    for algorithm, algorithm_label in (("greedy", "Greedy"), ("astar", "A*")):
        for heuristic in HEURISTIC_ORDER:
            experiments.append(
                ExperimentSpec(
                    algorithm=algorithm,
                    heuristic=heuristic,
                    key=f"{algorithm}:{heuristic}",
                    console_label=f"{algorithm_label} + {HEURISTIC_LABELS[heuristic]}",
                    plot_label=f"{algorithm_label}\n{HEURISTIC_LABELS[heuristic]}",
                )
            )

    return experiments


EXPERIMENTS = _build_experiments()


def _run_single(
    board_path: str,
    algorithm: str,
    heuristic_name: str | None,
    queue: mp.Queue,
) -> None:
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        board, state = parse_board(board_path)

        if heuristic_name is None:
            result = UNINFORMED[algorithm](board, state)
        else:
            result = INFORMED[algorithm](
                board, state, HEURISTIC_VARIANTS[heuristic_name]
            )

    queue.put(
        {
            "success": result.success,
            "cost": result.cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "time_elapsed": result.time_elapsed,
        }
    )


def run_with_timeout(
    board_path: Path,
    spec: ExperimentSpec,
    timeout: float,
) -> RunData:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_run_single,
        args=(str(board_path), spec.algorithm, spec.heuristic, queue),
    )
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


def collect_rows(board_path: Path, runs: int, timeout: float) -> list[RowData]:
    rows: list[RowData] = []

    for run_idx in range(1, runs + 1):
        print(f"\n=== Corrida {run_idx}/{runs} ===")

        for item_idx, spec in enumerate(EXPERIMENTS, start=1):
            print(
                f"[{item_idx:02d}/{len(EXPERIMENTS):02d}] {spec.console_label:30s} ...",
                end=" ",
                flush=True,
            )
            data = run_with_timeout(board_path, spec, timeout)
            status = (
                "TIMEOUT"
                if data["timed_out"]
                else ("OK" if data["success"] else "FAIL")
            )
            print(
                f"{status:7s} expanded={data['expanded_nodes']:>8} time={float(data['time_elapsed']):.4f}s"
            )
            rows.append(
                {
                    "board": board_path.stem,
                    "run": run_idx,
                    "algorithm": spec.algorithm,
                    "heuristic": spec.heuristic or "-",
                    "label": spec.key,
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


def summarize_rows(rows: list[RowData]) -> dict[str, SummaryData]:
    summary: dict[str, SummaryData] = {}

    for spec in EXPERIMENTS:
        subset = [row for row in rows if row["label"] == spec.key]
        successes = [row for row in subset if row["success"] and not row["timed_out"]]

        if successes:
            costs = [float(row["cost"]) for row in successes]
            nodes = [float(row["expanded_nodes"]) for row in successes]
            times = [float(row["time_elapsed"]) for row in successes]
            cost_mean = statistics.fmean(costs)
            cost_std = statistics.pstdev(costs) if len(costs) > 1 else 0.0
            nodes_mean = statistics.fmean(nodes)
            nodes_std = statistics.pstdev(nodes) if len(nodes) > 1 else 0.0
            time_mean = statistics.fmean(times)
            time_std = statistics.pstdev(times) if len(times) > 1 else 0.0
        else:
            cost_mean = math.nan
            cost_std = math.nan
            nodes_mean = math.nan
            nodes_std = math.nan
            time_mean = math.nan
            time_std = math.nan

        summary[spec.key] = {
            "success_rate": len(successes) / len(subset) if subset else 0.0,
            "timeout_rate": (
                sum(1 for row in subset if row["timed_out"]) / len(subset)
                if subset
                else 0.0
            ),
            "cost_mean": cost_mean,
            "cost_std": cost_std,
            "nodes_mean": nodes_mean,
            "nodes_std": nodes_std,
            "time_mean": time_mean,
            "time_std": time_std,
        }

    return summary


def _bar_style(spec: ExperimentSpec) -> dict[str, str]:
    if spec.heuristic is None:
        return {
            "facecolor": UNINFORMED_COLORS[spec.algorithm],
            "edgecolor": ALGORITHM_EDGE_COLORS[spec.algorithm],
            "hatch": "",
        }

    return {
        "facecolor": HEURISTIC_COLORS[spec.heuristic],
        "edgecolor": ALGORITHM_EDGE_COLORS[spec.algorithm],
        "hatch": HEURISTIC_HATCHES[spec.heuristic],
    }


def _add_group_bands(ax: plt.Axes) -> None:
    groups = [
        (-0.5, 2.5, "No informados", "#F2EFEA"),
        (2.5, 9.5, "Greedy", "#EAF7F4"),
        (9.5, 16.5, "A*", "#FCEFE8"),
    ]

    for start, end, label, color in groups:
        ax.axvspan(start, end, color=color, alpha=0.75, zorder=0)
        ax.text(
            (start + end) / 2,
            1.01,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color="#3A3A3A",
        )


def _positive_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value) and value > 0]


def build_plot(rows: list[RowData], board_path: Path, outpath: Path, runs: int) -> Path:
    summary = summarize_rows(rows)
    x_positions = list(range(len(EXPERIMENTS)))
    labels = [spec.plot_label for spec in EXPERIMENTS]
    node_means = [summary[spec.key]["nodes_mean"] for spec in EXPERIMENTS]
    node_stds = [summary[spec.key]["nodes_std"] for spec in EXPERIMENTS]
    time_means = [summary[spec.key]["time_mean"] for spec in EXPERIMENTS]
    time_stds = [summary[spec.key]["time_std"] for spec in EXPERIMENTS]

    fig, (ax_nodes, ax_time) = plt.subplots(
        2,
        1,
        figsize=(19, 11),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
        facecolor="white",
    )

    for ax in (ax_nodes, ax_time):
        _add_group_bands(ax)
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=1)
        ax.set_axisbelow(True)

    for index, spec in enumerate(EXPERIMENTS):
        style = _bar_style(spec)

        ax_nodes.bar(
            x_positions[index],
            node_means[index],
            width=0.72,
            yerr=node_stds[index],
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=1.8,
            hatch=style["hatch"],
            ecolor="#1F1F1F",
            capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
            zorder=3,
        )
        ax_time.bar(
            x_positions[index],
            time_means[index],
            width=0.72,
            yerr=time_stds[index],
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=1.8,
            hatch=style["hatch"],
            ecolor="#1F1F1F",
            capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
            zorder=3,
        )

    node_positive = _positive_values(node_means)
    time_positive = _positive_values(time_means)

    if node_positive:
        ax_nodes.set_yscale("log")
        ax_nodes.set_ylim(min(node_positive) * 0.75, max(node_positive) * 1.45)
    if time_positive:
        ax_time.set_yscale("log")
        ax_time.set_ylim(min(time_positive) * 0.7, max(time_positive) * 1.6)

    ax_nodes.set_title("Nodos expandidos", fontsize=16, weight="bold")
    ax_nodes.set_ylabel("Media +/- sigma", fontsize=11)
    ax_time.set_title("Tiempo de ejecucion", fontsize=16, weight="bold")
    ax_time.set_ylabel("Media +/- sigma (s)", fontsize=11)

    ax_time.set_xticks(x_positions)
    ax_time.set_xticklabels(labels, rotation=32, ha="right", fontsize=10)

    legend_handles = [
        Patch(facecolor="#F5F5F5", edgecolor="#1F1F1F", label="Sin heuristica"),
        Patch(facecolor="#2A9D8F", edgecolor="#1F5C57", label="Manhattan"),
        Patch(facecolor="#4C78A8", edgecolor="#1F5C57", label="Euclidean"),
        Patch(facecolor="#E9C46A", edgecolor="#1F5C57", label="Hungarian"),
        Patch(facecolor="#E76F51", edgecolor="#1F5C57", label="Weighted Hung."),
        Patch(
            facecolor="#FFFFFF", edgecolor="#4A4A4A", hatch="//", label="+ Dead Square"
        ),
    ]
    ax_nodes.legend(
        handles=legend_handles,
        ncol=6,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        frameon=False,
    )

    fig.suptitle(
        f"Comparativa en un mismo tablero ({runs} corridas)",
        fontsize=20,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.02,
        (
            f"Tablero: {board_path.name} | barras de error = sigma | "
            "combos con DS usan max(h, dead_square)"
        ),
        ha="center",
        fontsize=10,
        color="#5A5A5A",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.19)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def print_summary(rows: list[RowData]) -> None:
    summary = summarize_rows(rows)
    print("\n=== Resumen (media de tiempo) ===")
    for spec in sorted(EXPERIMENTS, key=lambda item: summary[item.key]["time_mean"]):
        time_mean = summary[spec.key]["time_mean"]
        nodes_mean = summary[spec.key]["nodes_mean"]
        success_rate = summary[spec.key]["success_rate"]
        if not math.isfinite(time_mean):
            print(f"- {spec.console_label:30s} | sin corridas exitosas")
            continue
        print(
            f"- {spec.console_label:30s} | time={time_mean:.4f}s | nodes={int(round(nodes_mean)):>8} | success={success_rate:.0%}"
        )


def _default_csv_path(board_path: Path) -> Path:
    return ROOT / "results" / "benchmark" / f"{board_path.stem}_all_algorithms.csv"


def _default_plot_path(board_path: Path) -> Path:
    return ROOT / "results" / "plots" / f"{board_path.stem}_all_algorithms.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara todos los algoritmos sobre un mismo tablero"
    )
    parser.add_argument(
        "--board",
        type=Path,
        default=DEFAULT_BOARD,
        help="Tablero a ejecutar para todas las corridas",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Cantidad de corridas por configuracion (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout por ejecucion en segundos (default: 30)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV de salida con las corridas crudas",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="PNG de salida para la figura",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs debe ser >= 1")
    if args.timeout <= 0:
        raise ValueError("--timeout debe ser > 0")

    parse_board(args.board)

    csv_path = args.csv or _default_csv_path(args.board)
    plot_path = args.plot or _default_plot_path(args.board)

    print("=== Benchmark de un tablero ===")
    print(f"Tablero:   {args.board}")
    print(f"Corridas:  {args.runs}")
    print(f"Configs:   {len(EXPERIMENTS)}")
    print(f"Timeout:   {args.timeout}s")
    print(f"CSV:       {csv_path}")
    print(f"Plot:      {plot_path}")

    rows = collect_rows(args.board, args.runs, args.timeout)
    write_csv(rows, csv_path)
    build_plot(rows, args.board, plot_path, args.runs)
    print_summary(rows)

    print(f"\nCSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
