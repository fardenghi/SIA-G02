#!/usr/bin/env python3
"""Benchmark y graficos de heuristicas vs cantidad de cajas.

Uso:
    uv run python scripts/box_count_traps_plots.py
    uv run python scripts/box_count_traps_plots.py --csv-only
    uv run python scripts/box_count_traps_plots.py --plot-only
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tp1_search.search.greedy import greedy  # noqa: E402
from tp1_search.sokoban.heuristics import (  # noqa: E402
    euclidean_heuristic,
    hungarian_heuristic,
    manhattan_heuristic,
    weighted_hungarian_heuristic,
)
from tp1_search.sokoban.parser import parse_board  # noqa: E402

BOARD_DIR = ROOT / "boards" / "sokoban" / "box_count_traps"
BOARDS = sorted(board for board in BOARD_DIR.glob("boxes_*.txt"))

HEURISTICS = {
    "manhattan": manhattan_heuristic,
    "euclidean": euclidean_heuristic,
    "hungarian": hungarian_heuristic,
    "weighted_hungarian": weighted_hungarian_heuristic,
}

HEURISTIC_ORDER = [
    "manhattan",
    "euclidean",
    "hungarian",
    "weighted_hungarian",
]

HEURISTIC_LABELS = {
    "manhattan": "Manhattan",
    "euclidean": "Euclidean",
    "hungarian": "Hungarian",
    "weighted_hungarian": "Weighted Hung.",
}

HEURISTIC_COLORS = {
    "manhattan": "#2A9D8F",
    "euclidean": "#4C78A8",
    "hungarian": "#E9C46A",
    "weighted_hungarian": "#E76F51",
}

CSV_COLUMNS = [
    "board",
    "box_count",
    "heuristic",
    "run",
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
    run: int


def _run_single(board_path: str, heuristic_name: str, queue: mp.Queue) -> None:
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        board, state = parse_board(board_path)
        result = greedy(board, state, HEURISTICS[heuristic_name])

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


def collect_rows(timeout: float, runs: int) -> list[RowData]:
    rows: list[RowData] = []

    for run_idx in range(1, runs + 1):
        print(f"\n=== Corrida {run_idx}/{runs} ===")

        for board_path in BOARDS:
            board, state = parse_board(board_path)
            box_count = len(state.boxes)

            for heuristic in HEURISTIC_ORDER:
                print(
                    f"[{board_path.stem}] run {run_idx}/{runs} | Greedy + {heuristic:18s} ...",
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
                    f"{status:7s} expanded={data['expanded_nodes']:>8} time={float(data['time_elapsed']):.3f}s"
                )
                rows.append(
                    {
                        "board": board_path.stem,
                        "box_count": box_count,
                        "heuristic": heuristic,
                        "run": run_idx,
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


def load_csv(path: Path) -> list[RowData]:
    rows: list[RowData] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "board": row["board"],
                    "box_count": int(row["box_count"]),
                    "heuristic": row["heuristic"],
                    "run": int(row["run"]),
                    "success": row["success"] == "True",
                    "cost": int(row["cost"]),
                    "expanded_nodes": int(row["expanded_nodes"]),
                    "frontier_nodes": int(row["frontier_nodes"]),
                    "time_elapsed": float(row["time_elapsed"]),
                    "timed_out": row["timed_out"] == "True",
                }
            )
    return rows


def _series(rows: list[RowData], heuristic: str) -> list[RowData]:
    return sorted(
        [row for row in rows if row["heuristic"] == heuristic],
        key=lambda row: row["box_count"],
    )


def _compute_time_ranges(
    series_stats: dict[str, tuple[list[float], list[float], list[int]]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    all_points: list[tuple[float, float]] = []
    for heuristic in HEURISTIC_ORDER:
        means, stds, _ = series_stats[heuristic]
        for mean, std in zip(means, stds, strict=False):
            if not math.isnan(mean) and mean > 0:
                all_points.append((mean, std))

    sorted_means = sorted(mean for mean, _ in all_points)
    ratios = [
        sorted_means[i + 1] / sorted_means[i] for i in range(len(sorted_means) - 1)
    ]

    if len(ratios) >= 2:
        split_indices = sorted(
            sorted(range(len(ratios)), key=lambda i: ratios[i], reverse=True)[:2]
        )
        low_gap_idx, high_gap_idx = split_indices[0], split_indices[1]
    elif len(ratios) == 1:
        low_gap_idx = 0
        high_gap_idx = 0
    else:
        low_gap_idx = 0
        high_gap_idx = 0

    low_cluster_max = sorted_means[low_gap_idx]
    mid_cluster_min = sorted_means[low_gap_idx + 1]
    mid_cluster_max = sorted_means[high_gap_idx]
    high_cluster_min = sorted_means[high_gap_idx + 1]

    low_points = [(m, s) for m, s in all_points if m <= low_cluster_max]
    mid_points = [
        (m, s) for m, s in all_points if mid_cluster_min <= m <= mid_cluster_max
    ]
    high_points = [(m, s) for m, s in all_points if m >= high_cluster_min]

    low_range = (0.0, max(m + s for m, s in low_points) * 1.35)
    mid_range = (
        min(max(m - s, 0.0) for m, s in mid_points) * 0.9,
        max(m + s for m, s in mid_points) * 1.16,
    )
    high_range = (
        min(max(m - s, 0.0) for m, s in high_points) * 0.95,
        max(m + s for m, s in high_points) * 1.08,
    )

    return low_range, mid_range, high_range


def _mask_band(
    means: list[float], stds: list[float], lower: float, upper: float
) -> tuple[list[float | None], list[float | None]]:
    masked_means: list[float | None] = []
    masked_stds: list[float | None] = []
    for mean, std in zip(means, stds, strict=False):
        if math.isnan(mean) or not (lower <= mean <= upper):
            masked_means.append(None)
            masked_stds.append(None)
        else:
            masked_means.append(mean)
            masked_stds.append(std)
    return masked_means, masked_stds


def _nice_step(value: float) -> float:
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)

    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3.5:
        nice_fraction = 2
    elif fraction < 7.5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * (10**exponent)


def _next_nice_step(step: float) -> float:
    exponent = math.floor(math.log10(step))
    fraction = step / (10**exponent)

    if fraction < 1.5:
        return 2 * (10**exponent)
    if fraction < 3.5:
        return 5 * (10**exponent)
    return 10 ** (exponent + 1)


def _power_tick_labels(lower: float, upper: float) -> tuple[list[float], list[str]]:
    ratio = upper / lower

    if ratio <= 10:
        step = _nice_step((upper - lower) / 3)

        while True:
            start = math.floor(lower / step) * step
            if start < lower:
                start += step

            vals: list[float] = []
            value = start
            while value <= upper * (1 + 1e-9):
                vals.append(value)
                value += step

            if 2 <= len(vals) <= 4:
                break

            if len(vals) < 2:
                vals = [lower, upper]
                break

            step = _next_nice_step(step)
    else:
        vals = []
        for mantissas in ([1], [1, 5], [1, 2, 5]):
            candidates: list[float] = []
            min_exp = math.floor(math.log10(lower)) - 1
            max_exp = math.ceil(math.log10(upper)) + 1

            for exp in range(min_exp, max_exp + 1):
                scale = 10**exp
                for mantissa in mantissas:
                    value = mantissa * scale
                    if lower <= value <= upper:
                        candidates.append(value)

            if 2 <= len(candidates) <= 4:
                vals = candidates
                break

            if not vals and candidates:
                vals = candidates

        if len(vals) > 4:
            stride = math.ceil(len(vals) / 4)
            vals = vals[::stride]

    texts = [f"{value:.2g}" for value in vals]
    return vals, texts


def _plot_time_metric_plotly(
    series_stats: dict[str, tuple[list[float], list[float], list[int]]],
    box_counts: list[int],
    title: str,
    outpath: Path,
    timeout: float,
    runs: int,
) -> Path:
    fig = make_subplots(
        rows=1,
        cols=len(box_counts),
        subplot_titles=[f"{box_count} cajas" for box_count in box_counts],
        horizontal_spacing=0.06,
    )

    for col_idx, box_count in enumerate(box_counts, start=1):
        panel_values: list[tuple[float, float]] = []

        for heuristic in HEURISTIC_ORDER:
            means, stds, _ = series_stats[heuristic]
            idx = box_counts.index(box_count)
            mean = means[idx]
            std = stds[idx]
            panel_values.append((mean, std))

            fig.add_trace(
                go.Bar(
                    x=[HEURISTIC_LABELS[heuristic]],
                    y=[mean],
                    name=HEURISTIC_LABELS[heuristic],
                    legendgroup=heuristic,
                    showlegend=col_idx == 1,
                    marker=dict(color=HEURISTIC_COLORS[heuristic]),
                    error_y=dict(
                        type="data",
                        array=[std],
                        symmetric=True,
                        color="#202020",
                        thickness=2,
                        width=8,
                        visible=True,
                    ),
                    hovertemplate=(
                        f"cajas={box_count}<br>heuristica={HEURISTIC_LABELS[heuristic]}"
                        + "<br>tiempo medio=%{y:.8f}s<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        min_value = min(max(mean - std, 1e-6) for mean, std in panel_values)
        max_value = max(mean + std for mean, std in panel_values)
        tickvals, ticktext = _power_tick_labels(min_value * 0.8, max_value * 1.25)
        fig.update_yaxes(
            type="log",
            range=[math.log10(min_value * 0.8), math.log10(max_value * 1.25)],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            gridcolor="rgba(180,180,180,0.25)",
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor="#2F2F2F",
            ticks="outside",
            tickwidth=2,
            ticklen=8,
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(
            showline=True,
            linewidth=2,
            linecolor="#2F2F2F",
            ticks="outside",
            tickwidth=2,
            ticklen=8,
            tickangle=-20,
            row=1,
            col=col_idx,
        )

    fig.update_yaxes(title_text="Tiempo (s)", row=1, col=1)
    fig.update_layout(
        title=dict(
            text=f"{title} (media ± σ, {runs} corridas)", x=0.5, xanchor="center"
        ),
        template="plotly_white",
        width=1700,
        height=850,
        font=dict(size=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(l=90, r=40, t=130, b=120),
        barmode="group",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outpath.with_suffix(".html"))
    fig.write_image(outpath)
    return outpath


def _plot_metric(
    rows: list[RowData],
    metric: str,
    ylabel: str,
    title: str,
    outpath: Path,
    timeout: float,
    runs: int,
) -> Path:
    box_counts = sorted({row["box_count"] for row in rows})
    timeout_offsets = {
        heuristic: (idx - (len(HEURISTIC_ORDER) - 1) / 2) * 0.045
        for idx, heuristic in enumerate(HEURISTIC_ORDER)
    }
    positive_values: list[float] = []
    series_stats: dict[str, tuple[list[float], list[float], list[int]]] = {}

    for heuristic in HEURISTIC_ORDER:
        means: list[float] = []
        stds: list[float] = []
        timeout_x: list[int] = []

        for box_count in box_counts:
            subset = [
                row
                for row in rows
                if row["heuristic"] == heuristic and row["box_count"] == box_count
            ]

            if metric == "time_elapsed":
                samples = [float(row[metric]) for row in subset]
                means.append(statistics.fmean(samples))
                stds.append(statistics.pstdev(samples) if len(samples) > 1 else 0.0)
                positive_values.extend(value for value in samples if value > 0)
                if all(row["timed_out"] for row in subset):
                    timeout_x.append(box_count)
            else:
                samples = [
                    float(row[metric])
                    for row in subset
                    if not row["timed_out"] and float(row[metric]) > 0
                ]
                if samples:
                    means.append(statistics.fmean(samples))
                    stds.append(statistics.pstdev(samples) if len(samples) > 1 else 0.0)
                    positive_values.extend(samples)
                else:
                    means.append(math.nan)
                    stds.append(math.nan)
                    timeout_x.append(box_count)

        series_stats[heuristic] = (means, stds, timeout_x)

    if metric == "time_elapsed":
        return _plot_time_metric_plotly(
            series_stats, box_counts, title, outpath, timeout, runs
        )

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.patch.set_facecolor("white")

    for heuristic in HEURISTIC_ORDER:
        means, stds, _ = series_stats[heuristic]
        if metric == "time_elapsed":
            ax.errorbar(
                box_counts,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.4,
                markersize=6.8,
                capsize=5,
                elinewidth=1.5,
                color=HEURISTIC_COLORS[heuristic],
                label=HEURISTIC_LABELS[heuristic],
            )
        else:
            ax.plot(
                box_counts,
                means,
                marker="o",
                linewidth=2.4,
                markersize=6.8,
                color=HEURISTIC_COLORS[heuristic],
                label=HEURISTIC_LABELS[heuristic],
            )

    if metric == "expanded_nodes":
        for heuristic in HEURISTIC_ORDER:
            timeout_x = [
                row["box_count"]
                for row in _series(rows, heuristic)
                if row["timed_out"]
            ]
            if timeout_x:
                ymin, ymax = ax.get_ylim()
                timeout_y = ymax * 0.85
                ax.scatter(
                    [x + timeout_offsets[heuristic] for x in timeout_x],
                    [timeout_y] * len(timeout_x),
                    marker="X",
                    s=86,
                    color=HEURISTIC_COLORS[heuristic],
                    zorder=4,
                )
    else:
        ax.axhline(timeout, color="#999999", linestyle=":", linewidth=1.3)

    if metric == "time_elapsed":
        title = f"{title} (media ± sigma, {runs} corridas)"

    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("Cantidad de cajas")
    ax.set_ylabel(ylabel)
    ax.set_xticks(box_counts)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=10)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graficos de heuristicas vs cantidad de cajas sobre tableros con trampas"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Timeout por corrida en segundos (default: 12)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Cantidad de corridas por tablero/heuristica para el calculo de variacion (default: 5)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT
        / "results"
        / "benchmark"
        / "heuristics_by_boxcount_traps_greedy.csv",
        help="CSV de salida",
    )
    parser.add_argument(
        "--nodes-plot",
        type=Path,
        default=ROOT / "results" / "plots" / "heuristics_by_boxcount_nodes.png",
        help="PNG de salida para nodos expandidos",
    )
    parser.add_argument(
        "--time-plot",
        type=Path,
        default=ROOT / "results" / "plots" / "heuristics_by_boxcount_time.png",
        help="PNG de salida para tiempo",
    )
    parser.add_argument(
        "--cost-plot",
        type=Path,
        default=ROOT / "results" / "plots" / "heuristics_by_boxcount_cost.png",
        help="PNG de salida para costo de solucion",
    )
    parser.add_argument(
        "--frontier-plot",
        type=Path,
        default=ROOT / "results" / "plots" / "heuristics_by_boxcount_frontier.png",
        help="PNG de salida para nodos frontera",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Solo genera el CSV de benchmark; no regenera los graficos",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Solo regenera los graficos a partir del CSV existente",
    )
    args = parser.parse_args()

    if args.csv_only and args.plot_only:
        raise SystemExit("No se puede usar --csv-only y --plot-only al mismo tiempo")

    if args.plot_only:
        rows = load_csv(args.csv)
        csv_path = args.csv
    else:
        rows = collect_rows(args.timeout, args.runs)
        csv_path = write_csv(rows, args.csv)

    if args.csv_only:
        print(f"CSV:   {csv_path}")
        return

    nodes_path = _plot_metric(
        rows,
        metric="expanded_nodes",
        ylabel="Nodos expandidos",
        title="Greedy: nodos expandidos vs cantidad de cajas",
        outpath=args.nodes_plot,
        timeout=args.timeout,
        runs=args.runs,
    )
    time_path = _plot_metric(
        rows,
        metric="time_elapsed",
        ylabel="Tiempo (s)",
        title="Greedy: tiempo vs cantidad de cajas",
        outpath=args.time_plot,
        timeout=args.timeout,
        runs=args.runs,
    )
    cost_path = _plot_metric(
        rows,
        metric="cost",
        ylabel="Costo (pasos)",
        title="Greedy: costo de solución vs cantidad de cajas",
        outpath=args.cost_plot,
        timeout=args.timeout,
        runs=args.runs,
    )
    frontier_path = _plot_metric(
        rows,
        metric="frontier_nodes",
        ylabel="Nodos frontera (max)",
        title="Greedy: nodos frontera vs cantidad de cajas",
        outpath=args.frontier_plot,
        timeout=args.timeout,
        runs=args.runs,
    )

    print(f"CSV:      {csv_path}")
    print(f"Nodos:    {nodes_path}")
    print(f"Tiempo:   {time_path}")
    print(f"Costo:    {cost_path}")
    print(f"Frontera: {frontier_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
