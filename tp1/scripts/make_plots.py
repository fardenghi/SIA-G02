#!/usr/bin/env python3
"""Visualización de resultados del benchmark.

Uso:
    uv run python scripts/make_plots.py [--input PATH] [--outdir PATH]

Lee el CSV generado por run_batch.py y produce:
  1. Barras agrupadas: nodos expandidos por algoritmo, por tablero  (PNG)
  2. Barras agrupadas: tiempo de ejecución por algoritmo, por tablero (PNG)
  3. Barras agrupadas: costo de la solución por algoritmo, por tablero (PNG)
  4. Heatmap: algoritmo × tablero → nodos expandidos  (PNG)
  5. Comparación de heurísticas para A* y Greedy (PNG)
  6. Dashboard interactivo con plotly (HTML)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent


# ── helpers ──────────────────────────────────────────────────────────────


def algo_label(row: pd.Series) -> str:
    """Human-readable label: 'A* (manhattan)' or 'BFS'."""
    h = row["heuristic"]
    a = row["algorithm"]
    if h != "-":
        return f"{a.upper()} ({h})"
    return a.upper()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df.apply(algo_label, axis=1)
    # Replace -1 (timeout/crash) with NaN for plotting
    df.loc[df["expanded_nodes"] < 0, "expanded_nodes"] = float("nan")
    df.loc[df["frontier_nodes"] < 0, "frontier_nodes"] = float("nan")
    return df


# ── matplotlib plots ─────────────────────────────────────────────────────


def _grouped_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    outpath: Path,
    log_scale: bool = False,
) -> None:
    """Generic grouped bar chart: one group per board, bars per algorithm config."""
    boards = df["board"].unique()
    labels = df["label"].unique()
    n_boards = len(boards)
    n_labels = len(labels)

    x = np.arange(n_boards)
    width = 0.8 / n_labels

    fig, ax = plt.subplots(figsize=(max(10, n_boards * 1.5), 6))

    for i, label in enumerate(labels):
        vals = []
        for board in boards:
            subset = df[(df["board"] == board) & (df["label"] == label)]
            if len(subset) == 1:
                v = subset[metric].iloc[0]
                vals.append(v if not pd.isna(v) else 0)
            else:
                vals.append(0)

        offset = (i - n_labels / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label)

        # Annotate timeouts / failures
        for j, board in enumerate(boards):
            subset = df[(df["board"] == board) & (df["label"] == label)]
            if len(subset) == 1:
                row = subset.iloc[0]
                if row["timed_out"]:
                    ax.text(
                        x[j] + offset,
                        vals[j],
                        "T/O",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="red",
                    )
                elif not row["success"] and not row["timed_out"]:
                    ax.text(
                        x[j] + offset,
                        vals[j],
                        "FAIL",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="orange",
                    )

    ax.set_xlabel("Tablero")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(boards, rotation=45, ha="right")
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_expanded(df: pd.DataFrame, outdir: Path) -> None:
    _grouped_bar(
        df,
        "expanded_nodes",
        "Nodos expandidos",
        "Nodos expandidos por algoritmo y tablero",
        outdir / "expanded_nodes.png",
        log_scale=True,
    )


def plot_time(df: pd.DataFrame, outdir: Path) -> None:
    _grouped_bar(
        df,
        "time_elapsed",
        "Tiempo (s)",
        "Tiempo de ejecución por algoritmo y tablero",
        outdir / "time_elapsed.png",
        log_scale=True,
    )


def plot_cost(df: pd.DataFrame, outdir: Path) -> None:
    _grouped_bar(
        df,
        "cost",
        "Costo (pasos)",
        "Costo de la solución por algoritmo y tablero",
        outdir / "solution_cost.png",
        log_scale=False,
    )


def plot_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """Heatmap: algorithm config × board → expanded nodes."""
    pivot = df.pivot_table(
        index="label", columns="board", values="expanded_nodes", aggfunc="first"
    )

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.2), max(5, len(pivot.index) * 0.5))
    )
    data = pivot.values.astype(float)

    # Use log scale for color
    import matplotlib.colors as mcolors

    norm = mcolors.LogNorm(vmin=max(1, np.nanmin(data[data > 0])), vmax=np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            if np.isnan(val):
                text = "T/O"
                color = "red"
            else:
                text = f"{int(val):,}"
                color = "white" if val > np.nanmedian(data) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Nodos expandidos (log)")
    ax.set_title("Heatmap: nodos expandidos por algoritmo y tablero")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_expanded.png", dpi=150)
    plt.close(fig)
    print(f"  -> {outdir / 'heatmap_expanded.png'}")


def plot_heuristic_comparison(df: pd.DataFrame, outdir: Path) -> None:
    """Compare heuristics head-to-head for A* and Greedy separately."""
    for algo in ["astar", "greedy"]:
        sub = df[
            (df["algorithm"] == algo) & (df["success"] == True) & (~df["timed_out"])
        ].copy()
        if sub.empty:
            continue

        boards = sub["board"].unique()
        heuristics = sub["heuristic"].unique()
        n_boards = len(boards)
        n_h = len(heuristics)
        x = np.arange(n_boards)
        width = 0.8 / n_h

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for i, h in enumerate(heuristics):
            h_sub = sub[sub["heuristic"] == h]
            exp_vals = [
                h_sub[h_sub["board"] == b]["expanded_nodes"].values[0]
                if len(h_sub[h_sub["board"] == b])
                else 0
                for b in boards
            ]
            time_vals = [
                h_sub[h_sub["board"] == b]["time_elapsed"].values[0]
                if len(h_sub[h_sub["board"] == b])
                else 0
                for b in boards
            ]
            offset = (i - n_h / 2 + 0.5) * width
            ax1.bar(x + offset, exp_vals, width, label=h)
            ax2.bar(x + offset, time_vals, width, label=h)

        for ax, ylabel, title_metric in [
            (ax1, "Nodos expandidos", "expandidos"),
            (ax2, "Tiempo (s)", "tiempo"),
        ]:
            ax.set_xlabel("Tablero")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{algo.upper()}: nodos {title_metric} por heurística")
            ax.set_xticks(x)
            ax.set_xticklabels(boards, rotation=45, ha="right")
            ax.set_yscale("log")
            ax.legend()

        fig.tight_layout()
        outfile = outdir / f"heuristic_cmp_{algo}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"  -> {outfile}")


# ── plotly interactive dashboard ─────────────────────────────────────────


def plotly_dashboard(df: pd.DataFrame, outdir: Path) -> None:
    """Interactive HTML dashboard with plotly."""

    # Filter out timeouts for cleaner visuals (they'll show as gaps)
    df_ok = df[~df["timed_out"]].copy()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Nodos expandidos (log)",
            "Tiempo de ejecución (log)",
            "Costo de la solución",
            "Nodos expandidos: heatmap",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "heatmap"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 1) Expanded nodes bar
    for label in df_ok["label"].unique():
        sub = df_ok[df_ok["label"] == label]
        fig.add_trace(
            go.Bar(
                name=label, x=sub["board"], y=sub["expanded_nodes"], legendgroup=label
            ),
            row=1,
            col=1,
        )

    # 2) Time bar
    for label in df_ok["label"].unique():
        sub = df_ok[df_ok["label"] == label]
        fig.add_trace(
            go.Bar(
                name=label,
                x=sub["board"],
                y=sub["time_elapsed"],
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 3) Cost bar
    df_solved = df_ok[df_ok["success"] == True]
    for label in df_solved["label"].unique():
        sub = df_solved[df_solved["label"] == label]
        fig.add_trace(
            go.Bar(
                name=label,
                x=sub["board"],
                y=sub["cost"],
                legendgroup=label,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 4) Heatmap
    pivot = df.pivot_table(
        index="label", columns="board", values="expanded_nodes", aggfunc="first"
    )
    # Replace NaN with 0 for display
    z_data = pivot.fillna(0).values.tolist()
    text_data = pivot.map(
        lambda v: f"{int(v):,}" if not pd.isna(v) else "T/O"
    ).values.tolist()

    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=list(pivot.columns),
            y=list(pivot.index),
            text=text_data,
            texttemplate="%{text}",
            colorscale="YlOrRd",
            showscale=True,
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)

    fig.update_layout(
        height=900,
        width=1400,
        title_text="Benchmark Sokoban — Comparación de algoritmos de búsqueda",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )

    outfile = outdir / "dashboard.html"
    fig.write_html(str(outfile))
    print(f"  -> {outfile}")


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar visualizaciones del benchmark de Sokoban"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "results" / "benchmark" / "benchmark.csv"),
        help="Ruta al CSV de benchmark",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "results" / "benchmark" / "plots"),
        help="Directorio de salida para los gráficos",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        print(f"Error: no se encontró el CSV en {csv_path}", file=sys.stderr)
        print("Ejecutá primero: uv run python scripts/run_batch.py", file=sys.stderr)
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Visualización Benchmark Sokoban ===")
    print(f"Input:  {csv_path}")
    print(f"Output: {outdir}")
    print()

    df = load_data(csv_path)
    print(
        f"Filas: {len(df)}  |  Tableros: {df['board'].nunique()}  |  Configs: {df['label'].nunique()}"
    )
    print()

    print("Generando gráficos matplotlib...")
    plot_expanded(df, outdir)
    plot_time(df, outdir)
    plot_cost(df, outdir)
    plot_heatmap(df, outdir)
    plot_heuristic_comparison(df, outdir)

    print("\nGenerando dashboard interactivo (plotly)...")
    plotly_dashboard(df, outdir)

    print(
        f"\n=== Listo! {len(list(outdir.iterdir()))} archivos generados en {outdir} ==="
    )


if __name__ == "__main__":
    main()
