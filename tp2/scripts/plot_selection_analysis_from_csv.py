"""
Regenera los gráficos de compare_selection_analysis a partir de CSVs.

Lee:
- final_fitness_summary.csv
- evolution_fitness.csv

Genera:
- final_fitness_from_csv.png
- evolution_fitness_from_csv.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


FITNESS_AXIS_PADDING_RATIO = 0.26
FITNESS_AXIS_MIN_MARGIN = 0.002
FITNESS_AXIS_MIN_RANGE = 0.025

EVOLUTION_ZOOM_MIN_GEN = 20
EVOLUTION_ZOOM_MAX_GEN = 140
EVOLUTION_ZOOM_PADDING_RATIO = 0.008
EVOLUTION_ZOOM_MIN_MARGIN = 0.0002
EVOLUTION_ZOOM_MIN_RANGE = 0.0015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenera gráficos de selección desde CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="output/selection_analysis",
        help="Directorio con los CSV de entrada",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output/selection_analysis",
        help="Directorio de salida para los gráficos",
    )
    return parser.parse_args()


def compute_fitness_axis_limits(
    avgs: list[float], stds: list[float]
) -> tuple[float, float]:
    min_with_error = min(avg - std for avg, std in zip(avgs, stds))
    max_with_error = max(avg + std for avg, std in zip(avgs, stds))
    span = max_with_error - min_with_error
    margin = max(FITNESS_AXIS_MIN_MARGIN, span * FITNESS_AXIS_PADDING_RATIO)

    y_min = max(0.0, min_with_error - margin)
    y_max = min(1.0, max_with_error + margin)

    if y_max - y_min < FITNESS_AXIS_MIN_RANGE:
        center = (y_min + y_max) / 2
        half_range = FITNESS_AXIS_MIN_RANGE / 2
        y_min = max(0.0, center - half_range)
        y_max = min(1.0, center + half_range)

    return y_min, y_max


def plot_evolution(df_evolution: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    methods = df_evolution[["method", "label"]].drop_duplicates().reset_index(drop=True)
    method_rows = sorted(
        methods.itertuples(index=False, name=None),
        key=lambda row: str(row[0]),
    )

    series_data = []

    for idx, (method, label) in enumerate(method_rows):
        color = cmap(int(idx) % 10)
        label_str = str(label)

        sub = df_evolution[df_evolution["method"] == method]
        gens = np.asarray(sub["generation"], dtype=np.int64)
        avg = np.asarray(sub["avg_fitness"], dtype=np.float64)
        std = np.asarray(sub["std_fitness"], dtype=np.float64)
        order = np.argsort(gens)
        gens = gens[order]
        avg = avg[order]
        std = std[order]

        series_data.append((color, gens, avg, std))

        ax.plot(gens, avg, label=label_str, linewidth=1.8, color=color)
        ax.fill_between(gens, avg - std, avg + std, alpha=0.15, color=color)

    ax_zoom = inset_axes(
        ax,
        width="44%",
        height="52%",
        loc="center",
        borderpad=1.0,
    )
    zoom_min = np.inf
    zoom_max = -np.inf
    has_zoom_data = False

    for color, gens, avg, std in series_data:
        mask = (gens >= EVOLUTION_ZOOM_MIN_GEN) & (gens <= EVOLUTION_ZOOM_MAX_GEN)
        if not np.any(mask):
            continue

        has_zoom_data = True
        z_gens = gens[mask]
        z_avg = avg[mask]
        z_std = std[mask]
        zoom_min = min(zoom_min, float(np.min(z_avg - z_std)))
        zoom_max = max(zoom_max, float(np.max(z_avg + z_std)))

        ax_zoom.plot(z_gens, z_avg, linewidth=1.5, color=color)
        ax_zoom.fill_between(
            z_gens, z_avg - z_std, z_avg + z_std, alpha=0.15, color=color
        )

    if has_zoom_data:
        span = zoom_max - zoom_min
        margin = max(EVOLUTION_ZOOM_MIN_MARGIN, span * EVOLUTION_ZOOM_PADDING_RATIO)
        y_min = max(0.0, zoom_min - margin)
        y_max = min(1.0, zoom_max + margin)

        if y_max - y_min < EVOLUTION_ZOOM_MIN_RANGE:
            center = (y_min + y_max) / 2
            half_range = EVOLUTION_ZOOM_MIN_RANGE / 2
            y_min = max(0.0, center - half_range)
            y_max = min(1.0, center + half_range)

        ax_zoom.set_xlim(EVOLUTION_ZOOM_MIN_GEN, EVOLUTION_ZOOM_MAX_GEN)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_title(
            f"Zoom G{EVOLUTION_ZOOM_MIN_GEN}-G{EVOLUTION_ZOOM_MAX_GEN}", fontsize=9
        )
        ax_zoom.grid(True, alpha=0.25)
        ax_zoom.tick_params(labelsize=8)
    else:
        ax_zoom.set_visible(False)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (promedio ± std)")
    ax.set_title("Evolución del Fitness — Métodos de Selección")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "evolution_fitness_from_csv.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico evolución guardado: {path}")


def plot_final_fitness(df_summary: pd.DataFrame, output_dir: Path):
    sorted_df = df_summary.sort_values(by="avg_fitness", ascending=False).reset_index(
        drop=True
    )

    names = sorted_df["label"].astype(str).tolist()
    avgs = sorted_df["avg_fitness"].astype(float).tolist()
    stds = sorted_df["std_fitness"].astype(float).tolist()

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x_pos = np.arange(len(names))
    bars = ax.bar(
        x_pos, avgs, yerr=stds, color=colors, capsize=5, error_kw={"linewidth": 1.5}
    )

    y_min, y_max = compute_fitness_axis_limits(avgs, stds)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, ha="right")

    text_offset = max(0.0004, (y_max - y_min) * 0.006)
    top_margin = max(0.0008, (y_max - y_min) * 0.01)
    for bar, avg, std in zip(bars, avgs, stds):
        y_target = avg + std + text_offset
        y_text = min(y_target, y_max - top_margin)
        va = "bottom" if y_text >= y_target else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{avg:.4f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.set_xlabel("Método de selección")
    ax.set_ylabel("Fitness final (promedio ± std)")
    ax.set_title("Fitness Final por Método de Selección")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "final_fitness_from_csv.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico fitness final guardado: {path}")


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = input_dir / "final_fitness_summary.csv"
    evolution_path = input_dir / "evolution_fitness.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"No existe: {summary_path}")
    if not evolution_path.exists():
        raise FileNotFoundError(f"No existe: {evolution_path}")

    df_summary = pd.read_csv(summary_path)
    df_evolution = pd.read_csv(evolution_path)

    required_summary = {"method", "label", "avg_fitness", "std_fitness"}
    required_evolution = {"method", "label", "generation", "avg_fitness", "std_fitness"}
    if not required_summary.issubset(set(df_summary.columns)):
        missing = required_summary.difference(set(df_summary.columns))
        raise ValueError(
            f"final_fitness_summary.csv sin columnas requeridas: {sorted(missing)}"
        )
    if not required_evolution.issubset(set(df_evolution.columns)):
        missing = required_evolution.difference(set(df_evolution.columns))
        raise ValueError(
            f"evolution_fitness.csv sin columnas requeridas: {sorted(missing)}"
        )

    plot_evolution(df_evolution, output_dir)
    plot_final_fitness(df_summary, output_dir)
    print(f"Todo guardado en: {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
