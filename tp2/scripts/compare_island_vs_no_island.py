"""
Compara una corrida con islas contra una sin islas usando metrics.csv ya generados.

Genera dos figuras:
- Una con la evolucion completa del fitness.
- Una con tres zooms para las primeras 250, 500 y 1000 generaciones.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ISLAND_PATH = "output/island_paper"
DEFAULT_NO_ISLAND_PATH = "output/mona_no_island"
DEFAULT_OUTPUT_DIR = "output/island_vs_no_island"
ZOOM_LIMITS = (250, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara fitness de una corrida con islas vs una sin islas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--island",
        type=str,
        default=DEFAULT_ISLAND_PATH,
        help="Directorio de salida o archivo metrics.csv de la corrida con islas",
    )
    parser.add_argument(
        "--no-island",
        type=str,
        default=DEFAULT_NO_ISLAND_PATH,
        help="Directorio de salida o archivo metrics.csv de la corrida sin islas",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde guardar los graficos",
    )
    parser.add_argument(
        "--fitness-column",
        type=str,
        default="best_fitness",
        help="Columna de fitness a graficar dentro del metrics.csv",
    )
    parser.add_argument(
        "--label-island",
        type=str,
        default="Con islas",
        help="Etiqueta para la curva con islas",
    )
    parser.add_argument(
        "--label-no-island",
        type=str,
        default="Sin islas",
        help="Etiqueta para la curva sin islas",
    )
    return parser.parse_args()


def resolve_metrics_path(raw_path: str) -> Path:
    path = Path(raw_path)
    metrics_path = path / "metrics.csv" if path.is_dir() else path

    if not metrics_path.exists():
        raise FileNotFoundError(f"No existe el archivo de metricas: {metrics_path}")

    if metrics_path.name != "metrics.csv":
        raise ValueError(
            f"Se esperaba un archivo metrics.csv o un directorio que lo contenga: {metrics_path}"
        )

    return metrics_path


def load_metrics(metrics_path: Path, fitness_column: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)

    required_columns = {"generation", fitness_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas en {metrics_path}: {sorted(missing_columns)}"
        )

    if "run_id" in df.columns and df["run_id"].nunique() > 1:
        raise ValueError(
            f"{metrics_path} contiene multiples run_id. Usa un CSV con una sola corrida."
        )

    return df.sort_values("generation").reset_index(drop=True)


def plot_full_comparison(
    island_df: pd.DataFrame,
    no_island_df: pd.DataFrame,
    fitness_column: str,
    label_island: str,
    label_no_island: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fitness_evolution_full.png"

    plt.figure(figsize=(11, 6))
    plt.plot(
        island_df["generation"],
        island_df[fitness_column],
        label=label_island,
        linewidth=2,
        color="tab:blue",
    )
    plt.plot(
        no_island_df["generation"],
        no_island_df[fitness_column],
        label=label_no_island,
        linewidth=2,
        color="tab:orange",
    )
    plt.xlabel("Generacion")
    plt.ylabel("Fitness")
    plt.title("Evolucion completa del fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_first_n_generations(
    island_df: pd.DataFrame,
    no_island_df: pd.DataFrame,
    fitness_column: str,
    label_island: str,
    label_no_island: str,
    output_dir: Path,
    limit: int,
) -> Path:
    output_path = output_dir / f"fitness_evolution_first_{limit}.png"

    island_slice = island_df[island_df["generation"] <= limit]
    no_island_slice = no_island_df[no_island_df["generation"] <= limit]

    plt.figure(figsize=(11, 6))
    plt.plot(
        island_slice["generation"],
        island_slice[fitness_column],
        label=label_island,
        linewidth=2,
        color="tab:blue",
    )
    plt.plot(
        no_island_slice["generation"],
        no_island_slice[fitness_column],
        label=label_no_island,
        linewidth=2,
        color="tab:orange",
    )
    plt.xlim(0, limit)
    plt.xlabel("Generacion")
    plt.ylabel("Fitness")
    plt.title(f"Evolucion del fitness en las primeras {limit} generaciones")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_zoom_comparison(
    island_df: pd.DataFrame,
    no_island_df: pd.DataFrame,
    fitness_column: str,
    label_island: str,
    label_no_island: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fitness_evolution_first_250_500_1000.png"

    fig, axes = plt.subplots(1, len(ZOOM_LIMITS), figsize=(18, 5), sharey=True)

    for ax, limit in zip(axes, ZOOM_LIMITS):
        island_slice = island_df[island_df["generation"] <= limit]
        no_island_slice = no_island_df[no_island_df["generation"] <= limit]

        ax.plot(
            island_slice["generation"],
            island_slice[fitness_column],
            label=label_island,
            linewidth=2,
            color="tab:blue",
        )
        ax.plot(
            no_island_slice["generation"],
            no_island_slice[fitness_column],
            label=label_no_island,
            linewidth=2,
            color="tab:orange",
        )
        ax.set_xlim(0, limit)
        ax.set_title(f"Primeras {limit} generaciones")
        ax.set_xlabel("Generacion")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Fitness")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Comparacion temprana del fitness")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()

    island_metrics_path = resolve_metrics_path(args.island)
    no_island_metrics_path = resolve_metrics_path(args.no_island)

    island_df = load_metrics(island_metrics_path, args.fitness_column)
    no_island_df = load_metrics(no_island_metrics_path, args.fitness_column)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_plot_path = plot_full_comparison(
        island_df,
        no_island_df,
        args.fitness_column,
        args.label_island,
        args.label_no_island,
        output_dir,
    )
    first_100_plot_path = plot_first_n_generations(
        island_df,
        no_island_df,
        args.fitness_column,
        args.label_island,
        args.label_no_island,
        output_dir,
        limit=100,
    )
    zoom_plot_path = plot_zoom_comparison(
        island_df,
        no_island_df,
        args.fitness_column,
        args.label_island,
        args.label_no_island,
        output_dir,
    )

    island_final = island_df.iloc[-1][args.fitness_column]
    no_island_final = no_island_df.iloc[-1][args.fitness_column]

    print(f"CSV con islas: {island_metrics_path}")
    print(f"CSV sin islas: {no_island_metrics_path}")
    print(f"Grafico completo: {full_plot_path}")
    print(f"Grafico primeras 100 generaciones: {first_100_plot_path}")
    print(f"Grafico zoom 250/500/1000: {zoom_plot_path}")
    print(f"Fitness final {args.label_island}: {island_final:.6f}")
    print(f"Fitness final {args.label_no_island}: {no_island_final:.6f}")


if __name__ == "__main__":
    main()
