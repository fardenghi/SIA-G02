"""
Análisis comparativo de estrategias de supervivencia.

Ejecuta el AG múltiples veces con cada estrategia de supervivencia (additive
y exclusive), agrega los resultados con promedio ± desviación estándar y
genera gráficos comparativos robustos.

Salidas
-------
survival_raw.csv      — una fila por (strategy, run, generation)
survival_summary.csv  — promedio ± std de fitness final por estrategia
evolution_fitness.png — curvas promedio ± banda std por generación
final_fitness.png     — barras horizontales con error bars del fitness final
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.engine import create_engine, EvolutionConfig
from src.genetic.mutation import MutationParams, MutationType
from src.rendering.canvas import resize_image


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SURVIVAL_METHODS = ["additive", "exclusive"]

LABELS = {
    "additive": "Supervivencia Aditiva",
    "exclusive": "Supervivencia Exclusiva",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis comparativo de estrategias de supervivencia (promedio ± std)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", help="Imagen objetivo")
    parser.add_argument("--triangles", "-t", type=int, default=100, help="Triángulos por individuo")
    parser.add_argument("--generations", "-g", type=int, default=2000, help="Generaciones máximas")
    parser.add_argument("--population", "-p", type=int, default=100, help="Tamaño de población")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Corridas por estrategia")
    parser.add_argument(
        "--fitness", "-f", type=str, default="linear",
        choices=["linear", "rmse", "inverse_normalized", "exponential",
                 "inverse_mse", "detail_weighted", "composite", "ssim", "edge_loss"],
        help="Función de fitness",
    )
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de la imagen")
    parser.add_argument(
        "--output", "-o", type=str, default="output/survival_analysis",
        help="Directorio de salida",
    )
    # Operadores fijos (modificables por CLI)
    parser.add_argument("--selection", type=str, default="probabilistic_tournament",
                        choices=["elite", "tournament", "probabilistic_tournament",
                                 "roulette", "universal", "boltzmann", "rank"],
                        help="Método de selección (fijo)")
    parser.add_argument("--crossover", type=str, default="uniform",
                        choices=["single_point", "two_point", "uniform", "annular",
                                 "spatial_zindex", "arithmetic"],
                        help="Método de cruza (fijo)")
    parser.add_argument("--mutation", type=str, default="uniform_multigen",
                        choices=["single_gene", "limited_multigen", "uniform_multigen",
                                 "complete", "error_map_guided"],
                        help="Método de mutación (fijo)")
    parser.add_argument(
        "--backend", type=str, default="gpu", choices=["cpu", "gpu"],
        help="Backend de renderizado",
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Saltar la ejecución del AG y generar los gráficos desde los CSV existentes "
             "en el directorio de salida (survival_raw.csv y survival_summary.csv).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def run_once(
    survival_method: str,
    target_image: Image.Image,
    evo_config: EvolutionConfig,
    selection_method: str,
    crossover_method: str,
    mutation_method: str,
    fitness_method: str,
    renderer: str = "gpu",
) -> dict:
    """Ejecuta el AG una vez y retorna resultados incluyendo la curva completa."""
    mutation_type_map = {
        "single_gene": MutationType.SINGLE_GENE,
        "limited_multigen": MutationType.LIMITED_MULTIGEN,
        "uniform_multigen": MutationType.UNIFORM_MULTIGEN,
        "complete": MutationType.COMPLETE,
        "error_map_guided": MutationType.ERROR_MAP_GUIDED,
    }
    mutation_params = MutationParams(
        mutation_type=mutation_type_map[mutation_method],
        probability=0.3,
        gene_probability=0.1,
        position_delta=0.1,
        color_delta=30,
        alpha_delta=0.1,
        field_probability=1.0,
    )
    engine = create_engine(
        target_image=target_image,
        config=evo_config,
        selection_method=selection_method,
        tournament_size=3,
        threshold=0.75,
        boltzmann_t0=100.0,
        boltzmann_tc=1.0,
        boltzmann_k=0.005,
        crossover_method=crossover_method,
        crossover_probability=0.8,
        mutation_params=mutation_params,
        survival_method=survival_method,
        survival_selection_method="elite",
        offspring_ratio=1.0,
        fitness_method=fitness_method,
        renderer=renderer,
    )
    result = engine.run()

    curve = np.array([h["best_fitness"] for h in result.history], dtype=np.float64)
    generations = np.array([h["generation"] for h in result.history], dtype=np.int64)

    return {
        "best_fitness": result.best_fitness,
        "elapsed_time": result.elapsed_time,
        "curve": curve,
        "generations": generations,
    }


def run_all(
    target_image: Image.Image,
    evo_config: EvolutionConfig,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corre todas las estrategias × corridas y devuelve:
      raw_df     — una fila por (strategy, run, generation)
      summary_df — una fila por (strategy) con promedio ± std del fitness final
    """
    raw_records: list[dict] = []
    final_records: list[dict] = []

    total = len(SURVIVAL_METHODS)
    for idx, method in enumerate(SURVIVAL_METHODS, 1):
        label = LABELS[method]
        print(f"[{idx}/{total}] {label} ({args.runs} corridas)...")
        run_finals = []
        run_times = []

        for run_idx in range(args.runs):
            print(f"    corrida {run_idx + 1}/{args.runs}...")
            r = run_once(
                method, target_image, evo_config,
                args.selection, args.crossover, args.mutation, args.fitness,
                args.backend,
            )
            run_finals.append(r["best_fitness"])
            run_times.append(r["elapsed_time"])

            # Filas del CSV raw: una por generación
            for gen, fit in zip(r["generations"], r["curve"]):
                raw_records.append({
                    "strategy": method,
                    "label": label,
                    "run": run_idx,
                    "generation": int(gen),
                    "best_fitness": float(fit),
                })

            # Fila de resumen final por corrida (útil para boxplots futuros)
            final_records.append({
                "strategy": method,
                "label": label,
                "run": run_idx,
                "best_fitness": float(r["best_fitness"]),
                "elapsed_time_s": float(r["elapsed_time"]),
            })

        avg_f = float(np.mean(run_finals))
        std_f = float(np.std(run_finals))
        print(f"       avg fitness: {avg_f:.6f} ± {std_f:.6f}  |  "
              f"avg tiempo: {np.mean(run_times):.1f}s")

    raw_df = pd.DataFrame.from_records(raw_records)
    finals_df = pd.DataFrame.from_records(final_records)

    summary_df = finals_df.groupby(["strategy", "label"], as_index=False).agg(
        runs=("best_fitness", "size"),
        avg_fitness=("best_fitness", "mean"),
        std_fitness=("best_fitness", "std"),
        avg_time_s=("elapsed_time_s", "mean"),
        std_time_s=("elapsed_time_s", "std"),
    )

    return raw_df, summary_df


# ---------------------------------------------------------------------------
# Persistencia CSV
# ---------------------------------------------------------------------------

def save_csvs(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    raw_path = output_dir / "survival_raw.csv"
    summary_path = output_dir / "survival_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"  CSV raw guardado:     {raw_path}")
    print(f"  CSV summary guardado: {summary_path}")


# ---------------------------------------------------------------------------
# Gráficos (desde DataFrames)
# ---------------------------------------------------------------------------

def plot_evolution(raw_df: pd.DataFrame, output_dir: Path) -> None:
    """Curvas promedio ± banda std por generación, leídas del CSV raw."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, method in enumerate(SURVIVAL_METHODS):
        subset = raw_df[raw_df["strategy"] == method]
        label = LABELS.get(method, method)
        color = colors[i % len(colors)]

        # Promedio y std por generación (sobre las múltiples corridas)
        grouped = subset.groupby("generation")["best_fitness"]
        avg = grouped.mean()
        std = grouped.std().fillna(0)
        gens = avg.index.to_numpy()

        ax.plot(gens, avg.values, label=label, linewidth=1.8, color=color)
        ax.fill_between(gens, avg.values - std.values, avg.values + std.values,
                        alpha=0.15, color=color)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (promedio ± std)")
    ax.set_title("Evolución del Fitness — Estrategias de Supervivencia")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "evolution_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico evolución guardado: {path}")


def plot_final_fitness(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Barras horizontales con error bars del fitness final, leídas del CSV summary."""
    sorted_df = summary_df.sort_values("avg_fitness", ascending=False)

    names = [LABELS.get(r, r) for r in sorted_df["strategy"]]
    avgs = sorted_df["avg_fitness"].to_numpy()
    stds = sorted_df["std_fitness"].fillna(0).to_numpy()
    colors = list(plt.cm.tab10.colors[: len(sorted_df)])

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, avgs, xerr=stds, color=colors, capsize=5,
                   error_kw={"linewidth": 1.5})

    for bar, avg, std in zip(bars, avgs, stds):
        ax.text(
            bar.get_width() + std + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{avg:.4f}",
            va="center",
            fontsize=10,
        )

    # Zoom automático: suprime el cero y centra en el rango relevante
    margin = max((avgs.max() - avgs.min()) * 0.5, 0.005)
    label_space = (avgs.max() - avgs.min() + 2 * margin) * 0.25  # espacio para etiquetas
    x_min = max(0.0, avgs.min() - stds.max() - margin)
    x_max = avgs.max() + stds.max() + margin + label_space
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Fitness final (promedio ± std)")
    ax.set_title("Fitness Final por Estrategia de Supervivencia")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "final_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico fitness final guardado: {path}")


# ---------------------------------------------------------------------------
# Resumen consola
# ---------------------------------------------------------------------------

def print_summary(summary_df: pd.DataFrame) -> None:
    col_w = 30
    print()
    print("=" * 70)
    print("RESUMEN — ESTRATEGIAS DE SUPERVIVENCIA")
    print("=" * 70)
    print(f"{'Estrategia':<{col_w}} {'Avg Fitness':>12} {'Std Fitness':>12} {'Avg Tiempo':>12}")
    print("-" * 70)

    sorted_df = summary_df.sort_values("avg_fitness", ascending=False)
    medals = ["1°", "2°"]
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        medal = medals[i] if i < len(medals) else "  "
        label = LABELS.get(row["strategy"], row["strategy"])
        print(
            f"{medal} {label:<{col_w - 3}} "
            f"{row['avg_fitness']:>12.6f} "
            f"{row['std_fitness']:>12.6f} "
            f"{row['avg_time_s']:>11.2f}s"
        )

    print("=" * 70)
    best = sorted_df.iloc[0]
    best_label = LABELS.get(best["strategy"], best["strategy"])
    print(f"Ganador: {best_label}  "
          f"(avg fitness {best['avg_fitness']:.6f} ± {best['std_fitness']:.6f})")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Modo: regenerar gráficos desde CSVs existentes ---
    if args.from_csv:
        raw_path = output_dir / "survival_raw.csv"
        summary_path = output_dir / "survival_summary.csv"
        if not raw_path.exists() or not summary_path.exists():
            print(f"ERROR: No se encontraron los CSV en '{output_dir}'.")
            print(f"  Esperados: {raw_path.name}, {summary_path.name}")
            return 1
        print(f"Leyendo datos desde CSV en: {output_dir}/")
        raw_df = pd.read_csv(raw_path)
        summary_df = pd.read_csv(summary_path)
        print("Generando gráficos...")
        plot_evolution(raw_df, output_dir)
        plot_final_fitness(summary_df, output_dir)
        print_summary(summary_df)
        print(f"\nGráficos guardados en: {output_dir}/")
        return 0

    # --- Modo normal: correr el AG ---
    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)
    print(f"Imagen: {args.image}  ({target_image.size[0]}x{target_image.size[1]}px)")
    print(f"Generaciones: {args.generations} | Población: {args.population} | "
          f"Triángulos: {args.triangles} | Corridas: {args.runs}")
    print(f"Fitness: {args.fitness} | Selección: {args.selection} | "
          f"Cruza: {args.crossover} | Mutación: {args.mutation} | Backend: {args.backend}")
    print()

    evo_config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
    )

    raw_df, summary_df = run_all(target_image, evo_config, args)

    print()
    print("Guardando datos y generando gráficos...")
    save_csvs(raw_df, summary_df, output_dir)
    plot_evolution(raw_df, output_dir)
    plot_final_fitness(summary_df, output_dir)
    print_summary(summary_df)
    print(f"\nTodo guardado en: {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
