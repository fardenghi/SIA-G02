"""
Análisis comparativo de métodos de mutación.

Compara single_gene vs uniform_multigen ejecutando el AG múltiples veces por
cada método, agrega los resultados con promedio ± desviación estándar y genera
gráficos comparativos robustos.
"""

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

MUTATION_METHODS = ["single_gene", "uniform_multigen"]

LABELS = {
    "single_gene": "Mutación Gen Simple",
    "uniform_multigen": "Mutación Multigen Uniforme",
}

MUTATION_TYPE_MAP = {
    "single_gene": MutationType.SINGLE_GENE,
    "uniform_multigen": MutationType.UNIFORM_MULTIGEN,
}

FITNESS_AXIS_PADDING_RATIO = 1.0
FITNESS_AXIS_MIN_MARGIN = 0.02
FITNESS_AXIS_MIN_RANGE = 0.12


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis comparativo de mutación: single_gene vs uniform_multigen (promedio ± std)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", required=True, help="Imagen objetivo")
    parser.add_argument(
        "--triangles", "-t", type=int, default=100, help="Triángulos por individuo"
    )
    parser.add_argument(
        "--generations", "-g", type=int, default=2000, help="Generaciones máximas"
    )
    parser.add_argument(
        "--population", "-p", type=int, default=100, help="Tamaño de población"
    )
    parser.add_argument("--runs", "-r", type=int, default=3, help="Corridas por método")
    parser.add_argument(
        "--fitness",
        "-f",
        type=str,
        default="linear",
        choices=[
            "linear",
            "rmse",
            "inverse_normalized",
            "exponential",
            "inverse_mse",
            "detail_weighted",
            "composite",
            "ssim",
            "edge_loss",
        ],
        help="Función de fitness",
    )
    parser.add_argument(
        "--max-size", type=int, default=128, help="Tamaño máximo de la imagen"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/mutation_analysis",
        help="Directorio de salida",
    )
    # Operadores fijos (modificables por CLI)
    parser.add_argument(
        "--selection",
        type=str,
        default="probabilistic_tournament",
        choices=[
            "elite",
            "tournament",
            "probabilistic_tournament",
            "roulette",
            "universal",
            "boltzmann",
            "rank",
        ],
        help="Método de selección (fijo)",
    )
    parser.add_argument(
        "--survival",
        type=str,
        default="additive",
        choices=["additive", "exclusive"],
        help="Estrategia de supervivencia (fija)",
    )
    parser.add_argument(
        "--crossover",
        type=str,
        default="uniform",
        choices=[
            "single_point",
            "two_point",
            "uniform",
            "annular",
            "spatial_zindex",
            "arithmetic",
        ],
        help="Método de cruza (fijo)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------


def run_once(
    mutation_method: str,
    target_image: Image.Image,
    evo_config: EvolutionConfig,
    selection_method: str,
    crossover_method: str,
    survival_method: str,
    fitness_method: str,
) -> dict:
    """Ejecuta el AG una vez y retorna resultados."""
    # Semilla única por proceso: evita corridas idénticas en fork de Linux
    random.seed()
    np.random.seed(int.from_bytes(os.urandom(4), "big"))

    mutation_params = MutationParams(
        mutation_type=MUTATION_TYPE_MAP[mutation_method],
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
    )
    result = engine.run()
    return {
        "best_fitness": result.best_fitness,
        "history": result.history,
        "elapsed_time": result.elapsed_time,
    }


def run_method(
    mutation_method: str,
    target_image: Image.Image,
    evo_config: EvolutionConfig,
    selection_method: str,
    crossover_method: str,
    survival_method: str,
    fitness_method: str,
    num_runs: int,
) -> dict:
    """Ejecuta el AG num_runs veces en paralelo y agrega resultados."""
    print(f"    lanzando {num_runs} corridas en paralelo...")
    with ProcessPoolExecutor(max_workers=num_runs) as executor:
        futures = {
            executor.submit(
                run_once,
                mutation_method,
                target_image,
                evo_config,
                selection_method,
                crossover_method,
                survival_method,
                fitness_method,
            ): i
            for i in range(num_runs)
        }
        run_results = []
        for future in as_completed(futures):
            r = future.result()
            run_results.append(r)
            print(
                f"    corrida {len(run_results)}/{num_runs} completada  "
                f"fitness: {r['best_fitness']:.6f}  tiempo: {r['elapsed_time']:.1f}s"
            )

    all_finals = [r["best_fitness"] for r in run_results]
    all_times = [r["elapsed_time"] for r in run_results]
    all_histories = [r["history"] for r in run_results]

    num_gens = len(all_histories[0])
    avg_history = []
    std_history = []
    for gen_idx in range(num_gens):
        vals = [h[gen_idx]["best_fitness"] for h in all_histories]
        avg_history.append(np.mean(vals))
        std_history.append(np.std(vals))

    generations = [h["generation"] for h in all_histories[0]]

    return {
        "method": mutation_method,
        "avg_fitness": float(np.mean(all_finals)),
        "std_fitness": float(np.std(all_finals)),
        "avg_time": float(np.mean(all_times)),
        "run_fitnesses": all_finals,
        "run_times": all_times,
        "avg_history": avg_history,
        "std_history": std_history,
        "generations": generations,
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------


def save_csv_results(results: list, output_dir: Path):
    """Exporta CSVs para reconstruir los gráficos sin re-ejecutar el experimento."""
    sorted_results = sorted(results, key=lambda r: r["avg_fitness"], reverse=True)

    summary_rows = []
    evolution_rows = []
    run_rows = []

    for r in sorted_results:
        method = r["method"]
        label = str(LABELS.get(method, method))

        summary_rows.append(
            {
                "method": method,
                "label": label,
                "avg_fitness": r["avg_fitness"],
                "std_fitness": r["std_fitness"],
                "avg_time": r["avg_time"],
            }
        )

        for gen, avg, std in zip(r["generations"], r["avg_history"], r["std_history"]):
            evolution_rows.append(
                {
                    "method": method,
                    "label": label,
                    "generation": gen,
                    "avg_fitness": avg,
                    "std_fitness": std,
                }
            )

        for run_idx, (fitness, elapsed_time) in enumerate(
            zip(r["run_fitnesses"], r["run_times"]), start=1
        ):
            run_rows.append(
                {
                    "method": method,
                    "label": label,
                    "run": run_idx,
                    "best_fitness": fitness,
                    "elapsed_time": elapsed_time,
                }
            )

    summary_path = output_dir / "final_fitness_summary.csv"
    evolution_path = output_dir / "evolution_fitness.csv"
    runs_path = output_dir / "run_results.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(evolution_rows).to_csv(evolution_path, index=False)
    pd.DataFrame(run_rows).to_csv(runs_path, index=False)

    print(f"  CSV resumen guardado: {summary_path}")
    print(f"  CSV evolución guardado: {evolution_path}")
    print(f"  CSV corridas guardado: {runs_path}")


def compute_fitness_axis_limits(avgs: list, stds: list) -> tuple[float, float]:
    """Calcula límites de eje con zoom moderado para barras de fitness."""
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


def plot_evolution(results: list, output_dir: Path):
    """Curvas promedio ± banda std por generación."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(10)]

    for i, r in enumerate(results):
        gens = r["generations"]
        avg = np.array(r["avg_history"])
        std = np.array(r["std_history"])
        color = colors[i % len(colors)]
        label = str(LABELS.get(r["method"], r["method"]))

        ax.plot(gens, avg, label=label, linewidth=1.8, color=color)
        ax.fill_between(gens, avg - std, avg + std, alpha=0.15, color=color)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (promedio ± std)")
    ax.set_title("Evolución del Fitness — Métodos de Mutación")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "evolution_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico evolución guardado: {path}")


def plot_final_fitness(results: list, output_dir: Path):
    """Barras verticales con error bars y zoom en escala de fitness."""
    sorted_results = sorted(results, key=lambda r: r["avg_fitness"], reverse=True)

    names = [str(LABELS.get(r["method"], r["method"])) for r in sorted_results]
    avgs = [r["avg_fitness"] for r in sorted_results]
    stds = [r["std_fitness"] for r in sorted_results]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(sorted_results))]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(names))
    bars = ax.bar(
        x_pos, avgs, yerr=stds, color=colors, capsize=5, error_kw={"linewidth": 1.5}
    )

    y_min, y_max = compute_fitness_axis_limits(avgs, stds)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=0, ha="center")

    for bar, avg, std in zip(bars, avgs, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            avg + std + 0.001,
            f"{avg:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Método de mutación")
    ax.set_ylabel("Fitness final (promedio ± std)")
    ax.set_title("Fitness Final por Método de Mutación")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "final_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico fitness final guardado: {path}")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------


def print_summary(results: list):
    col_w = 30
    print()
    print("=" * 70)
    print("RESUMEN — MÉTODOS DE MUTACIÓN")
    print("=" * 70)
    print(
        f"{'Método':<{col_w}} {'Avg Fitness':>12} {'Std Fitness':>12} {'Avg Tiempo':>12}"
    )
    print("-" * 70)

    sorted_results = sorted(results, key=lambda r: r["avg_fitness"], reverse=True)
    medals = ["1°", "2°"]
    for i, r in enumerate(sorted_results):
        medal = medals[i] if i < len(medals) else "  "
        label = LABELS.get(r["method"], r["method"])
        print(
            f"{medal} {label:<{col_w - 3}} "
            f"{r['avg_fitness']:>12.6f} "
            f"{r['std_fitness']:>12.6f} "
            f"{r['avg_time']:>11.2f}s"
        )

    print("=" * 70)
    best = sorted_results[0]
    print(
        f"Ganador: {LABELS.get(best['method'], best['method'])}  "
        f"(avg fitness {best['avg_fitness']:.6f} ± {best['std_fitness']:.6f})"
    )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)
    print(f"Imagen: {args.image}  ({target_image.size[0]}x{target_image.size[1]}px)")
    print(
        f"Generaciones: {args.generations} | Población: {args.population} | "
        f"Triángulos: {args.triangles} | Corridas: {args.runs}"
    )
    print(
        f"Fitness: {args.fitness} | Selección: {args.selection} | "
        f"Supervivencia: {args.survival} | Cruza: {args.crossover}"
    )
    print(f"Métodos a comparar: single_gene vs uniform_multigen")
    print()

    evo_config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
    )

    results = []
    total = len(MUTATION_METHODS)
    for idx, method in enumerate(MUTATION_METHODS, 1):
        label = LABELS[method]
        print(f"[{idx}/{total}] {label} ({args.runs} corridas)...")
        r = run_method(
            method,
            target_image,
            evo_config,
            args.selection,
            args.crossover,
            args.survival,
            args.fitness,
            args.runs,
        )
        results.append(r)
        print(
            f"       avg fitness: {r['avg_fitness']:.6f} ± {r['std_fitness']:.6f}  |  "
            f"avg tiempo: {r['avg_time']:.1f}s"
        )

    print()
    print("Generando salidas...")
    save_csv_results(results, output_dir)
    plot_evolution(results, output_dir)
    plot_final_fitness(results, output_dir)
    print_summary(results)
    print(f"\nTodo guardado en: {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
