"""
Comparativa de métodos de mutación.

Corre todos los métodos de mutación disponibles sobre la misma imagen
con diferentes probabilidades de mutación (Pm), y genera gráficos
comparativos de la evolución del fitness.

Métodos de mutación según la teoría:
- single_gene: Muta exactamente 1 gen (triángulo)
- limited_multigen: Muta entre [1, M] genes
- uniform_multigen: Cada gen tiene probabilidad independiente de mutar
- complete: Muta todos los genes del individuo
"""

import argparse
import sys
from pathlib import Path
from itertools import product

# Agregar el directorio raíz al path para poder importar src
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from PIL import Image

from src.genetic.engine import create_engine, EvolutionConfig
from src.genetic.mutation import create_mutation_params
from src.rendering.canvas import resize_image
from src.utils.export import save_result_image

MUTATION_METHODS = [
    "single_gene",
    "limited_multigen",
    "uniform_multigen",
    "complete",
]

LABELS = {
    "single_gene": "Gen Único",
    "limited_multigen": "Multigen Limitada",
    "uniform_multigen": "Multigen Uniforme",
    "complete": "Completa",
}

DEFAULT_PROBABILITIES = [0.1, 0.3, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparativa de métodos de mutación",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", required=True, help="Imagen objetivo")
    parser.add_argument(
        "--generations", "-g", type=int, default=200, help="Generaciones"
    )
    parser.add_argument(
        "--population", "-p", type=int, default=50, help="Tamaño de población"
    )
    parser.add_argument(
        "--triangles", "-t", type=int, default=50, help="Triángulos por individuo"
    )
    parser.add_argument(
        "--max-size", type=int, default=128, help="Tamaño máximo de la imagen"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/comparativa_mutation",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=MUTATION_METHODS,
        default=MUTATION_METHODS,
        help="Métodos a comparar (por defecto todos)",
    )
    parser.add_argument(
        "--probabilities",
        nargs="+",
        type=float,
        default=DEFAULT_PROBABILITIES,
        help="Probabilidades de mutación (Pm) a probar",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="tournament",
        help="Método de selección a usar (fijo para todos)",
    )
    parser.add_argument(
        "--crossover",
        type=str,
        default="single_point",
        help="Método de cruza a usar (fijo para todos)",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=3,
        help="Máximo de genes a mutar (M) para limited_multigen",
    )
    parser.add_argument(
        "--gene-probability",
        type=float,
        default=0.1,
        help="Probabilidad por gen para uniform_multigen",
    )
    return parser.parse_args()


def run_experiment(
    method: str,
    probability: float,
    target_image: Image.Image,
    config: EvolutionConfig,
    selection: str,
    crossover: str,
    max_genes: int,
    gene_probability: float,
) -> dict:
    """Corre un experimento de mutación y devuelve los resultados."""
    mutation_params = create_mutation_params(
        mutation_method=method,
        probability=probability,
        gene_probability=gene_probability,
        max_genes=max_genes,
    )

    engine = create_engine(
        target_image=target_image,
        config=config,
        selection_method=selection,
        tournament_size=3,
        crossover_method=crossover,
        crossover_probability=0.8,
        mutation_params=mutation_params,
        threshold=0.75,
        boltzmann_t0=100.0,
        boltzmann_tc=1.0,
        boltzmann_k=0.005,
    )

    result = engine.run()

    return {
        "method": method,
        "probability": probability,
        "best_fitness": result.best_fitness,
        "generations": result.generations,
        "elapsed_time": result.elapsed_time,
        "history": result.history,
        "best_individual": result.best_individual,
        "width": engine.width,
        "height": engine.height,
    }


def plot_comparison_by_method(results: list, output_dir: Path):
    """Genera gráfico comparativo agrupado por método."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    methods = list(set(r["method"] for r in results))
    probabilities = sorted(set(r["probability"] for r in results))

    # Colores para cada probabilidad
    colors = plt.cm.viridis(
        [
            i / (len(probabilities) - 1) if len(probabilities) > 1 else 0.5
            for i in range(len(probabilities))
        ]
    )

    # --- Gráfico 1: Evolución del fitness por método ---
    ax = axes[0, 0]
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        for r, color in zip(method_results, colors):
            gens = [h["generation"] for h in r["history"]]
            best = [h["best_fitness"] for h in r["history"]]
            label = f"{LABELS[method]} (Pm={r['probability']})"
            ax.plot(gens, best, label=label, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (1 / (1 + MSE))")
    ax.set_title("Evolución del Fitness — Todos los experimentos")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Gráfico 2: Fitness final por método (barras agrupadas) ---
    ax2 = axes[0, 1]

    x = range(len(methods))
    width = 0.8 / len(probabilities)

    for i, prob in enumerate(probabilities):
        fitnesses = []
        for method in methods:
            r = next(
                (
                    r
                    for r in results
                    if r["method"] == method and r["probability"] == prob
                ),
                None,
            )
            fitnesses.append(r["best_fitness"] if r else 0)

        offset = (i - len(probabilities) / 2 + 0.5) * width
        bars = ax2.bar(
            [xi + offset for xi in x],
            fitnesses,
            width,
            label=f"Pm={prob}",
            color=colors[i],
            alpha=0.85,
        )

    ax2.set_xlabel("Método de Mutación")
    ax2.set_ylabel("Fitness final")
    ax2.set_title("Fitness final por método y probabilidad")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    # --- Gráfico 3: Heatmap de fitness ---
    ax3 = axes[1, 0]

    fitness_matrix = []
    for method in methods:
        row = []
        for prob in probabilities:
            r = next(
                (
                    r
                    for r in results
                    if r["method"] == method and r["probability"] == prob
                ),
                None,
            )
            row.append(r["best_fitness"] if r else 0)
        fitness_matrix.append(row)

    im = ax3.imshow(fitness_matrix, cmap="YlGn", aspect="auto")
    ax3.set_xticks(range(len(probabilities)))
    ax3.set_xticklabels([f"Pm={p}" for p in probabilities])
    ax3.set_yticks(range(len(methods)))
    ax3.set_yticklabels([LABELS[m] for m in methods])
    ax3.set_title("Heatmap de Fitness Final")

    # Anotar valores en el heatmap
    for i in range(len(methods)):
        for j in range(len(probabilities)):
            text = ax3.text(
                j,
                i,
                f"{fitness_matrix[i][j]:.4f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.colorbar(im, ax=ax3, label="Fitness")

    # --- Gráfico 4: Tiempo de ejecución ---
    ax4 = axes[1, 1]

    for i, prob in enumerate(probabilities):
        times = []
        for method in methods:
            r = next(
                (
                    r
                    for r in results
                    if r["method"] == method and r["probability"] == prob
                ),
                None,
            )
            times.append(r["elapsed_time"] if r else 0)

        offset = (i - len(probabilities) / 2 + 0.5) * width
        ax4.bar(
            [xi + offset for xi in x],
            times,
            width,
            label=f"Pm={prob}",
            color=colors[i],
            alpha=0.85,
        )

    ax4.set_xlabel("Método de Mutación")
    ax4.set_ylabel("Tiempo (s)")
    ax4.set_title("Tiempo de ejecución por método y probabilidad")
    ax4.set_xticks(x)
    ax4.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right")
    ax4.legend(fontsize=9)
    ax4.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Comparativa de Métodos de Mutación", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "comparativa_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico guardado: {path}")


def plot_by_probability(results: list, output_dir: Path):
    """Genera un gráfico por cada probabilidad para comparación más clara."""
    probabilities = sorted(set(r["probability"] for r in results))
    methods = list(set(r["method"] for r in results))

    for prob in probabilities:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        prob_results = [r for r in results if r["probability"] == prob]

        # Evolución del fitness
        ax = axes[0]
        for r in prob_results:
            gens = [h["generation"] for h in r["history"]]
            best = [h["best_fitness"] for h in r["history"]]
            ax.plot(gens, best, label=LABELS[r["method"]], linewidth=1.8)

        ax.set_xlabel("Generación")
        ax.set_ylabel("Fitness")
        ax.set_title(f"Evolución del Fitness — Pm={prob}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Barras de fitness final
        ax2 = axes[1]
        names = [LABELS[r["method"]] for r in prob_results]
        fitnesses = [r["best_fitness"] for r in prob_results]
        times = [r["elapsed_time"] for r in prob_results]

        colors = plt.cm.tab10.colors[: len(prob_results)]
        bars = ax2.barh(names, fitnesses, color=colors)
        ax2.set_xlabel("Fitness final")
        ax2.set_title(f"Fitness final — Pm={prob}")
        ax2.grid(True, axis="x", alpha=0.3)

        for bar, t in zip(bars, times):
            ax2.text(
                bar.get_width() * 1.002,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s",
                va="center",
                fontsize=8,
                color="gray",
            )

        plt.tight_layout()
        path = output_dir / f"comparativa_pm_{prob}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Gráfico guardado: {path}")


def save_result_images(results: list, output_dir: Path):
    """Guarda la imagen final de cada experimento."""
    images_dir = output_dir / "imagenes"
    images_dir.mkdir(exist_ok=True)

    for r in results:
        filename = f"{r['method']}_pm{r['probability']}.png"
        path = images_dir / filename
        save_result_image(r["best_individual"], r["width"], r["height"], path)


def print_summary_table(results: list):
    """Imprime tabla resumen en consola."""
    col_w = 25
    print()
    print("=" * 80)
    print("RESUMEN COMPARATIVA - MÉTODOS DE MUTACIÓN")
    print("=" * 80)
    print(f"{'Método':<{col_w}} {'Pm':>6} {'Fitness final':>14} {'Tiempo (s)':>12}")
    print("-" * 80)

    sorted_results = sorted(results, key=lambda r: r["best_fitness"], reverse=True)
    for i, r in enumerate(sorted_results):
        medal = ["1.", "2.", "3."][i] if i < 3 else "  "
        print(
            f"{medal} {LABELS[r['method']]:<{col_w - 3}} "
            f"{r['probability']:>6.2f} "
            f"{r['best_fitness']:>14.6f} "
            f"{r['elapsed_time']:>12.2f}"
        )

    print("=" * 80)
    best = sorted_results[0]
    print(
        f"Ganador: {LABELS[best['method']]} con Pm={best['probability']}  "
        f"(fitness {best['best_fitness']:.6f})"
    )
    print("=" * 80)

    # Tabla resumen por método (promedio)
    print()
    print("RESUMEN POR MÉTODO (promedio de probabilidades)")
    print("-" * 60)
    methods = list(set(r["method"] for r in results))
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        avg_fitness = sum(r["best_fitness"] for r in method_results) / len(
            method_results
        )
        avg_time = sum(r["elapsed_time"] for r in method_results) / len(method_results)
        print(
            f"{LABELS[method]:<{col_w}} fitness={avg_fitness:.6f}  tiempo={avg_time:.1f}s"
        )
    print("-" * 60)


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar imagen
    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)

    print(f"Imagen: {args.image}  ({target_image.size[0]}x{target_image.size[1]}px)")
    print(
        f"Generaciones: {args.generations} | Población: {args.population} | Triángulos: {args.triangles}"
    )
    print(f"Selección: {args.selection} | Cruza: {args.crossover}")
    print(f"Métodos de mutación: {', '.join(args.methods)}")
    print(f"Probabilidades (Pm): {', '.join(str(p) for p in args.probabilities)}")
    print(
        f"Max genes (M): {args.max_genes} | Gene probability: {args.gene_probability}"
    )
    print()

    config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
    )

    # Generar todas las combinaciones método x probabilidad
    experiments = list(product(args.methods, args.probabilities))
    total = len(experiments)

    results = []
    for idx, (method, prob) in enumerate(experiments, 1):
        print(f"[{idx}/{total}] {LABELS[method]} con Pm={prob}...")
        r = run_experiment(
            method=method,
            probability=prob,
            target_image=target_image,
            config=config,
            selection=args.selection,
            crossover=args.crossover,
            max_genes=args.max_genes,
            gene_probability=args.gene_probability,
        )
        results.append(r)
        print(
            f"       fitness: {r['best_fitness']:.6f}  |  tiempo: {r['elapsed_time']:.1f}s"
        )

    print()
    print("Generando salidas...")
    plot_comparison_by_method(results, output_dir)
    plot_by_probability(results, output_dir)
    save_result_images(results, output_dir)
    print_summary_table(results)
    print(f"\nTodo guardado en: {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
