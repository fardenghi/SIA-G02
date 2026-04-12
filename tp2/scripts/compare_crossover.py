"""
Comparativa de métodos de cruza (crossover).

Corre todos los métodos de cruza disponibles sobre la misma imagen
y genera un gráfico comparativo de la evolución del fitness.
"""

import argparse
import sys
import time
from pathlib import Path

# Agregar el directorio raíz al path para poder importar src
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from PIL import Image

from src.genetic.engine import create_engine, EvolutionConfig
from src.genetic.mutation import MutationParams
from src.rendering.canvas import Canvas, resize_image
from src.utils.export import save_result_image

CROSSOVER_METHODS = [
    "single_point",
    "two_point",
    "uniform",
    "annular",
]

LABELS = {
    "single_point": "Un Punto",
    "two_point": "Dos Puntos",
    "uniform": "Uniforme",
    "annular": "Anular (Circular)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparativa de métodos de cruza (crossover)",
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
        "--triangles", "-t", type=int, default=50, help="Genes por individuo"
    )
    parser.add_argument(
        "--shape",
        choices=["triangle", "ellipse"],
        default="triangle",
        help="Familia de formas por corrida",
    )
    parser.add_argument(
        "--max-size", type=int, default=128, help="Tamaño máximo de la imagen"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/comparativa_crossover",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=CROSSOVER_METHODS,
        default=CROSSOVER_METHODS,
        help="Métodos a comparar (por defecto todos)",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="tournament",
        help="Método de selección a usar (fijo para todos)",
    )
    return parser.parse_args()


def run_method(
    method: str, target_image: Image.Image, config: EvolutionConfig, selection: str
) -> dict:
    """Corre un método de cruza y devuelve los resultados."""
    engine = create_engine(
        target_image=target_image,
        config=config,
        selection_method=selection,
        tournament_size=3,
        crossover_method=method,
        crossover_probability=0.8,
        mutation_params=MutationParams(probability=0.3, gene_probability=0.1),
        threshold=0.75,
        boltzmann_t0=100.0,
        boltzmann_tc=1.0,
        boltzmann_k=0.005,
    )

    result = engine.run()

    return {
        "method": method,
        "best_fitness": result.best_fitness,
        "generations": result.generations,
        "elapsed_time": result.elapsed_time,
        "history": result.history,
        "best_individual": result.best_individual,
        "width": engine.width,
        "height": engine.height,
    }


def plot_comparison(results: list, output_dir: Path):
    """Genera gráfico comparativo de evolución del fitness."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Gráfico 1: evolución del fitness (mejor por generación) ---
    ax = axes[0]
    for r in results:
        gens = [h["generation"] for h in r["history"]]
        best = [h["best_fitness"] for h in r["history"]]
        ax.plot(gens, best, label=LABELS[r["method"]], linewidth=1.8)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (1 / (1 + MSE))")
    ax.set_title("Evolución del Fitness — Mejor individuo")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Gráfico 2: barras de fitness final ---
    ax2 = axes[1]
    names = [LABELS[r["method"]] for r in results]
    fitnesses = [r["best_fitness"] for r in results]
    times = [r["elapsed_time"] for r in results]

    colors = plt.cm.tab10.colors[: len(results)]
    bars = ax2.barh(names, fitnesses, color=colors)
    ax2.set_xlabel("Fitness final")
    ax2.set_title("Fitness final por método")
    ax2.grid(True, axis="x", alpha=0.3)

    # Anotar tiempo en cada barra
    for bar, t in zip(bars, times):
        ax2.text(
            bar.get_width() * 1.002,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s",
            va="center",
            fontsize=8,
            color="gray",
        )

    plt.suptitle(
        "Comparativa de Métodos de Cruza (Crossover)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    path = output_dir / "comparativa_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico guardado: {path}")


def save_result_images(results: list, output_dir: Path):
    """Guarda la imagen final de cada método."""
    images_dir = output_dir / "imagenes"
    images_dir.mkdir(exist_ok=True)

    for r in results:
        path = images_dir / f"{r['method']}.png"
        save_result_image(r["best_individual"], r["width"], r["height"], path)


def print_summary_table(results: list):
    """Imprime tabla resumen en consola."""
    col_w = 30
    print()
    print("=" * 65)
    print("RESUMEN COMPARATIVA - MÉTODOS DE CRUZA")
    print("=" * 65)
    print(f"{'Método':<{col_w}} {'Fitness final':>14} {'Tiempo (s)':>12}")
    print("-" * 65)

    sorted_results = sorted(results, key=lambda r: r["best_fitness"], reverse=True)
    for i, r in enumerate(sorted_results):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(
            f"{medal} {LABELS[r['method']]:<{col_w - 3}} "
            f"{r['best_fitness']:>14.6f} "
            f"{r['elapsed_time']:>12.2f}"
        )

    print("=" * 65)
    best = sorted_results[0]
    print(f"Ganador: {LABELS[best['method']]}  (fitness {best['best_fitness']:.6f})")
    print("=" * 65)


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar imagen
    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)
    print(f"Imagen: {args.image}  ({target_image.size[0]}x{target_image.size[1]}px)")
    print(
        f"Generaciones: {args.generations} | Población: {args.population} | Genes: {args.triangles} | Forma: {args.shape}"
    )
    print(f"Selección: {args.selection} (fijo para todas las comparaciones)")
    print(f"Métodos de cruza a comparar: {', '.join(args.methods)}")
    print()

    config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        shape_type=args.shape,
        max_generations=args.generations,
    )

    results = []
    total = len(args.methods)
    for idx, method in enumerate(args.methods, 1):
        print(f"[{idx}/{total}] Crossover: {LABELS[method]}...")
        r = run_method(method, target_image, config, args.selection)
        results.append(r)
        print(
            f"       fitness: {r['best_fitness']:.6f}  |  tiempo: {r['elapsed_time']:.1f}s"
        )

    print()
    print("Generando salidas...")
    plot_comparison(results, output_dir)
    save_result_images(results, output_dir)
    print_summary_table(results)
    print(f"\nTodo guardado en: {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
