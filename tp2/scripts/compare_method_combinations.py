"""
Comparativa de combinaciones de operadores del AG.

Ejecuta el AG con dos combinaciones distintas de operadores
(selección, supervivencia, cruza, mutación) sobre la misma imagen
y genera gráficos comparativos de resultados visuales, evolución
del fitness y tiempo de ejecución.

Las combinaciones se definen en compare_method_combinations_config.yaml.
"""

import argparse
import sys
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.engine import create_engine, EvolutionConfig
from src.genetic.mutation import MutationParams, MutationType
from src.rendering.canvas import Canvas, resize_image
from src.utils.export import save_result_image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparativa de combinaciones de operadores del AG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", required=True, help="Imagen objetivo")
    parser.add_argument("--triangles", "-t", type=int, default=50, help="Triángulos por individuo")
    parser.add_argument("--generations", "-g", type=int, default=200, help="Generaciones máximas")
    parser.add_argument("--population", "-p", type=int, default=50, help="Tamaño de población")
    parser.add_argument(
        "--fitness", "-f", type=str, default="linear",
        choices=["linear", "rmse", "inverse_normalized", "exponential",
                 "inverse_mse", "detail_weighted", "composite", "ssim", "edge_loss"],
        help="Método de fitness",
    )
    parser.add_argument(
        "--config", "-c", type=str,
        default=str(Path(__file__).parent / "compare_method_combinations_config.yaml"),
        help="Archivo YAML con las dos combinaciones de operadores",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="output/compare_combinations",
        help="Directorio de salida",
    )
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de la imagen")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carga de configuración
# ---------------------------------------------------------------------------

def load_combinations_config(path: str) -> tuple[dict, dict]:
    """Carga el YAML y extrae las dos combinaciones."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "combination_a" not in data or "combination_b" not in data:
        raise ValueError(
            f"El archivo '{path}' debe contener las claves 'combination_a' y 'combination_b'."
        )

    return data["combination_a"], data["combination_b"]


# ---------------------------------------------------------------------------
# Construcción del engine
# ---------------------------------------------------------------------------

def _selection_kwargs(sel: dict) -> dict:
    """Extrae los kwargs de selección relevantes según el método."""
    method = sel["method"]
    kwargs = {"selection_method": method}

    if method == "tournament":
        kwargs["tournament_size"] = sel.get("tournament_size", 3)
    elif method == "probabilistic_tournament":
        kwargs["threshold"] = sel.get("threshold", 0.75)
    elif method == "boltzmann":
        kwargs["boltzmann_t0"] = sel.get("boltzmann_t0", 100.0)
        kwargs["boltzmann_tc"] = sel.get("boltzmann_tc", 1.0)
        kwargs["boltzmann_k"] = sel.get("boltzmann_k", 0.005)

    return kwargs


def build_engine(combo: dict, target_image: Image.Image, evo_config: EvolutionConfig, fitness_method: str = "linear"):
    """Construye un GeneticEngine a partir de una combinación de operadores."""
    sel = combo.get("selection", {})
    sur = combo.get("survival", {})
    cx = combo.get("crossover", {})
    mut = combo.get("mutation", {})

    sel_kwargs = _selection_kwargs(sel)

    mutation_method = mut.get("method", "uniform_multigen")
    method_map = {
        "single_gene": MutationType.SINGLE_GENE,
        "limited_multigen": MutationType.LIMITED_MULTIGEN,
        "uniform_multigen": MutationType.UNIFORM_MULTIGEN,
        "complete": MutationType.COMPLETE,
        "error_map_guided": MutationType.ERROR_MAP_GUIDED,
    }
    mutation_type = method_map.get(mutation_method, MutationType.UNIFORM_MULTIGEN)

    mutation_params = MutationParams(
        mutation_type=mutation_type,
        probability=mut.get("probability", 0.3),
        gene_probability=mut.get("gene_probability", 0.1),
        position_delta=mut.get("position_delta", 0.1),
        color_delta=mut.get("color_delta", 30),
        alpha_delta=mut.get("alpha_delta", 0.1),
        field_probability=mut.get("field_probability", 1.0),
    )

    return create_engine(
        target_image=target_image,
        config=evo_config,
        selection_method=sel_kwargs.pop("selection_method"),
        tournament_size=sel_kwargs.get("tournament_size", 3),
        threshold=sel_kwargs.get("threshold", 0.75),
        boltzmann_t0=sel_kwargs.get("boltzmann_t0", 100.0),
        boltzmann_tc=sel_kwargs.get("boltzmann_tc", 1.0),
        boltzmann_k=sel_kwargs.get("boltzmann_k", 0.005),
        crossover_method=cx.get("method", "single_point"),
        crossover_probability=cx.get("probability", 0.8),
        mutation_params=mutation_params,
        survival_method=sur.get("method", "exclusive"),
        survival_selection_method=sur.get("selection_method", "elite"),
        offspring_ratio=sur.get("offspring_ratio", 1.0),
        fitness_method=fitness_method,
    )


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def run_combination(combo: dict, target_image: Image.Image, evo_config: EvolutionConfig, fitness_method: str = "linear") -> dict:
    """Ejecuta el AG con una combinación y retorna los resultados."""
    engine = build_engine(combo, target_image, evo_config, fitness_method)
    result = engine.run()
    return {
        "name": combo.get("name", "Sin nombre"),
        "best_fitness": result.best_fitness,
        "elapsed_time": result.elapsed_time,
        "history": result.history,
        "best_individual": result.best_individual,
        "width": engine.width,
        "height": engine.height,
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def plot_fitness_comparison(results: list, output_dir: Path):
    """Curvas de evolución del mejor fitness por generación."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        gens = [h["generation"] for h in r["history"]]
        best = [h["best_fitness"] for h in r["history"]]
        ax.plot(gens, best, label=r["name"], linewidth=1.8)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolución del Fitness — Mejor individuo por generación")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "comparison_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico fitness guardado: {path}")


def plot_image_comparison(results: list, output_dir: Path):
    """Comparación lado a lado de las imágenes resultado."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))

    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        canvas = Canvas(width=r["width"], height=r["height"])
        img = canvas.render(r["best_individual"])
        ax.imshow(np.array(img))
        ax.set_title(f"{r['name']}\nFitness: {r['best_fitness']:.6f}", fontsize=11)
        ax.axis("off")

    plt.suptitle("Comparación de Imágenes Resultado", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "comparison_images.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Comparación de imágenes guardada: {path}")


def plot_time_comparison(results: list, output_dir: Path):
    """Gráfico de barras horizontales con el tiempo de ejecución."""
    fig, ax = plt.subplots(figsize=(8, 4))

    names = [r["name"] for r in results]
    times = [r["elapsed_time"] for r in results]
    colors = plt.cm.tab10.colors[: len(results)]

    bars = ax.barh(names, times, color=colors)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Tiempo (segundos)")
    ax.set_title("Tiempo de Ejecución por Combinación")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "comparison_time.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico de tiempos guardado: {path}")


# ---------------------------------------------------------------------------
# Resumen consola
# ---------------------------------------------------------------------------

def print_summary(results: list):
    col_w = 30
    print()
    print("=" * 60)
    print("RESUMEN COMPARATIVA")
    print("=" * 60)
    print(f"{'Combinación':<{col_w}} {'Fitness final':>14} {'Tiempo (s)':>12}")
    print("-" * 60)

    sorted_results = sorted(results, key=lambda r: r["best_fitness"], reverse=True)
    for i, r in enumerate(sorted_results):
        medal = ["1°", "2°"][i] if i < 2 else "  "
        print(
            f"{medal} {r['name']:<{col_w - 3}} "
            f"{r['best_fitness']:>14.6f} "
            f"{r['elapsed_time']:>12.2f}"
        )

    print("=" * 60)
    best = sorted_results[0]
    print(f"Ganador: {best['name']}  (fitness {best['best_fitness']:.6f})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar imagen
    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)
    print(f"Imagen: {args.image}  ({target_image.size[0]}x{target_image.size[1]}px)")

    # Cargar combinaciones
    combo_a, combo_b = load_combinations_config(args.config)
    print(f"Combinación A: {combo_a.get('name', 'Sin nombre')}")
    print(f"Combinación B: {combo_b.get('name', 'Sin nombre')}")
    print(f"Fitness: {args.fitness} | Generaciones: {args.generations} | "
          f"Población: {args.population} | Triángulos: {args.triangles}")
    print()

    # EvolutionConfig compartida
    evo_config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
    )

    # Ejecutar ambas combinaciones
    results = []
    for idx, combo in enumerate([combo_a, combo_b], 1):
        name = combo.get("name", f"Combinación {idx}")
        print(f"[{idx}/2] Ejecutando '{name}'...")
        r = run_combination(combo, target_image, evo_config, args.fitness)
        results.append(r)
        print(f"       Fitness: {r['best_fitness']:.6f}  |  Tiempo: {r['elapsed_time']:.1f}s")

    # Generar salidas
    print()
    print("Generando salidas...")
    plot_fitness_comparison(results, output_dir)
    plot_image_comparison(results, output_dir)
    plot_time_comparison(results, output_dir)
    print_summary(results)
    print(f"\nTodo guardado en: {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
