"""
Comparativa de estrategias de supervivencia.

Corre las distintas estrategias de supervivencia sobre la misma imagen
y genera un gráfico comparativo de la evolución del fitness.

Estrategias comparadas:
- Exclusiva (K=N): Todos los hijos reemplazan a los padres
- Exclusiva (K>N): Genera más hijos y selecciona los mejores
- Aditiva (K=N): Padres e hijos compiten juntos
- Aditiva (K>N): Pool más grande para selección
"""

import argparse
import sys
from pathlib import Path

# Agregar el directorio raíz al path para poder importar src
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from PIL import Image

from src.genetic.engine import create_engine, EvolutionConfig
from src.genetic.mutation import MutationParams
from src.rendering.canvas import resize_image
from src.utils.export import save_result_image


# Configuraciones de supervivencia a comparar
SURVIVAL_CONFIGS = [
    {
        "id": "exclusive_1.0",
        "method": "exclusive",
        "offspring_ratio": 1.0,
        "label": "Exclusiva (K=N)",
    },
    {
        "id": "exclusive_1.5",
        "method": "exclusive",
        "offspring_ratio": 1.5,
        "label": "Exclusiva (K=1.5N)",
    },
    {
        "id": "exclusive_2.0",
        "method": "exclusive",
        "offspring_ratio": 2.0,
        "label": "Exclusiva (K=2N)",
    },
    {
        "id": "additive_1.0",
        "method": "additive",
        "offspring_ratio": 1.0,
        "label": "Aditiva (K=N)",
    },
    {
        "id": "additive_1.5",
        "method": "additive",
        "offspring_ratio": 1.5,
        "label": "Aditiva (K=1.5N)",
    },
    {
        "id": "additive_2.0",
        "method": "additive",
        "offspring_ratio": 2.0,
        "label": "Aditiva (K=2N)",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparativa de estrategias de supervivencia",
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
        default="output/comparativa_survival",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="tournament",
        help="Método de selección de padres (fijo para todos)",
    )
    parser.add_argument(
        "--survival-selection",
        type=str,
        default="elite",
        help="Método de selección para supervivientes",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[c["id"] for c in SURVIVAL_CONFIGS],
        default=None,
        help="Configuraciones a comparar (por defecto todas)",
    )
    return parser.parse_args()


def get_config_by_id(config_id: str) -> dict:
    """Obtiene una configuración por su ID."""
    for cfg in SURVIVAL_CONFIGS:
        if cfg["id"] == config_id:
            return cfg
    raise ValueError(f"Configuración no encontrada: {config_id}")


def run_config(
    cfg: dict,
    target_image: Image.Image,
    evo_config: EvolutionConfig,
    selection: str,
    survival_selection: str,
) -> dict:
    """Corre una configuración de supervivencia y devuelve los resultados."""
    engine = create_engine(
        target_image=target_image,
        config=evo_config,
        selection_method=selection,
        tournament_size=3,
        crossover_method="uniform",
        crossover_probability=0.8,
        mutation_params=MutationParams(probability=0.3, gene_probability=0.1),
        threshold=0.75,
        boltzmann_t0=100.0,
        boltzmann_tc=1.0,
        boltzmann_k=0.005,
        survival_method=cfg["method"],
        survival_selection_method=survival_selection,
        offspring_ratio=cfg["offspring_ratio"],
    )

    result = engine.run()

    return {
        "id": cfg["id"],
        "label": cfg["label"],
        "method": cfg["method"],
        "offspring_ratio": cfg["offspring_ratio"],
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

    # Colores distintos para aditiva vs exclusiva
    colors_exclusive = ["#1f77b4", "#2ca02c", "#9467bd"]  # Azules/verdes
    colors_additive = ["#ff7f0e", "#d62728", "#8c564b"]  # Naranjas/rojos

    color_map = {}
    exc_idx = 0
    add_idx = 0
    for r in results:
        if r["method"] == "exclusive":
            color_map[r["id"]] = colors_exclusive[exc_idx % len(colors_exclusive)]
            exc_idx += 1
        else:
            color_map[r["id"]] = colors_additive[add_idx % len(colors_additive)]
            add_idx += 1

    # --- Gráfico 1: evolución del fitness ---
    ax = axes[0]
    for r in results:
        gens = [h["generation"] for h in r["history"]]
        best = [h["best_fitness"] for h in r["history"]]
        linestyle = "-" if r["method"] == "exclusive" else "--"
        ax.plot(
            gens,
            best,
            label=r["label"],
            linewidth=1.8,
            color=color_map[r["id"]],
            linestyle=linestyle,
        )

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (1 / (1 + MSE))")
    ax.set_title("Evolución del Fitness — Mejor individuo")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Gráfico 2: barras de fitness final ---
    ax2 = axes[1]
    names = [r["label"] for r in results]
    fitnesses = [r["best_fitness"] for r in results]
    times = [r["elapsed_time"] for r in results]
    colors = [color_map[r["id"]] for r in results]

    bars = ax2.barh(names, fitnesses, color=colors)
    ax2.set_xlabel("Fitness final")
    ax2.set_title("Fitness final por estrategia")
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
        "Comparativa de Estrategias de Supervivencia", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    path = output_dir / "comparativa_fitness.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Gráfico guardado: {path}")


def save_result_images(results: list, output_dir: Path):
    """Guarda la imagen final de cada configuración."""
    images_dir = output_dir / "imagenes"
    images_dir.mkdir(exist_ok=True)

    for r in results:
        path = images_dir / f"{r['id']}.png"
        save_result_image(r["best_individual"], r["width"], r["height"], path)


def print_summary_table(results: list):
    """Imprime tabla resumen en consola."""
    col_w = 25
    print()
    print("=" * 75)
    print("RESUMEN COMPARATIVA - ESTRATEGIAS DE SUPERVIVENCIA")
    print("=" * 75)
    print(
        f"{'Estrategia':<{col_w}} {'Método':<12} {'K/N':>6} {'Fitness':>12} {'Tiempo':>10}"
    )
    print("-" * 75)

    sorted_results = sorted(results, key=lambda r: r["best_fitness"], reverse=True)
    for i, r in enumerate(sorted_results):
        medal = ["1", "2", "3"][i] if i < 3 else " "
        print(
            f"{medal}  {r['label']:<{col_w - 3}} "
            f"{r['method']:<12} "
            f"{r['offspring_ratio']:>6.1f} "
            f"{r['best_fitness']:>12.6f} "
            f"{r['elapsed_time']:>9.2f}s"
        )

    print("=" * 75)
    best = sorted_results[0]
    print(f"Ganador: {best['label']}  (fitness {best['best_fitness']:.6f})")
    print("=" * 75)


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
    print(f"Selección padres: {args.selection}")
    print(f"Selección supervivientes: {args.survival_selection}")
    print()

    config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
    )

    # Determinar qué configuraciones correr
    if args.configs:
        configs_to_run = [get_config_by_id(cid) for cid in args.configs]
    else:
        configs_to_run = SURVIVAL_CONFIGS

    print(f"Estrategias a comparar: {len(configs_to_run)}")
    for cfg in configs_to_run:
        print(f"  - {cfg['label']}")
    print()

    results = []
    total = len(configs_to_run)
    for idx, cfg in enumerate(configs_to_run, 1):
        print(f"[{idx}/{total}] {cfg['label']}...")
        r = run_config(
            cfg, target_image, config, args.selection, args.survival_selection
        )
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
