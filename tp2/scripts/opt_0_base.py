"""
Script 0: Baseline.
Ejecuta el Algoritmo Genético con la configuración base:
- Fitness: linear
- Selección: torneo probabilístico
- Cruza: single point
- Mutación: uniforme multigen
- Supervivencia: aditiva

Se encarga de guardar la evolución del fitness y capturas visuales.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.engine import create_engine, EvolutionConfig
from src.rendering.canvas import resize_image
from src.rendering import create_renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimizaciones [0]: Ejecución Baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", default="input/firefox.png", help="Imagen objetivo")
    parser.add_argument("--triangles", "-t", type=int, default=100, help="Triángulos por individuo")
    parser.add_argument("--generations", "-g", type=int, default=2000, help="Generaciones máximas")
    parser.add_argument("--population", "-p", type=int, default=100, help="Tamaño de población")
    parser.add_argument("--save-interval", type=int, default=50, help="Guardar imagen cada N generaciones")
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de la imagen base")
    parser.add_argument(
        "--output", "-o", type=str, default="output/opt_0_base",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--backend", type=str, default="gpu", choices=["cpu", "gpu"],
        help="Backend de renderizado",
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Si se especifica, no ejecuta el AG y solo recrea los gráficos usando los CSV gurdados.",
    )
    return parser.parse_args()


def plot_fitness(df: pd.DataFrame, output_dir: Path):
    gens = df["generation"]
    bests = df["best_fitness"]
    avgs = df["avg_fitness"]

    plt.figure(figsize=(10, 5))
    plt.plot(gens, bests, label="Mejor Fitness", color='green', linewidth=2)
    plt.plot(gens, avgs, label="Fitness Promedio", color='blue', alpha=0.5, linestyle='--')
    plt.xlabel("Generación")
    plt.ylabel("Fitness (Linear)")
    plt.title("Evolución del Fitness (Baseline)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "evolution.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "base_history.csv"

    if args.from_csv:
        if not csv_path.exists():
            print(f"Error: {csv_path} no encontrado, debes ejecutar normalmente primero.")
            return 1
        print(f"[{Path(__file__).name}] Re-generando gráficos desde CSV...")
        df = pd.read_csv(csv_path)
        plot_fitness(df, output_dir)
        print(f"Todo re-guardado en {output_dir}/")
        return 0

    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)

    print(f"[{Path(__file__).name}] Iniciando Baseline")
    print(f" Imagen: {args.image} ({target_image.width}x{target_image.height})")
    print(f" Generaciones: {args.generations} | Tribus: {args.population} | Tris: {args.triangles}")
    print()

    # Configuración BASE
    evo_config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
    )

    engine = create_engine(
        target_image=target_image,
        config=evo_config,
        selection_method="probabilistic_tournament",
        crossover_method="single_point",
        mutation_method="uniform_multigen",
        survival_method="additive",
        survival_selection_method="elite",
        fitness_method="linear",
        renderer=args.backend,
    )

    renderer = create_renderer(
        width=engine.width,
        height=engine.height,
        backend=args.backend,
        shape_type=evo_config.shape_type,
    )

    start_t = time.time()
    
    history_records = []

    def on_generation(gen: int, pop, stats: dict):
        history_records.append({
            "generation": gen,
            "best_fitness": stats["best_fitness"],
            "avg_fitness": stats["avg_fitness"]
        })

        if gen % 10 == 0:
            print(f" Gen {gen:5d} | Best: {stats['best_fitness']:10.6f} | Avg: {stats['avg_fitness']:10.6f}")

        if args.save_interval > 0 and gen > 0 and gen % args.save_interval == 0:
            best_ind = pop.best
            filename = output_dir / f"gen_{gen:05d}.png"
            renderer.save(best_ind, str(filename))

    engine.on_generation(on_generation)

    print("Corriendo motor genético...")
    # Render inicial y ejecución
    result = engine.run()

    # Guardar final target image and the result image explicitly
    target_image.save(output_dir / "target.png")
    renderer.save(result.best_individual, str(output_dir / "final_result.png"))

    print(f"\nTerminado en {time.time() - start_t:.1f}s.")
    print(f"Mejor fitness final: {result.best_fitness:.6f}")

    # Exportar CSV
    df = pd.DataFrame(history_records)
    df.to_csv(csv_path, index=False)
    print(f"Datos exportados a {csv_path}")

    print("Generando gráfico de evolución...")
    plot_fitness(df, output_dir)
    print(f"Todo guardado en {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
