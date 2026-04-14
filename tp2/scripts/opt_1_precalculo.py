"""
Script 1: Precalculo.
Ejecuta el Algoritmo Genético con la configuración base dos veces:
1. Sin precalculo (Grid Seeding = 0.0) -> Población inicial 100% aleatoria.
2. Con precalculo (Grid Seeding = X)   -> Parte de la población iniciada inteligentemente.

Compara el fitness inicial (Gen 0) y el progreso evolutivo general entre ambos.
Se extraen imágenes de la generación 0 (para contrastar visualmente el punto de partida)
y se exportan los datos a un CSV para graficarlos.
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
        description="Optimizaciones [1]: Impacto del Precalculo (Grid Seeding)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", default="input/firefox.png", help="Imagen objetivo")
    parser.add_argument("--triangles", "-t", type=int, default=100, help="Triángulos por individuo")
    parser.add_argument("--generations", "-g", type=int, default=2000, help="Generaciones máximas")
    parser.add_argument("--population", "-p", type=int, default=100, help="Tamaño de población")
    parser.add_argument("--seed-ratio", "-s", type=float, default=0.1, help="Porcentaje de precalculo (ej 0.1=10%)")
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de la imagen base")
    parser.add_argument(
        "--output", "-o", type=str, default="output/opt_1_precalculo",
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


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    df_base = df[df["strategy"] == "Sin Precalculo"]
    df_seed = df[df["strategy"] != "Sin Precalculo"]
    
    seed_name = df_seed["strategy"].iloc[0] if not df_seed.empty else "Con Precalculo"

    plt.figure(figsize=(10, 5))
    plt.plot(df_base["generation"], df_base["best_fitness"], label="Aleatorio Absoluto (Base)", color='red', linewidth=2)
    plt.plot(df_seed["generation"], df_seed["best_fitness"], label=seed_name, color='blue', linewidth=2)
    plt.xlabel("Generación")
    plt.ylabel("Fitness (Linear)")
    plt.title("Impacto del Precalculo Inicial (Grid Seeding)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "evolution_precalc.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "precalc_history.csv"

    if args.from_csv:
        if not csv_path.exists():
            print(f"Error: {csv_path} no encontrado, debes ejecutar normalmente primero.")
            return 1
        print(f"[{Path(__file__).name}] Re-generando gráficos desde CSV...")
        df = pd.read_csv(csv_path)
        plot_comparison(df, output_dir)
        print(f"Todo re-guardado en {output_dir}/")
        return 0

    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)

    # Creamos un renderer para extraer las imágenes
    renderer = create_renderer(
        width=target_image.width,
        height=target_image.height,
        backend=args.backend,
        shape_type="triangle",
    )

    print(f"[{Path(__file__).name}] Iniciando Analisis de Precalculo")
    print()

    target_image.save(output_dir / "target.png")

    labels = [("Sin Precalculo", 0.0), (f"Con Precalculo ({args.seed_ratio*100}%)", args.seed_ratio)]
    
    all_history = []

    for name, ratio in labels:
        print(f"--- Corriendo Configuración: {name} ---")
        evo_config = EvolutionConfig(
            population_size=args.population,
            num_triangles=args.triangles,
            max_generations=args.generations,
            seed_ratio=ratio
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
            renderer=renderer,
        )

        # Usamos callback para extraer la generacion 0
        def on_generation(gen: int, pop, stats: dict):
            if gen % 100 == 0:
                print(f" [{name}] Gen {gen:5d} -> Fitness: {stats['best_fitness']:.5f}")
                
            all_history.append({
                "strategy": name,
                "generation": gen,
                "best_fitness": stats["best_fitness"],
                "avg_fitness": stats["avg_fitness"]
            })
            
            if gen == 0:
                print(f" [!] PISO INICIAL Gen 0 -> Fitness: {stats['best_fitness']:.5f}")
                img_name = "gen000_random.png" if ratio == 0.0 else "gen000_precalc.png"
                renderer.save(pop.best, str(output_dir / img_name))

        engine.on_generation(on_generation)

        start_t = time.time()
        res = engine.run()
        print(f" Listo. Tiempo: {time.time() - start_t:.1f}s | Final fitness: {res.best_fitness:.5f}\n")
        
        # Save final images as well to compare
        img_name = "final_random.png" if ratio == 0.0 else "final_precalc.png"
        renderer.save(res.best_individual, str(output_dir / img_name))

    # Guardar a CSV
    df = pd.DataFrame(all_history)
    df.to_csv(csv_path, index=False)
    print(f"Datos exportados a {csv_path}")

    print("Generando gráfico comparativo...")
    plot_comparison(df, output_dir)
    print(f"Resultados guardados en {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
