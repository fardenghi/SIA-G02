"""
Script 2: Complejidad de Funciones de Fitness.
Ejecuta el Algoritmo Genético variando únicamente la métrica de Fitness:
1. ssim
2. detail_weighted
3. edge_loss
4. Dinámica (Arranca en linear, y rota entre las otras en caso de estancamiento).

Guarda 5 imágenes a lo largo de la corrida (0%, 25%, 50%, 75%, 100%) para cada
métrica, permitiendo compararlas visualmente al final. Guara los historiales 
en opt_2_history.csv.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.engine import create_engine, EvolutionConfig
from src.rendering.canvas import resize_image
from src.rendering import create_renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimizaciones [2]: Análisis Visial de Métricas de Fitness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", default="input/firefox.png", help="Imagen objetivo")
    parser.add_argument("--triangles", "-t", type=int, default=100, help="Triángulos por individuo")
    parser.add_argument("--generations", "-g", type=int, default=2000, help="Generaciones máximas")
    parser.add_argument("--population", "-p", type=int, default=100, help="Tamaño de población")
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de la imagen base")
    parser.add_argument(
        "--output", "-o", type=str, default="output/opt_2_fitness",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--backend", type=str, default="gpu", choices=["cpu", "gpu"],
        help="Backend de renderizado",
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Si se especifica, no ejecuta el AG, sólo avisa que los CSV están listos (para este script las imágenes son lo principal).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "opt_2_history.csv"
    
    if args.from_csv:
        if not csv_path.exists():
            print(f"Error: {csv_path} no encontrado.")
            return 1
        print(f"[{Path(__file__).name}] Los CSV ya fueron generados. En este script las imágenes son el principal output, no se redibujan plots.")
        return 0

    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)

    # Renderer para extraer las imágenes
    renderer = create_renderer(
        width=target_image.width,
        height=target_image.height,
        backend=args.backend,
        shape_type="triangle",
    )

    target_image.save(output_dir / "target.png")

    runs = [
        {"name": "ssim", "fitness_method": "ssim", "dynamic": False},
        {"name": "detail_weighted", "fitness_method": "detail_weighted", "dynamic": False},
        {"name": "edge_loss", "fitness_method": "edge_loss", "dynamic": False},
        {"name": "dinamica", "fitness_method": "linear", "dynamic": True},
    ]

    snapshots_at = {
        0, 
        max(1, int(args.generations * 0.25)), 
        max(1, int(args.generations * 0.50)), 
        max(1, int(args.generations * 0.75)), 
        args.generations
    }

    print(f"[{Path(__file__).name}] Iniciando Analisis de Fitness")
    print(f" Sacando snapshots en las generaciones: {sorted(list(snapshots_at))}")
    print()
    
    all_history = []

    for r in runs:
        run_name = r["name"]
        print(f"--- Corriendo: {run_name.upper()} ---")

        # Si es dinámica, agregamos la lista de transiciones
        transitions = ["detail_weighted", "edge_loss", "ssim"] if r["dynamic"] else None

        evo_config = EvolutionConfig(
            population_size=args.population,
            num_triangles=args.triangles,
            max_generations=args.generations,
            transition_methods=transitions
        )

        engine = create_engine(
            target_image=target_image,
            config=evo_config,
            selection_method="probabilistic_tournament",
            crossover_method="single_point",
            mutation_method="uniform_multigen",
            survival_method="additive",
            survival_selection_method="elite",
            fitness_method=r["fitness_method"],
            renderer=renderer,
        )

        def make_callback(run_id: str):
            def on_generation(gen: int, pop, stats: dict):
                
                all_history.append({
                    "strategy": run_id,
                    "generation": gen,
                    "active_method": engine.active_fitness_method,
                    "best_fitness": stats["best_fitness"],
                    "avg_fitness": stats["avg_fitness"]
                })
                
                # Print local logs
                if gen % (max(1, args.generations // 10)) == 0:
                    print(f"   [{run_id}] Gen {gen:5d} -> Fitness local: {stats['best_fitness']:.5f} ({engine.active_fitness_method})")
                
                # Snapshot si macha la etapa
                if gen in snapshots_at:
                    img_name = f"{run_id}_gen_{gen:04d}_{engine.active_fitness_method}.png"
                    renderer.save(pop.best, str(output_dir / img_name))

            return on_generation

        engine.on_generation(make_callback(run_name))

        start_t = time.time()
        res = engine.run()
        print(f" -> Listo. Tiempo: {time.time() - start_t:.1f}s\n")
        
        # Save last in case it stopped early
        final_img = f"{run_name}_final_{engine.active_fitness_method}.png"
        renderer.save(res.best_individual, str(output_dir / final_img))

    # Guardar CSV
    df = pd.DataFrame(all_history)
    df.to_csv(csv_path, index=False)
    print(f"Datos exportados a {csv_path}")

    print(f"Todas las variantes han completado. Revisa las imagenes en: {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
