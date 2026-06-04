"""
Script 3: Cruza Espacial (Spatial Z-Index vs Single Point).
Este script tiene dos partes:
1. Una Demostración Sintética y Visual: Crea dos padres polarizados (Padre 1:
   Triángulos Rojos a la izquierda; Padre 2: Triángulos Azules a la derecha). 
   Cruza a ambos usando Single Point y luego Spatial Z-Index para observar cómo
   el hijo de Spatial hereda geométricamente la parte izquierda del padre 1 
   y la derecha del padre 2 (preservando el layout visual original).
2. Comparación de Rendimiento Global: Ejecuta el AG una vez con cada cruza
   para comparar las curvas de fitness a lo largo de las generaciones. Exporta a CSV.
"""

import argparse
import sys
import time
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.engine import create_engine, EvolutionConfig
from src.genetic.individual import Individual, Triangle, Ellipse
from src.genetic.crossover import SpatialZIndexCrossover, SinglePointCrossover
from src.rendering.canvas import resize_image
from src.rendering import create_renderer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimizaciones [3]: Cruza Espacial (Z-Index) vs Single Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", default="input/firefox.png", help="Imagen objetivo")
    parser.add_argument("--triangles", "-t", type=int, default=100, help="Triángulos por individuo")
    parser.add_argument("--generations", "-g", type=int, default=2000, help="Generaciones máximas")
    parser.add_argument("--population", "-p", type=int, default=100, help="Tamaño de población")
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de la imagen base")
    parser.add_argument("--shape", default="triangle", help="{trianlge, ellipse}")
    parser.add_argument(
        "--output", "-o", type=str, default="output/opt_3_cruza",
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
    df_sp = df[df["strategy"] == "single_point"]
    df_szind = df[df["strategy"] == "spatial_zindex"]

    plt.figure(figsize=(10, 5))
    if not df_sp.empty:
        plt.plot(df_sp["generation"], df_sp["best_fitness"], label="Cruza Simple (Single Point)", color='orange', linewidth=2)
    if not df_szind.empty:
        plt.plot(df_szind["generation"], df_szind["best_fitness"], label="Cruza Espacial (Spatial Z-Index)", color='purple', linewidth=2)
    plt.xlabel("Generación")
    plt.ylabel("Fitness (Linear)")
    plt.title("Comparativa de Operadores de Cruza")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "evolution_cruza.png", dpi=150)
    plt.close()


def generate_polarized_triangle(side: str) -> Triangle:
    """
    Genera un triángulo aleatorio, obligándolo a vivir puramente 
    en la mitad Izquierda (left) o Derecha (right) de la imagen.
    Asigna color sólido (rojo izquierdo, azul derecho).
    """
    if side == "left":
        vertices = [
            (random.uniform(0.0, 0.5), random.uniform(0.0, 1.0)),
            (random.uniform(0.0, 0.5), random.uniform(0.0, 1.0)),
            (random.uniform(0.0, 0.5), random.uniform(0.0, 1.0))
        ]
        color = (255, 0, 0, 0.6) # Red
    else:
        vertices = [
            (random.uniform(0.5, 1.0), random.uniform(0.0, 1.0)),
            (random.uniform(0.5, 1.0), random.uniform(0.0, 1.0)),
            (random.uniform(0.5, 1.0), random.uniform(0.0, 1.0))
        ]
        color = (0, 0, 255, 0.6) # Blue

    return Triangle(vertices=vertices, color=color)


def generate_polarized_ellipse(side: str) -> Ellipse:
    """
    Genera una elipse aleatoria, obligándola a vivir puramente
    en la mitad Izquierda (left) o Derecha (right) de la imagen.
    Asigna color sólido (rojo izquierdo, azul derecho).
    """
    import math

    if side == "left":
        center = (random.uniform(0.05, 0.45), random.uniform(0.05, 0.95))
        color = (255, 0, 0, 0.6)
    else:
        center = (random.uniform(0.55, 0.95), random.uniform(0.05, 0.95))
        color = (0, 0, 255, 0.6)

    radii = (random.uniform(0.02, 0.15), random.uniform(0.02, 0.15))
    angle = random.uniform(-math.pi, math.pi)
    return Ellipse(center=center, radii=radii, angle=angle, color=color)


def demo_spatial_crossover(renderer, output_dir: Path, num_triangles: int = 50, shape_type: str = "triangle"):
    print("\n--- PASO 1: DEMO SINTÉTICA DE LA CRUZA ESPACIAL ---")

    gen_fn = generate_polarized_ellipse if shape_type == "ellipse" else generate_polarized_triangle

    # 1. Crear Padre A: Todos los genes a la izquierda y Rojos
    triangles_A = [gen_fn("left") for _ in range(num_triangles)]
    parent_A = Individual(triangles=triangles_A)

    # 2. Crear Padre B: Todos los genes a la derecha y Azules
    triangles_B = [gen_fn("right") for _ in range(num_triangles)]
    parent_B = Individual(triangles=triangles_B)

    # Renderizarlos y guardarlos
    renderer.save(parent_A, str(output_dir / "demo_01_parent_A.png"))
    renderer.save(parent_B, str(output_dir / "demo_02_parent_B.png"))

    # 3. Cruzar mediante Single Point
    sp_crossover = SinglePointCrossover(probability=1.0)
    sp_child1, sp_child2 = sp_crossover.crossover(parent_A, parent_B)

    renderer.save(sp_child1, str(output_dir / "demo_03_child_SinglePoint.png"))

    # 4. Cruzar mediante Spatial Z-Index
    sz_crossover = SpatialZIndexCrossover(probability=1.0)
    sz_child1, sz_child2 = sz_crossover.crossover(parent_A, parent_B)

    renderer.save(sz_child1, str(output_dir / "demo_04_child_SpatialZIndex.png"))

    print(" ¡Demo completada! Revisa:")
    print(f"  - {output_dir}/demo_01_parent_A.png (Padre Rojo/Izquierdo)")
    print(f"  - {output_dir}/demo_02_parent_B.png (Padre Azul/Derecho)")
    print(f"  - {output_dir}/demo_03_child_SinglePoint.png (Mezcla cruda de arrays)")
    print(f"  - {output_dir}/demo_04_child_SpatialZIndex.png (Mantiene la coherencia espacial)")
    print()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "cruza_history.csv"
    
    if args.from_csv:
        if not csv_path.exists():
            print(f"Error: {csv_path} no encontrado.")
            return 1
        print(f"[{Path(__file__).name}] Re-generando gráficos desde CSV...")
        df = pd.read_csv(csv_path)
        plot_comparison(df, output_dir)
        print(f"Todo re-guardado en {output_dir}/")
        return 0

    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)
    target_image.save(output_dir / "target.png")

    renderer = create_renderer(
        width=target_image.width,
        height=target_image.height,
        backend=args.backend,
        shape_type=args.shape,
    )

    # 1. Demostración Macroscópica
    demo_spatial_crossover(renderer, output_dir, args.triangles, shape_type=args.shape)

    # 2. Ejecución Global de Comparación
    print("--- PASO 2: RENDIMIENTO AG (Single Point vs Spatial) ---")
    
    all_history = []
    labels = ["single_point", "spatial_zindex"]

    for cx in labels:
        print(f" Corriendo con cruza: {cx}")
        evo_config = EvolutionConfig(
            population_size=args.population,
            num_triangles=args.triangles,
            max_generations=args.generations,
            shape_type=args.shape,
        )

        engine = create_engine(
            target_image=target_image,
            config=evo_config,
            selection_method="tournament",
            crossover_method=cx,
            mutation_method="uniform_multigen",
            survival_method="additive",
            survival_selection_method="elite",
            fitness_method="linear",
            renderer=renderer,
        )
        
        def on_generation(gen: int, pop, stats: dict):
            all_history.append({
                "strategy": cx,
                "generation": gen,
                "best_fitness": stats["best_fitness"],
                "avg_fitness": stats["avg_fitness"]
            })
            if gen > 0 and gen % (max(1, args.generations // 10)) == 0:
                print(f"   [{cx}] Gen {gen:5d} -> Fitness: {stats['best_fitness']:.5f}")

        engine.on_generation(on_generation)

        start_t = time.time()
        res = engine.run()
        print(f"  -> Listo: {time.time() - start_t:.1f}s | Mejor Fitness Final: {res.best_fitness:.5f}")
        
        # Save end image
        renderer.save(res.best_individual, str(output_dir / f"final_{cx}.png"))

    df = pd.DataFrame(all_history)
    df.to_csv(csv_path, index=False)
    print(f"Datos exportados a {csv_path}")

    print("Generando gráfico comparativo...")
    plot_comparison(df, output_dir)
    print(f"Todo guardado en {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
