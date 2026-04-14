"""
Script 4: Perfiles de Mutación Dinámicos.
Ejecuta el Algoritmo Genético utilizando la función de fitness dinámica
y sus perfiles de mutación acoplados temporales.
Realiza seguimiento en vivo del método de mutación/fitness activo y 
captura imágenes cuando detecta un "estancamiento -> transición".
A su vez, genera un gráfico de evolución marcando con líneas verticales
los momentos exactos en los que el motor genético saltó de perfil.
Exporta todo a CSV.
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
        description="Optimizaciones [4]: Dinámica de Mutaciones Acopladas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", default="input/firefox.png", help="Imagen objetivo")
    parser.add_argument("--triangles", "-t", type=int, default=100, help="Triángulos por individuo")
    parser.add_argument("--generations", "-g", type=int, default=2000, help="Generaciones máximas")
    parser.add_argument("--population", "-p", type=int, default=100, help="Tamaño de población")
    parser.add_argument("--stagnation", "-s", type=int, default=20, help="Paciencia (Mínimo estancamiento para alternar)")
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de la imagen base")
    parser.add_argument(
        "--output", "-o", type=str, default="output/opt_4_mutacion",
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


def plot_evolution_with_transitions(df_hist: pd.DataFrame, df_trans: pd.DataFrame, output_dir: Path):
    gens = df_hist["generation"]
    bests = df_hist["best_fitness"]
    
    plt.figure(figsize=(12, 6))
    plt.plot(gens, bests, label="Mejor Fitness Acumulado", color='teal', linewidth=2)
    
    # Marcar transiciones
    colors = ['red', 'purple', 'green', 'orange']
    if not df_trans.empty:
        for idx, row in df_trans.iterrows():
            trans_gen = row["generation"]
            method_name = row["method_name"]
            c = colors[idx % len(colors)]
            plt.axvline(x=trans_gen, color=c, linestyle='--', alpha=0.7, 
                        label=f"Salto a: {method_name} (Gen {trans_gen})")

    plt.xlabel("Generación")
    plt.ylabel("Fitness (Absoluto)")
    plt.title("Evolución con Transiciones de Perfiles de Mutación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "evolution_transitions.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_hist_path = output_dir / "mutacion_history.csv"
    csv_trans_path = output_dir / "mutacion_transitions.csv"
    
    if args.from_csv:
        if not csv_hist_path.exists() or not csv_trans_path.exists():
            print(f"Error: CSVs no encontrados en {output_dir}.")
            return 1
        print(f"[{Path(__file__).name}] Re-generando gráficos desde CSV...")
        df_hist = pd.read_csv(csv_hist_path)
        df_trans = pd.read_csv(csv_trans_path)
        plot_evolution_with_transitions(df_hist, df_trans, output_dir)
        print(f"Todo re-guardado en {output_dir}/")
        return 0

    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)
    target_image.save(output_dir / "target.png")

    print(f"[{Path(__file__).name}] Iniciando Dinámica de Mutación y Fitness")
    print()

    # Transiciones (Linear inicial via fitness_method)
    transitions_list = ["ssim", "detail_weighted", "edge_loss"]
    
    evo_config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
        transition_methods=transitions_list,
        max_patience=args.stagnation,
        stagnation_threshold=0.0001
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

    # Variables de Seguimiento
    last_known_method = engine.active_fitness_method
    transition_log = []
    history_records = []
    
    # Queremos guardar una imagen justo ~5 generaciones depués de la transición
    # para ver "qué atacó rápido".
    pending_snapshots = []

    def tracking_callback(gen: int, pop, stats: dict):
        nonlocal last_known_method
        current_method = engine.active_fitness_method
        
        history_records.append({
            "generation": gen,
            "best_fitness": stats["best_fitness"],
            "avg_fitness": stats["avg_fitness"]
        })

        if current_method != last_known_method:
            print(f" [!] > MUTACIÓN DINÁMICA: Cambio de Perfil detectado a '{current_method}' en gen {gen}")
            transition_log.append({
                "generation": gen,
                "method_name": current_method
            })
            
            # Guardamos exactamente el momento pre-transición (estado de la pob)
            img_name = f"transition_{gen:04d}_PRE_{current_method}.png"
            renderer.save(pop.best, str(output_dir / img_name))
            
            # Programamos foticos 10 y 20 generaciones después para ver 
            # al cirujano/restaurador en acción
            pending_snapshots.append(gen + 10)
            pending_snapshots.append(gen + 20)
            
            last_known_method = current_method

        if gen in pending_snapshots:
            pending_snapshots.remove(gen)
            img_name = f"transition_{gen:04d}_POST_{current_method}.png"
            renderer.save(pop.best, str(output_dir / img_name))

    engine.on_generation(tracking_callback)

    start_t = time.time()
    res = engine.run()
    
    print(f" Listo. Tiempo: {time.time() - start_t:.1f}s | Mejor Fitness Final: {res.best_fitness:.5f}\n")
    renderer.save(res.best_individual, str(output_dir / "final_dynamic_mutation.png"))
    
    # Guardar CSVs
    df_hist = pd.DataFrame(history_records)
    df_hist.to_csv(csv_hist_path, index=False)
    
    df_trans = pd.DataFrame(transition_log)
    df_trans.to_csv(csv_trans_path, index=False)
    print(f"Datos exportados a {csv_hist_path} y {csv_trans_path}")

    print("Generando gráfico de transiciones...")
    plot_evolution_with_transitions(df_hist, df_trans, output_dir)
    print(f"Todo guardado en {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
