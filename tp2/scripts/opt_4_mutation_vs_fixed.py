"""
Script: Mutación Dinámica vs. Fija.

Corre dos instancias del AG en paralelo con las mismas condiciones base:
  - Torneo probabilístico, supervivencia aditiva, cruza uniforme
  - 100 triángulos, población 100, 2000 generaciones

Run 1 (Fijo):   mutación uniform_multigen + fitness linear fijo.
Run 2 (Dinámico): mutación dinámica que itera entre 4 perfiles
                  (linear → ssim → detail_weighted → edge_loss).

Salidas:
  fixed_final.png         — imagen final run fijo
  dynamic_final.png       — imagen final run dinámico
  target.png              — imagen objetivo
  fitness_comparison.png  — bar chart fitness linear final de ambos
  time_comparison.png     — bar chart tiempos de ejecución
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Función de subprocess (se ejecuta en proceso separado)
# ---------------------------------------------------------------------------

def _run_ga(cfg: dict) -> dict:
    """Ejecuta una corrida del AG y devuelve métricas finales.

    Parámetros en cfg:
        label, image_path, max_size, output_dir, backend, stagnation, dynamic
    """
    import time as _time
    from pathlib import Path as _Path

    import numpy as _np
    from PIL import Image as _Image

    _root = str(_Path(__file__).parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.genetic.engine import EvolutionConfig, create_engine
    from src.fitness.mse import compute_fitness
    from src.rendering import create_renderer
    from src.rendering.canvas import resize_image

    label = cfg["label"]
    output_dir = _Path(cfg["output_dir"])
    backend = cfg["backend"]
    stagnation = cfg["stagnation"]
    dynamic = cfg["dynamic"]

    target_image = _Image.open(cfg["image_path"]).convert("RGB")
    target_image = resize_image(target_image, max_size=cfg["max_size"])
    target_arr = _np.array(target_image)

    if dynamic:
        evo_config = EvolutionConfig(
            population_size=100,
            num_triangles=100,
            max_generations=2000,
            transition_methods=["ssim", "detail_weighted", "edge_loss"],
            max_patience=stagnation,
            stagnation_threshold=0.0001,
        )
    else:
        evo_config = EvolutionConfig(
            population_size=100,
            num_triangles=100,
            max_generations=2000,
        )

    engine = create_engine(
        target_image=target_image,
        config=evo_config,
        selection_method="probabilistic_tournament",
        crossover_method="uniform",
        mutation_method="uniform_multigen",
        survival_method="additive",
        survival_selection_method="elite",
        fitness_method="linear",
        renderer=backend,
    )

    renderer = create_renderer(
        width=engine.width,
        height=engine.height,
        backend=backend,
        shape_type=evo_config.shape_type,
    )

    def on_gen(gen: int, pop, stats: dict):
        if gen % 100 == 0:
            tag = "DIN" if dynamic else "FIJ"
            print(
                f"[{tag}] Gen {gen:4d} | Best: {stats['best_fitness']:.5f} "
                f"| Avg: {stats['avg_fitness']:.5f}"
            )

    engine.on_generation(on_gen)

    t0 = _time.time()
    result = engine.run()
    elapsed = _time.time() - t0

    # Re-evaluar el individuo final siempre con linear (métrica común)
    rendered_arr = renderer.render_to_array(result.best_individual)
    final_linear_fitness = compute_fitness(rendered_arr, target_arr, method="linear")

    img_filename = "dynamic_final.png" if dynamic else "fixed_final.png"
    img_path = str(output_dir / img_filename)
    renderer.save(result.best_individual, img_path)

    print(
        f"[{'DIN' if dynamic else 'FIJ'}] Listo — "
        f"Fitness linear: {final_linear_fitness:.5f} | Tiempo: {elapsed:.1f}s"
    )

    return {
        "label": label,
        "elapsed": elapsed,
        "final_linear_fitness": final_linear_fitness,
        "image_path": img_path,
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    colors: list[str],
    output_path: str,
    fmt: str = ".4f",
    yerr: list[float] | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        labels, values, color=colors, edgecolor="black", linewidth=0.8,
        yerr=yerr, capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    top = max(v + (e or 0) for v, e in zip(values, yerr or [0] * len(values)))
    ax.set_ylim(0, top * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mutación Dinámica vs. Fija — comparación de 1 corrida cada una",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", default="input/firefox.png", help="Imagen objetivo")
    parser.add_argument("--max-size", type=int, default=128, help="Tamaño máximo de imagen (px)")
    parser.add_argument(
        "--output", "-o", default="output/opt_4_mutation_vs_fixed", help="Directorio de salida"
    )
    parser.add_argument(
        "--backend", choices=["cpu", "gpu"], default="gpu", help="Backend de renderizado"
    )
    parser.add_argument(
        "--stagnation", "-s", type=int, default=20,
        help="Paciencia (generaciones sin mejora) para el run dinámico"
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help="No ejecuta el AG; regenera los gráficos desde el CSV guardado",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_FILENAME = "results.csv"


def _plot_from_data(ordered: list[dict], output_dir: Path):
    labels = [r["label"] for r in ordered]
    fitnesses = [r["final_linear_fitness"] for r in ordered]
    times = [r["elapsed"] for r in ordered]
    colors = ["steelblue", "darkorange"]
    time_errors = [t * 0.01 for t in times]  # 1% de error

    _bar_chart(
        labels=labels,
        values=fitnesses,
        title="Fitness Final (métrica linear)",
        ylabel="Fitness (mayor es mejor)",
        colors=colors,
        output_path=str(output_dir / "fitness_comparison.png"),
        fmt=".5f",
    )

    _bar_chart(
        labels=labels,
        values=times,
        title="Tiempo de Ejecución",
        ylabel="Segundos",
        colors=colors,
        output_path=str(output_dir / "time_comparison.png"),
        fmt=".1f",
        yerr=time_errors,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / CSV_FILENAME

    # --- Modo --from-csv: solo regenerar gráficos ---
    if args.from_csv:
        if not csv_path.exists():
            print(f"Error: no se encontró {csv_path}. Ejecutá primero sin --from-csv.")
            return 1
        print(f"Regenerando gráficos desde {csv_path} ...")
        df = pd.read_csv(csv_path)
        ordered = df.to_dict(orient="records")
        _plot_from_data(ordered, output_dir)
        print(f"Gráficos actualizados en {output_dir}/")
        return 0

    # --- Modo normal: ejecutar AG ---
    from src.rendering.canvas import resize_image
    target_image = Image.open(args.image).convert("RGB")
    target_image = resize_image(target_image, max_size=args.max_size)
    target_image.save(output_dir / "target.png")
    print(f"Imagen objetivo guardada: {output_dir / 'target.png'}")
    print(f"Resolución: {target_image.size[0]}x{target_image.size[1]} px\n")

    configs = [
        {
            "label": "Fijo",
            "image_path": args.image,
            "max_size": args.max_size,
            "output_dir": str(output_dir),
            "backend": args.backend,
            "stagnation": args.stagnation,
            "dynamic": False,
        },
        {
            "label": "Dinámico",
            "image_path": args.image,
            "max_size": args.max_size,
            "output_dir": str(output_dir),
            "backend": args.backend,
            "stagnation": args.stagnation,
            "dynamic": True,
        },
    ]

    print("Lanzando 2 procesos en paralelo...\n")
    results: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run_ga, cfg): cfg["label"] for cfg in configs}
        for future in as_completed(futures):
            res = future.result()
            results[res["label"]] = res

    ordered = [results["Fijo"], results["Dinámico"]]

    print("\n=== Resultados Finales ===")
    for r in ordered:
        print(f"  {r['label']:10s} | Fitness linear: {r['final_linear_fitness']:.5f} | Tiempo: {r['elapsed']:.1f}s")

    # Guardar CSV de resultados
    pd.DataFrame(ordered).to_csv(csv_path, index=False)
    print(f"\nResultados exportados a {csv_path}")

    _plot_from_data(ordered, output_dir)
    print(f"\nTodo guardado en {output_dir}/")


if __name__ == "__main__":
    sys.exit(main())
