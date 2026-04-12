"""
Compresor de Imágenes Evolutivo

Punto de entrada principal del sistema.
Aproxima una imagen objetivo usando formas traslúcidas mediante
algoritmos genéticos.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.utils.config import Config, load_config
from src.genetic.engine import create_engine
from src.rendering import create_renderer, resize_image
from src.utils.export import (
    save_result_image,
    export_shapes_json,
    export_shapes_csv,
    export_triangles_json,
    export_triangles_csv,
    save_fitness_plot,
    print_summary,
)
from src.utils.metrics import MetricsTracker


def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los argumentos.
    """
    parser = argparse.ArgumentParser(
        description="Compresor de Imágenes Evolutivo - "
        "Aproxima imágenes usando triángulos o elipses traslúcidas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Argumentos requeridos
    parser.add_argument(
        "--image", "-i", type=str, required=True, help="Ruta a la imagen objetivo"
    )

    parser.add_argument(
        "--triangles", "-t", type=int, default=None, help="Cantidad de genes"
    )

    parser.add_argument(
        "--shape",
        type=str,
        choices=["triangle", "ellipse"],
        default=None,
        help="Familia de formas del individuo",
    )

    # Argumentos opcionales
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Archivo de configuración YAML",
    )

    parser.add_argument(
        "--population", "-p", type=int, default=None, help="Tamaño de la población"
    )

    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=None,
        help="Número máximo de generaciones",
    )

    parser.add_argument(
        "--mutation-rate", "-m", type=float, default=None, help="Tasa de mutación"
    )

    parser.add_argument(
        "--selection",
        type=str,
        choices=[
            "elite",
            "tournament",
            "probabilistic_tournament",
            "roulette",
            "universal",
            "boltzmann",
            "rank",
            "ranking",
        ],
        default=None,
        help="Método de selección",
    )

    parser.add_argument(
        "--crossover",
        type=str,
        choices=["single_point", "two_point", "uniform", "annular"],
        default=None,
        help="Método de cruza",
    )

    parser.add_argument(
        "--mutation",
        type=str,
        choices=[
            "single_gene",
            "limited_multigen",
            "uniform_multigen",
            "complete",
            "error_map_guided",
        ],
        default=None,
        help="Método de mutación",
    )

    parser.add_argument(
        "--guided-ratio",
        type=float,
        default=None,
        help=(
            "Fracción de mutaciones guiadas por error map (solo para error_map_guided). "
            "Rango [0,1]. Recomendado: 0.7–0.8."
        ),
    )

    parser.add_argument(
        "--survival",
        type=str,
        choices=["additive", "exclusive"],
        default=None,
        help="Estrategia de supervivencia",
    )

    parser.add_argument(
        "--fitness",
        type=str,
        choices=[
            "linear",
            "rmse",
            "inverse_normalized",
            "exponential",
            "inverse_mse",
            "detail_weighted",
            "composite",
        ],
        default=None,
        help=(
            "Función de fitness: "
            "linear=1-MSE_norm (recomendado), "
            "rmse=1-RMSE/255, "
            "inverse_normalized=1/(1+MSE_norm), "
            "exponential=exp(-MSE_norm/scale), "
            "inverse_mse=1/(1+MSE) (no recomendado, valores muy pequeños), "
            "detail_weighted=MSE ponderado por detalle"
        ),
    )

    parser.add_argument(
        "--fitness-scale",
        type=float,
        default=None,
        help="Escala para fitness exponencial (default: 0.1). Menor = más presión selectiva.",
    )

    parser.add_argument(
        "--offspring-ratio",
        type=float,
        default=None,
        help="Ratio de hijos a generar respecto a la población (K = N * ratio)",
    )

    parser.add_argument(
        "--elite-count",
        type=int,
        default=None,
        help="Cantidad de individuos élite que pasan sin modificar a la siguiente generación (0 = desactivado)",
    )

    parser.add_argument(
        "--field-probability",
        type=float,
        default=None,
        help="Probabilidad de mutar cada float individual del triángulo (1.0 = todos, <1.0 = per-float)",
    )

    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Directorio de salida"
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Guardar imagen cada N generaciones (0 = solo al final)",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=256,
        help="Tamaño máximo de la imagen (se redimensiona si es mayor)",
    )

    parser.add_argument(
        "--renderer",
        type=str,
        choices=["cpu", "gpu"],
        default=None,
        help="Backend de renderizado: cpu (Pillow, default) o gpu (moderngl/OpenGL)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Modo silencioso (menos output)"
    )

    return parser.parse_args()


def setup_callbacks(
    engine,
    config: Config,
    output_dir: Path,
    quiet: bool,
    tracker: MetricsTracker | None = None,
    renderer: str = "cpu",
):
    """
    Configura los callbacks del motor evolutivo.

    Args:
        engine: Motor genético.
        config: Configuración.
        output_dir: Directorio de salida.
        quiet: Si es True, menos output.
        tracker: Tracker de métricas pandas (opcional).
    """
    canvas = create_renderer(
        width=engine.width,
        height=engine.height,
        backend=renderer,
        shape_type=config.shape_type,
    )
    log_interval = config.output.log_interval
    save_interval = config.output.save_interval
    start_time = time.time()

    def on_generation(gen: int, population, stats: dict):
        """Callback para cada generación."""
        # Log de progreso
        if not quiet and gen % log_interval == 0:
            best = stats["best_fitness"]
            avg = stats["avg_fitness"]
            print(f"Gen {gen:5d} | Best: {best:10.6f} | Avg: {avg:10.6f}")

        # Registrar métricas con pandas
        if tracker is not None:
            tracker.record(
                generation=gen,
                stats=stats,
                elapsed=time.time() - start_time,
            )

        # Guardar imagen intermedia
        if save_interval > 0 and gen > 0 and gen % save_interval == 0:
            best_ind = population.best
            img_path = output_dir / f"gen_{gen:05d}.png"
            canvas.save(best_ind, str(img_path))

    def on_improvement(gen: int, individual, fitness: float):
        """Callback cuando hay mejora."""
        if not quiet:
            print(f"  >> Mejora en gen {gen}: {fitness:.6f}")

    engine.on_generation(on_generation)
    engine.on_improvement(on_improvement)


def main():
    """Función principal del compresor evolutivo."""
    args = parse_args()

    # Cargar configuración
    cli_args = {
        "image": args.image,
        "triangles": args.triangles,
        "shape": args.shape,
        "population": args.population,
        "generations": args.generations,
        "mutation_rate": args.mutation_rate,
        "mutation_method": args.mutation,
        "guided_ratio": args.guided_ratio,
        "selection": args.selection,
        "crossover": args.crossover,
        "survival_method": args.survival,
        "offspring_ratio": args.offspring_ratio,
        "elite_count": args.elite_count,
        "field_probability": args.field_probability,
        "output": args.output,
        "save_interval": args.save_interval,
        "fitness_method": args.fitness,
        "fitness_scale": args.fitness_scale,
        "renderer": args.renderer,
    }
    # Filtrar None
    cli_args = {k: v for k, v in cli_args.items() if v is not None}

    config = load_config(args.config, **cli_args)

    # Validar configuración
    errors = config.validate()
    if errors:
        print("Errores de configuración:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    # Cargar y preparar imagen objetivo
    if not args.quiet:
        print(f"Cargando imagen: {config.target_path}")

    target_image = Image.open(config.target_path).convert("RGB")
    original_size = target_image.size
    target_image = resize_image(target_image, max_size=args.max_size)

    if not args.quiet:
        print(f"Tamaño original: {original_size}")
        print(f"Tamaño de trabajo: {target_image.size}")
        print(f"Forma: {config.shape_type}")
        print(f"Genes: {config.num_triangles}")
        print(f"Población: {config.population_size}")
        print(f"Generaciones máximas: {config.max_generations}")
        print()

    # Crear directorio de salida
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crear tracker de métricas
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_meta = {
        "selection": config.selection.method,
        "crossover": config.crossover.method,
        "mutation": config.mutation.method,
        "shape_type": config.shape_type,
        "triangles": config.num_triangles,
        "population": config.population_size,
        "survival": config.survival.method,
        "fitness_method": config.fitness.method,
        "max_generations": config.max_generations,
    }
    tracker = MetricsTracker(run_id=run_id, config_meta=config_meta)

    # Crear motor evolutivo
    engine = create_engine(
        target_image=target_image,
        config=config.to_evolution_config(),
        selection_method=config.selection.method,
        tournament_size=config.selection.tournament_size,
        crossover_method=config.crossover.method,
        crossover_probability=config.crossover.probability,
        mutation_params=config.mutation.to_params(),
        threshold=config.selection.threshold,
        boltzmann_t0=config.selection.boltzmann_t0,
        boltzmann_tc=config.selection.boltzmann_tc,
        boltzmann_k=config.selection.boltzmann_k,
        survival_method=config.survival.method,
        survival_selection_method=config.survival.selection_method,
        offspring_ratio=config.survival.offspring_ratio,
        fitness_method=config.fitness.method,
        fitness_scale=config.fitness.exponential_scale,
        fitness_detail_weight_base=config.fitness.detail_weight_base,
        fitness_composite_alpha=config.fitness.composite_alpha,
        fitness_composite_beta=config.fitness.composite_beta,
        fitness_composite_gamma=config.fitness.composite_gamma,
        renderer=config.rendering.backend,
        adaptive_sigma=config.mutation.to_adaptive_sigma(),
    )

    # Configurar callbacks
    setup_callbacks(
        engine,
        config,
        output_dir,
        args.quiet,
        tracker=tracker,
        renderer=config.rendering.backend,
    )

    # Ejecutar evolución
    if not args.quiet:
        print("Iniciando evolución...")
        print("-" * 50)

    result = engine.run()

    if not args.quiet:
        print("-" * 50)
        print("Evolución completada!")
        print()

    # Guardar resultados
    # Guardamos 2 versiones: una en la resolución de entrenamiento y otra en la resolución de la imagen original
    save_result_image(
        result.best_individual,
        engine.width,
        engine.height,
        output_dir / "result_train_res.png",
        shape_type=config.shape_type,
        backend=config.rendering.backend,
    )
    save_result_image(
        result.best_individual,
        original_size[0],
        original_size[1],
        output_dir / "result_high_res.png",
        shape_type=config.shape_type,
        backend=config.rendering.backend,
    )
    save_result_image(
        result.best_individual,
        engine.width,
        engine.height,
        output_dir / "result.png",
        shape_type=config.shape_type,
        backend=config.rendering.backend,
    )

    if config.output.export_triangles:
        export_shapes_json(result.best_individual, output_dir / "shapes.json")
        if config.shape_type == "triangle":
            export_triangles_json(result.best_individual, output_dir / "triangles.json")

    if config.output.plot_fitness:
        save_fitness_plot(result.history, output_dir / "fitness_evolution.png")

    if config.output.export_metrics_csv:
        tracker.export_csv(output_dir / "metrics.csv")
        if not args.quiet:
            print(f"  - Métricas CSV: {output_dir / 'metrics.csv'}")

    if config.output.export_triangles_csv:
        export_shapes_csv(
            result.best_individual,
            output_dir / "shapes.csv",
            run_id=run_id,
        )
        if config.shape_type == "triangle":
            export_triangles_csv(
                result.best_individual,
                output_dir / "triangles.csv",
                run_id=run_id,
            )
        if not args.quiet:
            print(f"  - Shapes CSV: {output_dir / 'shapes.csv'}")
            if config.shape_type == "triangle":
                print(f"  - Triángulos CSV: {output_dir / 'triangles.csv'}")

    # Imprimir resumen
    print_summary(result, output_dir, shape_type=config.shape_type)

    return 0


if __name__ == "__main__":
    sys.exit(main())
