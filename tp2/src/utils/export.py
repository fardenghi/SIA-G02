"""
Exportación de resultados.

Generación de archivos de salida: imágenes, JSON, métricas, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from src.genetic.individual import Individual
from src.rendering.canvas import Canvas


def save_result_image(
    individual: Individual, width: int, height: int, path: str | Path
):
    """
    Guarda la imagen renderizada del individuo.

    Args:
        individual: Individuo a renderizar.
        width: Ancho de la imagen.
        height: Alto de la imagen.
        path: Ruta de salida.
    """
    canvas = Canvas(width=width, height=height)
    canvas.save(individual, str(path))


def export_triangles_json(individual: Individual, path: str | Path):
    """
    Exporta los triángulos a un archivo JSON.

    Args:
        individual: Individuo con los triángulos.
        path: Ruta del archivo JSON.
    """
    data = individual.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_triangles_json(path: str | Path) -> Individual:
    """
    Carga un individuo desde un archivo JSON.

    Args:
        path: Ruta del archivo JSON.

    Returns:
        Individuo reconstruido.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Individual.from_dict(data)


def save_fitness_plot(history: List[Dict[str, Any]], path: str | Path):
    """
    Genera y guarda un gráfico de la evolución del fitness.

    Args:
        history: Historial de estadísticas por generación.
        path: Ruta del archivo de imagen.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib no disponible, omitiendo gráfico de fitness")
        return

    generations = [h["generation"] for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    avg_fitness = [h["avg_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, best_fitness, label="Mejor fitness", linewidth=2)
    ax.plot(generations, avg_fitness, label="Fitness promedio", linewidth=1, alpha=0.7)

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (1 / (1 + MSE))")
    ax.set_title("Evolución del Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Escala logarítmica si hay mucha variación
    if best_fitness[0] > 0 and best_fitness[-1] > 0:
        ratio = best_fitness[-1] / best_fitness[0]
        if ratio > 100:
            ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_metrics_csv(history: List[Dict[str, Any]], path: str | Path):
    """
    Guarda las métricas en formato CSV.

    Args:
        history: Historial de estadísticas.
        path: Ruta del archivo CSV.
    """
    import csv

    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def print_summary(result, output_dir: Path):
    """
    Imprime un resumen de los resultados.

    Args:
        result: Resultado de la evolución.
        output_dir: Directorio donde se guardaron los archivos.
    """
    print("=" * 50)
    print("RESUMEN DE RESULTADOS")
    print("=" * 50)
    print(f"Generaciones ejecutadas: {result.generations}")
    print(f"Tiempo de ejecución: {result.elapsed_time:.2f} segundos")
    print(f"Fitness final: {result.best_fitness:.6f}")
    print(f"Triángulos: {len(result.best_individual)}")

    if result.stopped_early:
        print("Estado: Parada temprana (umbral alcanzado)")
    else:
        print("Estado: Generaciones máximas alcanzadas")

    print()
    print("Archivos generados:")
    print(f"  - Imagen: {output_dir / 'result.png'}")
    print(f"  - Triángulos: {output_dir / 'triangles.json'}")
    print(f"  - Gráfico: {output_dir / 'fitness_evolution.png'}")
    print("=" * 50)


def create_animation_frames(history_dir: Path, output_path: Path, fps: int = 10):
    """
    Crea una animación GIF a partir de frames guardados.

    Args:
        history_dir: Directorio con las imágenes intermedias.
        output_path: Ruta del GIF de salida.
        fps: Frames por segundo.
    """
    from PIL import Image

    # Buscar todas las imágenes de generación
    frames = sorted(history_dir.glob("gen_*.png"))

    if not frames:
        print("No se encontraron frames para la animación")
        return

    images = [Image.open(f) for f in frames]

    # Agregar la imagen final
    result_path = history_dir / "result.png"
    if result_path.exists():
        # Agregar varios frames del resultado final
        final_img = Image.open(result_path)
        images.extend([final_img] * (fps * 2))  # 2 segundos en el final

    # Guardar como GIF
    duration = int(1000 / fps)  # ms por frame
    images[0].save(
        output_path, save_all=True, append_images=images[1:], duration=duration, loop=0
    )

    print(f"Animación guardada en: {output_path}")
