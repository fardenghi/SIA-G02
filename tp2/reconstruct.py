"""Script para reconstruir una imagen a partir de formas exportadas."""

import argparse
import sys
from pathlib import Path

from src.rendering import create_renderer
from src.utils.export import load_shapes_json


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Reconstruye una imagen desde un archivo JSON de formas"
    )

    parser.add_argument("input", type=str, help="Archivo JSON con las formas")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reconstructed.png",
        help="Ruta de la imagen de salida",
    )

    parser.add_argument(
        "--width", "-W", type=int, default=256, help="Ancho de la imagen"
    )

    parser.add_argument(
        "--height", "-H", type=int, default=256, help="Alto de la imagen"
    )

    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Factor de escala (ej: 2.0 para doble resolución)",
    )

    parser.add_argument(
        "--renderer",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Backend de renderizado para la reconstrucción",
    )

    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()

    # Verificar archivo de entrada
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Archivo no encontrado: {input_path}", file=sys.stderr)
        return 1

    # Cargar formas
    print(f"Cargando formas desde: {input_path}")
    individual = load_shapes_json(input_path)
    label = "Triángulos" if individual.shape_type == "triangle" else "Elipses"
    print(f"{label} cargadas: {len(individual)}")

    # Calcular dimensiones
    width = int(args.width * args.scale)
    height = int(args.height * args.scale)

    # Renderizar
    print(f"Renderizando imagen ({width}x{height})...")
    canvas = create_renderer(
        width=width,
        height=height,
        backend=args.renderer,
        shape_type=individual.shape_type,
    )
    image = canvas.render(individual)

    # Guardar
    output_path = Path(args.output)
    image.save(output_path)
    print(f"Imagen guardada en: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
