"""
Renderizado de triángulos sobre lienzo.

Generación de imágenes a partir de listas de triángulos usando Pillow.
Los triángulos se renderizan con transparencia (alpha blending) sobre
un lienzo blanco.
"""

from __future__ import annotations

from typing import Tuple, List
from PIL import Image, ImageDraw
import numpy as np

from src.genetic.individual import Individual, Triangle


class Canvas:
    """
    Lienzo para renderizar triángulos.

    Renderiza una lista de triángulos con transparencia sobre un
    fondo blanco, respetando el orden Z-index (los últimos triángulos
    se dibujan encima).

    Attributes:
        width: Ancho del lienzo en píxeles.
        height: Alto del lienzo en píxeles.
    """

    def __init__(self, width: int, height: int):
        """
        Inicializa el lienzo.

        Args:
            width: Ancho en píxeles.
            height: Alto en píxeles.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Las dimensiones deben ser positivas")

        self.width = width
        self.height = height

    @classmethod
    def from_image(cls, image: Image.Image) -> Canvas:
        """
        Crea un lienzo con las dimensiones de una imagen.

        Args:
            image: Imagen PIL de referencia.

        Returns:
            Canvas con las mismas dimensiones.
        """
        return cls(width=image.width, height=image.height)

    def render(self, individual: Individual) -> Image.Image:
        """
        Renderiza un individuo (lista de triángulos) a imagen.

        Args:
            individual: Individuo a renderizar.

        Returns:
            Imagen PIL en modo RGB.
        """
        # Crear imagen base blanca en RGBA para soportar transparencia
        image = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 255))

        # Renderizar cada triángulo en orden (Z-index)
        for triangle in individual.triangles:
            self._draw_triangle(image, triangle)

        # Convertir a RGB (aplanar alpha sobre blanco)
        return image.convert("RGB")

    def render_to_array(self, individual: Individual) -> np.ndarray:
        """
        Renderiza un individuo a array NumPy.

        Args:
            individual: Individuo a renderizar.

        Returns:
            Array NumPy de forma (height, width, 3) con valores uint8.
        """
        image = self.render(individual)
        return np.array(image, dtype=np.uint8)

    def _draw_triangle(self, image: Image.Image, triangle: Triangle):
        """
        Dibuja un triángulo con transparencia sobre la imagen.

        Usa alpha compositing para mezclar el triángulo con el contenido
        existente de la imagen.

        Args:
            image: Imagen RGBA sobre la cual dibujar.
            triangle: Triángulo a dibujar.
        """
        # Convertir coordenadas normalizadas a absolutas
        abs_vertices, abs_color = triangle.to_absolute(self.width, self.height)

        # Crear capa temporal para el triángulo
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Dibujar triángulo sólido en la capa
        # Pillow espera una lista de tuplas para el polígono
        polygon = [tuple(v) for v in abs_vertices]
        draw.polygon(polygon, fill=abs_color)

        # Componer sobre la imagen principal usando alpha
        image.paste(overlay, (0, 0), overlay)

    def save(self, individual: Individual, path: str):
        """
        Renderiza y guarda un individuo como imagen.

        Args:
            individual: Individuo a renderizar.
            path: Ruta del archivo de salida.
        """
        image = self.render(individual)
        image.save(path)


def load_target_image(path: str) -> Tuple[Image.Image, np.ndarray]:
    """
    Carga una imagen objetivo para comparación.

    Args:
        path: Ruta a la imagen.

    Returns:
        Tupla (imagen_pil, array_numpy) donde el array tiene forma
        (height, width, 3) con valores uint8.
    """
    image = Image.open(path).convert("RGB")
    array = np.array(image, dtype=np.uint8)
    return image, array


def resize_image(image: Image.Image, max_size: int = 256) -> Image.Image:
    """
    Redimensiona una imagen manteniendo la relación de aspecto.

    Args:
        image: Imagen a redimensionar.
        max_size: Tamaño máximo del lado más largo.

    Returns:
        Imagen redimensionada.
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image.copy()

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
