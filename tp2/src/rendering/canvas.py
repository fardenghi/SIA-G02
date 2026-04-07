"""
Renderizado de triángulos sobre lienzo.

Generación de imágenes a partir de listas de triángulos usando Pillow.
Los triángulos se renderizan con transparencia (alpha blending) sobre
un lienzo blanco.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List
from PIL import Image, ImageDraw
import numpy as np

if TYPE_CHECKING:
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

        Delega en render() (Pillow): la rasterización de polígonos y el
        alpha compositing de Pillow están implementados en C y son más
        rápidos que cualquier alternativa pura de numpy para imágenes
        pequeñas con muchos triángulos.

        Args:
            individual: Individuo a renderizar.

        Returns:
            Array NumPy de forma (height, width, 3) con valores uint8.
        """
        return np.array(self.render(individual), dtype=np.uint8)

    def _draw_triangle(self, image: Image.Image, triangle: Triangle):
        """
        Dibuja un triángulo con transparencia sobre la imagen.

        Usa alpha compositing para mezclar el triángulo con el contenido
        existente de la imagen.

        Args:
            image: Imagen RGBA sobre la cual dibujar.
            triangle: Triángulo a dibujar.
        """
        abs_vertices, abs_color = triangle.to_absolute(self.width, self.height)

        # Bounding box del triángulo: la capa temporal solo cubre esa región,
        # reduciendo el costo del paste de O(W×H) a O(bbox_area).
        vx = [v[0] for v in abs_vertices]
        vy = [v[1] for v in abs_vertices]
        x0 = max(0, min(vx) - 1)
        y0 = max(0, min(vy) - 1)
        x1 = min(self.width,  max(vx) + 2)
        y1 = min(self.height, max(vy) + 2)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            return

        overlay = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.polygon([(v[0] - x0, v[1] - y0) for v in abs_vertices], fill=abs_color)
        image.paste(overlay, (x0, y0), overlay)

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
