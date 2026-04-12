"""Renderizado dedicado de elipses sobre lienzo blanco."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from src.genetic.individual import Ellipse, Individual


class EllipseCanvas:
    """Lienzo CPU para renderizar individuos compuestos por elipses."""

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Las dimensiones deben ser positivas")

        self.width = width
        self.height = height

    @classmethod
    def from_image(cls, image: Image.Image) -> EllipseCanvas:
        return cls(width=image.width, height=image.height)

    def render(self, individual: Individual) -> Image.Image:
        return Image.fromarray(self.render_to_array(individual), mode="RGB")

    def render_to_array(self, individual: Individual) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 255.0, dtype=np.float32)

        for gene in individual.triangles:
            self._draw_ellipse(canvas, gene)

        return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)

    def _draw_ellipse(self, canvas: np.ndarray, ellipse: Ellipse):
        (cx, cy), (rx, ry), angle, abs_color = ellipse.to_absolute(
            self.width, self.height
        )
        r, g, b, alpha_255 = abs_color
        alpha = alpha_255 / 255.0
        if alpha <= 0:
            return

        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        half_w = math.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
        half_h = math.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)

        x0 = max(0, int(math.floor(cx - half_w)) - 1)
        y0 = max(0, int(math.floor(cy - half_h)) - 1)
        x1 = min(self.width, int(math.ceil(cx + half_w)) + 2)
        y1 = min(self.height, int(math.ceil(cy + half_h)) + 2)

        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        dx = grid_x - cx
        dy = grid_y - cy
        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy

        ellipse_eq = (local_x * local_x) / max(rx * rx, 1e-6) + (
            local_y * local_y
        ) / max(ry * ry, 1e-6)
        mask = ellipse_eq <= 1.0
        if not np.any(mask):
            return

        region = canvas[y0:y1, x0:x1]
        src = np.array([r, g, b], dtype=np.float32)
        region[mask] = src * alpha + region[mask] * (1.0 - alpha)

    def save(self, individual: Individual, path: str):
        self.render(individual).save(path)
