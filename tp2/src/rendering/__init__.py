"""Modulo de renderizado para triangulos y elipses."""

from src.rendering.canvas import Canvas, TriangleCanvas, load_target_image, resize_image
from src.rendering.ellipse_canvas import EllipseCanvas
from src.rendering.factory import Renderer, create_renderer
from src.rendering.gpu_canvas import GPUCanvas, TriangleGPUCanvas
from src.rendering.gpu_ellipse_canvas import EllipseGPUCanvas

__all__ = [
    "Canvas",
    "TriangleCanvas",
    "EllipseCanvas",
    "GPUCanvas",
    "TriangleGPUCanvas",
    "EllipseGPUCanvas",
    "Renderer",
    "create_renderer",
    "load_target_image",
    "resize_image",
]
