"""Factory de renderers segun backend y familia de formas."""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np
from PIL import Image

from src.rendering.canvas import Canvas
from src.rendering.ellipse_canvas import EllipseCanvas
from src.rendering.gpu_canvas import (
    GPUCanvas,
    MODERNGL_AVAILABLE as TRIANGLE_GPU_AVAILABLE,
)
from src.rendering.gpu_ellipse_canvas import (
    EllipseGPUCanvas,
    MODERNGL_AVAILABLE as ELLIPSE_GPU_AVAILABLE,
)


class Renderer(Protocol):
    width: int
    height: int

    def render(self, individual) -> Image.Image: ...
    def render_to_array(self, individual) -> np.ndarray: ...
    def save(self, individual, path: str): ...


def create_renderer(
    width: int,
    height: int,
    backend: str = "cpu",
    shape_type: str = "triangle",
) -> Renderer:
    shape_type = shape_type.lower()
    backend = backend.lower()

    if shape_type not in {"triangle", "ellipse"}:
        raise ValueError(f"shape_type desconocido: {shape_type}")
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"backend desconocido: {backend}")

    if backend == "gpu":
        if shape_type == "triangle":
            if TRIANGLE_GPU_AVAILABLE:
                return GPUCanvas(width=width, height=height)
            warnings.warn(
                "moderngl no instalado; usando CPU para triangulos",
                RuntimeWarning,
                stacklevel=2,
            )
            return Canvas(width=width, height=height)

        if ELLIPSE_GPU_AVAILABLE:
            return EllipseGPUCanvas(width=width, height=height)

        warnings.warn(
            "moderngl no instalado; usando CPU para elipses",
            RuntimeWarning,
            stacklevel=2,
        )
        return EllipseCanvas(width=width, height=height)

    if shape_type == "triangle":
        return Canvas(width=width, height=height)
    return EllipseCanvas(width=width, height=height)
