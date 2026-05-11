"""GPU renderer dedicado para elipses usando moderngl."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

try:
    import moderngl

    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False

if TYPE_CHECKING:
    from src.genetic.individual import Individual


_VERTEX_SHADER = """
#version 330 core
in vec2 in_pos;
in vec2 in_world;
in vec2 in_center;
in vec2 in_radii;
in float in_angle;
in vec4 in_color;
out vec2 v_world;
out vec2 v_center;
out vec2 v_radii;
out float v_angle;
out vec4 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_world = in_world;
    v_center = in_center;
    v_radii = in_radii;
    v_angle = in_angle;
    v_color = in_color;
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_world;
in vec2 v_center;
in vec2 v_radii;
in float v_angle;
in vec4 v_color;
out vec4 f_color;
void main() {
    vec2 d = v_world - v_center;
    float c = cos(v_angle);
    float s = sin(v_angle);
    vec2 local = vec2(c * d.x + s * d.y, -s * d.x + c * d.y);
    float inside = (local.x * local.x) / max(v_radii.x * v_radii.x, 1e-8)
                 + (local.y * local.y) / max(v_radii.y * v_radii.y, 1e-8);
    if (inside > 1.0) {
        discard;
    }
    f_color = v_color;
}
"""


class EllipseGPUCanvas:
    """Renderer offscreen para individuos compuestos por elipses."""

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Las dimensiones deben ser positivas")
        if not MODERNGL_AVAILABLE:
            raise RuntimeError("moderngl no instalado")

        self.width = width
        self.height = height

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.fbo = self.ctx.simple_framebuffer((width, height), components=4)

    def __del__(self):
        try:
            self.fbo.release()
            self.prog.release()
            self.ctx.release()
        except Exception:
            pass

    @classmethod
    def from_image(cls, image: Image.Image) -> EllipseGPUCanvas:
        return cls(width=image.width, height=image.height)

    def render(self, individual: Individual) -> Image.Image:
        self.fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        if individual.triangles:
            rows = []

            for ellipse in individual.triangles:
                (cx, cy), (rx, ry), angle, abs_color = ellipse.to_absolute(
                    self.width, self.height
                )
                r, g, b, a_255 = abs_color
                cos_theta = math.cos(angle)
                sin_theta = math.sin(angle)
                half_w = math.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
                half_h = math.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)

                left = max(0.0, cx - half_w)
                right = min(float(self.width), cx + half_w)
                top = max(0.0, cy - half_h)
                bottom = min(float(self.height), cy + half_h)
                if left >= right or top >= bottom:
                    continue

                left_n = left / self.width
                right_n = right / self.width
                top_n = top / self.height
                bottom_n = bottom / self.height
                center_nx = cx / self.width
                center_ny = cy / self.height
                radii_nx = rx / self.width
                radii_ny = ry / self.height
                color = (r / 255.0, g / 255.0, b / 255.0, a_255 / 255.0)

                quad = [
                    (left_n, top_n),
                    (right_n, top_n),
                    (right_n, bottom_n),
                    (left_n, top_n),
                    (right_n, bottom_n),
                    (left_n, bottom_n),
                ]

                for wx, wy in quad:
                    rows.append(
                        [
                            wx * 2.0 - 1.0,
                            1.0 - wy * 2.0,
                            wx,
                            wy,
                            center_nx,
                            center_ny,
                            radii_nx,
                            radii_ny,
                            angle,
                            color[0],
                            color[1],
                            color[2],
                            color[3],
                        ]
                    )

            if rows:
                data = np.array(rows, dtype=np.float32)
                vbo = self.ctx.buffer(data.tobytes())
                vao = self.ctx.vertex_array(
                    self.prog,
                    [
                        (
                            vbo,
                            "2f 2f 2f 2f 1f 4f",
                            "in_pos",
                            "in_world",
                            "in_center",
                            "in_radii",
                            "in_angle",
                            "in_color",
                        )
                    ],
                )
                vao.render(moderngl.TRIANGLES)
                vao.release()
                vbo.release()

        raw = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (self.width, self.height), raw)
        return ImageOps.flip(img)

    def render_to_array(self, individual: Individual) -> np.ndarray:
        return np.array(self.render(individual), dtype=np.uint8)

    def save(self, individual: Individual, path: str):
        self.render(individual).save(path)
