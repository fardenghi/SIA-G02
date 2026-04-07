"""
GPU-accelerated triangle renderer using moderngl (OpenGL offscreen FBO).

Interface is identical to Canvas: render(), render_to_array(), save(), from_image().
Uses a single draw call per individual instead of one Pillow overlay per triangle.

Install moderngl with:
    pip install moderngl
    # or, if using uv:
    uv sync --extra gpu
"""

from __future__ import annotations

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
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_color = in_color;
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = v_color;
}
"""


class GPUCanvas:
    """
    Offscreen triangle renderer using OpenGL via moderngl.

    The OpenGL context, shaders, and FBO are created once at construction
    and reused across all render() calls. VBO/VAO are created and released
    per render call.

    Alpha blending uses GL_SRC_ALPHA / GL_ONE_MINUS_SRC_ALPHA (the "over"
    operator), which matches Pillow's paste(overlay, mask=overlay) exactly.

    Attributes:
        width:  Canvas width in pixels.
        height: Canvas height in pixels.
    """

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Las dimensiones deben ser positivas")

        self.width = width
        self.height = height

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )

        # RGBA FBO — alpha needed so fragment shader can output it;
        # we only read back RGB at the end.
        self.fbo = self.ctx.simple_framebuffer((width, height), components=4)

    def __del__(self):
        """Release OpenGL resources on garbage collection."""
        try:
            self.fbo.release()
            self.prog.release()
            self.ctx.release()
        except Exception:
            pass

    @classmethod
    def from_image(cls, image: Image.Image) -> GPUCanvas:
        """Create a canvas with the same dimensions as an image."""
        return cls(width=image.width, height=image.height)

    def render(self, individual: Individual) -> Image.Image:
        """
        Render an Individual to a PIL RGB Image.

        All triangles are packed into a single VBO and drawn in one call.
        Z-order is preserved: triangles later in the list render on top.

        Coordinate transform:
            normalized [0,1] → NDC [-1,1]
            ndc_x =  x * 2 - 1
            ndc_y =  1 - y * 2   (flip Y: OpenGL Y-up vs. image Y-down)

        Returns:
            PIL Image in RGB mode, shape (height, width).
        """
        self.fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)  # opaque white background

        triangles = individual.triangles
        if triangles:
            n = len(triangles)
            # Per vertex: [ndc_x, ndc_y, r, g, b, a]  — 6 floats
            data = np.empty((n * 3, 6), dtype=np.float32)

            for i, tri in enumerate(triangles):
                r, g, b, a = tri.color  # a already in [0, 1]
                rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
                for j, (x, y) in enumerate(tri.vertices):
                    row = i * 3 + j
                    data[row, 0] = x * 2.0 - 1.0   # ndc_x
                    data[row, 1] = 1.0 - y * 2.0   # ndc_y (flip Y)
                    data[row, 2] = rf
                    data[row, 3] = gf
                    data[row, 4] = bf
                    data[row, 5] = a

            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self.prog, [(vbo, "2f 4f", "in_pos", "in_color")]
            )
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()

        # Read pixels back: OpenGL stores rows bottom-up; flip to image top-down.
        raw = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (self.width, self.height), raw)
        return ImageOps.flip(img)

    def render_to_array(self, individual: Individual) -> np.ndarray:
        """
        Render to a NumPy array of shape (height, width, 3), dtype uint8.
        """
        return np.array(self.render(individual), dtype=np.uint8)

    def save(self, individual: Individual, path: str):
        """Render and save an individual as an image file."""
        self.render(individual).save(path)
