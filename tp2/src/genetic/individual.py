"""
Representacion del genotipo (individuo).

Cada individuo es una lista ordenada de N formas traslucidas.
El orden de la lista determina el Z-index (orden de renderizado).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, TypeAlias


Point = Tuple[float, float]
Color = Tuple[int, int, int, float]
SHAPE_TYPES = {"triangle", "ellipse"}


def _validate_normalized_point(point: Point, label: str):
    x, y = point
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(
            f"{label} fuera de rango: ({x}, {y}). Las coordenadas deben estar en [0, 1]"
        )


def _validate_color(color: Color):
    r, g, b, a = color
    if not all(0 <= c <= 255 for c in (r, g, b)):
        raise ValueError(f"Valores RGB fuera de rango [0, 255]: ({r}, {g}, {b})")
    if not 0 <= a <= 1:
        raise ValueError(f"Valor alfa fuera de rango [0, 1]: {a}")


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Triangle:
    """
    Representa un triangulo traslucido.

    Attributes:
        vertices: Lista de 3 tuplas (x, y) con coordenadas normalizadas [0, 1].
        color: Tupla RGBA donde RGB estan en [0, 255] y A en [0, 1].
    """

    vertices: List[Point]
    color: Color
    shape_type: str = field(default="triangle", init=False, repr=False)

    def __post_init__(self):
        if len(self.vertices) != 3:
            raise ValueError("Un triángulo debe tener exactamente 3 vértices")

        for i, point in enumerate(self.vertices):
            _validate_normalized_point(point, f"Vertice {i}")

        _validate_color(self.color)

    @classmethod
    def random(cls, alpha_min: float = 0.1, alpha_max: float = 0.8) -> Triangle:
        vertices = [(random.random(), random.random()) for _ in range(3)]
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.uniform(alpha_min, alpha_max),
        )
        return cls(vertices=vertices, color=color)

    def to_absolute(
        self, width: int, height: int
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
        abs_vertices = [(int(x * width), int(y * height)) for x, y in self.vertices]
        r, g, b, a = self.color
        abs_color = (r, g, b, int(a * 255))
        return abs_vertices, abs_color

    def centroid_x(self) -> float:
        return sum(x for x, _ in self.vertices) / 3.0

    def bounding_box(self) -> Tuple[float, float, float, float]:
        xs = [x for x, _ in self.vertices]
        ys = [y for _, y in self.vertices]
        return min(xs), min(ys), max(xs), max(ys)

    def copy(self) -> Triangle:
        return Triangle(vertices=[v for v in self.vertices], color=self.color)

    def to_dict(self) -> dict:
        return {
            "type": "triangle",
            "vertices": [{"x": x, "y": y} for x, y in self.vertices],
            "color": {
                "r": self.color[0],
                "g": self.color[1],
                "b": self.color[2],
                "a": self.color[3],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> Triangle:
        vertices = [(v["x"], v["y"]) for v in data["vertices"]]
        color = (
            data["color"]["r"],
            data["color"]["g"],
            data["color"]["b"],
            data["color"]["a"],
        )
        return cls(vertices=vertices, color=color)


@dataclass
class Ellipse:
    """Representa una elipse traslucida rotada."""

    center: Point
    radii: Point
    angle: float
    color: Color
    shape_type: str = field(default="ellipse", init=False, repr=False)

    def __post_init__(self):
        _validate_normalized_point(self.center, "Centro")

        rx, ry = self.radii
        if not (0 < rx <= 1 and 0 < ry <= 1):
            raise ValueError(
                f"Radios fuera de rango: ({rx}, {ry}). Deben estar en (0, 1]"
            )

        if not math.isfinite(self.angle):
            raise ValueError(f"Angulo invalido: {self.angle}")

        self.angle = _normalize_angle(self.angle)
        _validate_color(self.color)

    @classmethod
    def random(cls, alpha_min: float = 0.1, alpha_max: float = 0.8) -> Ellipse:
        center = (random.random(), random.random())
        radii = (
            random.uniform(0.02, 0.35),
            random.uniform(0.02, 0.35),
        )
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.uniform(alpha_min, alpha_max),
        )
        return cls(
            center=center,
            radii=radii,
            angle=random.uniform(-math.pi, math.pi),
            color=color,
        )

    def to_absolute(
        self, width: int, height: int
    ) -> Tuple[
        Tuple[float, float], Tuple[float, float], float, Tuple[int, int, int, int]
    ]:
        cx, cy = self.center
        rx, ry = self.radii
        r, g, b, a = self.color
        return (
            (cx * width, cy * height),
            (max(1.0, rx * width), max(1.0, ry * height)),
            self.angle,
            (r, g, b, int(a * 255)),
        )

    def centroid_x(self) -> float:
        return self.center[0]

    def bounding_box(self) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        rx, ry = self.radii
        cos_theta = math.cos(self.angle)
        sin_theta = math.sin(self.angle)
        half_w = math.sqrt((rx * cos_theta) ** 2 + (ry * sin_theta) ** 2)
        half_h = math.sqrt((rx * sin_theta) ** 2 + (ry * cos_theta) ** 2)
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    def copy(self) -> Ellipse:
        return Ellipse(
            center=self.center,
            radii=self.radii,
            angle=self.angle,
            color=self.color,
        )

    def to_dict(self) -> dict:
        return {
            "type": "ellipse",
            "center": {"x": self.center[0], "y": self.center[1]},
            "radii": {"x": self.radii[0], "y": self.radii[1]},
            "angle": self.angle,
            "color": {
                "r": self.color[0],
                "g": self.color[1],
                "b": self.color[2],
                "a": self.color[3],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> Ellipse:
        center_data = data["center"]
        radii_data = data["radii"]
        color = (
            data["color"]["r"],
            data["color"]["g"],
            data["color"]["b"],
            data["color"]["a"],
        )
        return cls(
            center=(center_data["x"], center_data["y"]),
            radii=(radii_data["x"], radii_data["y"]),
            angle=data.get("angle", 0.0),
            color=color,
        )


ShapeGene: TypeAlias = Triangle | Ellipse


def get_shape_type(gene: ShapeGene) -> str:
    if isinstance(gene, Triangle):
        return "triangle"
    if isinstance(gene, Ellipse):
        return "ellipse"
    raise TypeError(f"Gen no soportado: {type(gene)!r}")


def deserialize_gene(data: dict, default_shape_type: str | None = None) -> ShapeGene:
    shape_type = data.get("type", default_shape_type or "triangle")
    if shape_type == "triangle":
        return Triangle.from_dict(data)
    if shape_type == "ellipse":
        return Ellipse.from_dict(data)
    raise ValueError(f"Tipo de forma desconocido: {shape_type}")


@dataclass
class Individual:
    """
    Representa un individuo (cromosoma) del algoritmo genetico.

    Mantiene la interfaz legacy basada en `triangles`, pero el contenido puede
    ser una lista homogenea de triangulos o elipses segun `shape_type`.
    """

    triangles: List[ShapeGene]
    fitness: float | None = field(default=None, compare=False)
    shape_type: str = field(default="triangle", compare=False)

    def __post_init__(self):
        if self.shape_type not in SHAPE_TYPES:
            raise ValueError(f"shape_type debe ser uno de {sorted(SHAPE_TYPES)}")

        if not self.triangles:
            return

        inferred_shape_type = get_shape_type(self.triangles[0])
        for gene in self.triangles[1:]:
            if get_shape_type(gene) != inferred_shape_type:
                raise ValueError(
                    "Un individuo debe contener una sola familia de formas"
                )
        self.shape_type = inferred_shape_type

    @property
    def genes(self) -> List[ShapeGene]:
        return self.triangles

    def __len__(self) -> int:
        return len(self.triangles)

    def __getitem__(self, index: int) -> ShapeGene:
        return self.triangles[index]

    def __setitem__(self, index: int, gene: ShapeGene):
        gene_shape_type = get_shape_type(gene)
        if self.triangles and gene_shape_type != self.shape_type:
            raise ValueError(
                f"No se puede insertar un gen {gene_shape_type} en un individuo {self.shape_type}"
            )
        self.triangles[index] = gene
        self.shape_type = gene_shape_type
        self.fitness = None

    @classmethod
    def random(
        cls,
        num_triangles: int,
        alpha_min: float = 0.1,
        alpha_max: float = 0.8,
        shape_type: str = "triangle",
    ) -> Individual:
        if shape_type not in SHAPE_TYPES:
            raise ValueError(f"shape_type debe ser uno de {sorted(SHAPE_TYPES)}")

        gene_cls = Triangle if shape_type == "triangle" else Ellipse
        genes = [
            gene_cls.random(alpha_min=alpha_min, alpha_max=alpha_max)
            for _ in range(num_triangles)
        ]
        return cls(triangles=genes, shape_type=shape_type)

    def copy(self) -> Individual:
        return Individual(
            triangles=[gene.copy() for gene in self.triangles],
            fitness=self.fitness,
            shape_type=self.shape_type,
        )

    def invalidate_fitness(self):
        self.fitness = None

    def to_dict(self) -> dict:
        shapes = [gene.to_dict() for gene in self.triangles]
        data = {
            "shape_type": self.shape_type,
            "num_shapes": len(self.triangles),
            "fitness": self.fitness,
            "shapes": shapes,
        }

        if self.shape_type == "triangle":
            data["num_triangles"] = len(self.triangles)
            data["triangles"] = shapes
        elif self.shape_type == "ellipse":
            data["num_ellipses"] = len(self.triangles)
            data["ellipses"] = shapes

        return data

    @classmethod
    def from_dict(cls, data: dict) -> Individual:
        shape_type = data.get("shape_type")

        if "shapes" in data:
            shapes_data = data["shapes"]
        elif "triangles" in data:
            shapes_data = data["triangles"]
            shape_type = "triangle"
        elif "ellipses" in data:
            shapes_data = data["ellipses"]
            shape_type = "ellipse"
        else:
            shapes_data = []

        if shape_type is None and shapes_data:
            shape_type = shapes_data[0].get("type", "triangle")
        shape_type = shape_type or "triangle"

        genes = [
            deserialize_gene(shape, default_shape_type=shape_type)
            for shape in shapes_data
        ]
        individual = cls(triangles=genes, shape_type=shape_type)
        individual.fitness = data.get("fitness")
        return individual
