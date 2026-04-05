"""
Representación del genotipo (individuo).

Cada individuo es una lista ordenada de N triángulos con sus propiedades.
El orden de los triángulos determina el Z-index (orden de renderizado).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple
import copy


@dataclass
class Triangle:
    """
    Representa un triángulo traslúcido.

    Attributes:
        vertices: Lista de 3 tuplas (x, y) con coordenadas normalizadas [0, 1].
        color: Tupla RGBA donde RGB están en [0, 255] y A en [0, 1].
    """

    vertices: List[Tuple[float, float]]
    color: Tuple[int, int, int, float]  # (R, G, B, Alpha)

    def __post_init__(self):
        """Valida los datos del triángulo."""
        if len(self.vertices) != 3:
            raise ValueError("Un triángulo debe tener exactamente 3 vértices")

        for i, (x, y) in enumerate(self.vertices):
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(
                    f"Vértice {i} fuera de rango: ({x}, {y}). "
                    "Las coordenadas deben estar en [0, 1]"
                )

        r, g, b, a = self.color
        if not all(0 <= c <= 255 for c in (r, g, b)):
            raise ValueError(f"Valores RGB fuera de rango [0, 255]: ({r}, {g}, {b})")
        if not 0 <= a <= 1:
            raise ValueError(f"Valor alfa fuera de rango [0, 1]: {a}")

    @classmethod
    def random(cls, alpha_min: float = 0.1, alpha_max: float = 0.8) -> Triangle:
        """
        Genera un triángulo aleatorio.

        Args:
            alpha_min: Valor mínimo de transparencia.
            alpha_max: Valor máximo de transparencia.

        Returns:
            Un nuevo triángulo con valores aleatorios.
        """
        vertices = [(random.random(), random.random()) for _ in range(3)]
        color = (
            random.randint(0, 255),  # R
            random.randint(0, 255),  # G
            random.randint(0, 255),  # B
            random.uniform(alpha_min, alpha_max),  # Alpha
        )
        return cls(vertices=vertices, color=color)

    def to_absolute(
        self, width: int, height: int
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
        """
        Convierte coordenadas normalizadas a absolutas.

        Args:
            width: Ancho de la imagen en píxeles.
            height: Alto de la imagen en píxeles.

        Returns:
            Tupla (vértices_absolutos, color_rgba_255) donde:
            - vértices_absolutos: Lista de 3 tuplas (x, y) en píxeles.
            - color_rgba_255: Tupla (R, G, B, A) con A en [0, 255].
        """
        abs_vertices = [(int(x * width), int(y * height)) for x, y in self.vertices]
        r, g, b, a = self.color
        abs_color = (r, g, b, int(a * 255))
        return abs_vertices, abs_color

    def copy(self) -> Triangle:
        """Retorna una copia profunda del triángulo."""
        return Triangle(vertices=[v for v in self.vertices], color=self.color)

    def to_dict(self) -> dict:
        """Convierte el triángulo a diccionario para serialización."""
        return {
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
        """Crea un triángulo desde un diccionario."""
        vertices = [(v["x"], v["y"]) for v in data["vertices"]]
        color = (
            data["color"]["r"],
            data["color"]["g"],
            data["color"]["b"],
            data["color"]["a"],
        )
        return cls(vertices=vertices, color=color)


@dataclass
class Individual:
    """
    Representa un individuo (cromosoma) en el algoritmo genético.

    Un individuo es una lista ordenada de triángulos. El orden determina
    el Z-index (los triángulos al final de la lista se renderizan encima).

    Attributes:
        triangles: Lista ordenada de triángulos.
        fitness: Valor de aptitud (mayor es mejor, None si no evaluado).
    """

    triangles: List[Triangle]
    fitness: float | None = field(default=None, compare=False)

    def __len__(self) -> int:
        """Retorna la cantidad de triángulos."""
        return len(self.triangles)

    def __getitem__(self, index: int) -> Triangle:
        """Accede a un triángulo por índice."""
        return self.triangles[index]

    def __setitem__(self, index: int, triangle: Triangle):
        """Modifica un triángulo por índice."""
        self.triangles[index] = triangle
        self.fitness = None  # Invalidar fitness al modificar

    @classmethod
    def random(
        cls, num_triangles: int, alpha_min: float = 0.1, alpha_max: float = 0.8
    ) -> Individual:
        """
        Genera un individuo aleatorio.

        Args:
            num_triangles: Cantidad de triángulos.
            alpha_min: Valor mínimo de transparencia.
            alpha_max: Valor máximo de transparencia.

        Returns:
            Un nuevo individuo con triángulos aleatorios.
        """
        triangles = [
            Triangle.random(alpha_min=alpha_min, alpha_max=alpha_max)
            for _ in range(num_triangles)
        ]
        return cls(triangles=triangles)

    def copy(self) -> Individual:
        """Retorna una copia profunda del individuo."""
        return Individual(
            triangles=[t.copy() for t in self.triangles], fitness=self.fitness
        )

    def invalidate_fitness(self):
        """Invalida el fitness (debe recalcularse)."""
        self.fitness = None

    def to_dict(self) -> dict:
        """Convierte el individuo a diccionario para serialización."""
        return {
            "num_triangles": len(self.triangles),
            "fitness": self.fitness,
            "triangles": [t.to_dict() for t in self.triangles],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Individual:
        """Crea un individuo desde un diccionario."""
        triangles = [Triangle.from_dict(t) for t in data["triangles"]]
        individual = cls(triangles=triangles)
        individual.fitness = data.get("fitness")
        return individual
