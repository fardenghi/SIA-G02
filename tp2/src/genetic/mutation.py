"""
Operadores de mutación.

Alteración de coordenadas, colores, transparencia y orden Z-index
de los triángulos.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

from src.genetic.individual import Individual, Triangle


@dataclass
class MutationParams:
    """
    Parámetros de mutación.

    Attributes:
        probability: Probabilidad de mutar un individuo.
        gene_probability: Probabilidad de mutar cada triángulo.
        position_delta: Magnitud máxima de perturbación en coordenadas.
        color_delta: Magnitud máxima de perturbación en color (0-255).
        alpha_delta: Magnitud máxima de perturbación en alfa.
    """

    probability: float = 0.3
    gene_probability: float = 0.1
    position_delta: float = 0.1
    color_delta: int = 30
    alpha_delta: float = 0.1

    def __post_init__(self):
        """Valida los parámetros."""
        if not 0 <= self.probability <= 1:
            raise ValueError("probability debe estar en [0, 1]")
        if not 0 <= self.gene_probability <= 1:
            raise ValueError("gene_probability debe estar en [0, 1]")
        if not 0 <= self.position_delta <= 1:
            raise ValueError("position_delta debe estar en [0, 1]")
        if not 0 <= self.color_delta <= 255:
            raise ValueError("color_delta debe estar en [0, 255]")
        if not 0 <= self.alpha_delta <= 1:
            raise ValueError("alpha_delta debe estar en [0, 1]")


def mutate_value(value: float, delta: float, min_val: float, max_val: float) -> float:
    """
    Muta un valor con perturbación gaussiana.

    Args:
        value: Valor actual.
        delta: Desviación estándar de la perturbación.
        min_val: Valor mínimo permitido.
        max_val: Valor máximo permitido.

    Returns:
        Valor mutado dentro del rango.
    """
    perturbation = random.gauss(0, delta)
    new_value = value + perturbation
    return max(min_val, min(max_val, new_value))


def mutate_int_value(value: int, delta: int, min_val: int, max_val: int) -> int:
    """
    Muta un valor entero con perturbación.

    Args:
        value: Valor actual.
        delta: Magnitud máxima de la perturbación.
        min_val: Valor mínimo permitido.
        max_val: Valor máximo permitido.

    Returns:
        Valor mutado dentro del rango.
    """
    perturbation = random.randint(-delta, delta)
    new_value = value + perturbation
    return max(min_val, min(max_val, new_value))


def mutate_triangle(triangle: Triangle, params: MutationParams) -> Triangle:
    """
    Muta un triángulo.

    Puede mutar:
    - Posiciones de vértices
    - Componentes de color (RGB)
    - Canal alfa

    Args:
        triangle: Triángulo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo triángulo mutado.
    """
    # Mutar vértices
    new_vertices = []
    for x, y in triangle.vertices:
        new_x = mutate_value(x, params.position_delta, 0.0, 1.0)
        new_y = mutate_value(y, params.position_delta, 0.0, 1.0)
        new_vertices.append((new_x, new_y))

    # Mutar color
    r, g, b, a = triangle.color
    new_r = mutate_int_value(r, params.color_delta, 0, 255)
    new_g = mutate_int_value(g, params.color_delta, 0, 255)
    new_b = mutate_int_value(b, params.color_delta, 0, 255)
    new_a = mutate_value(a, params.alpha_delta, 0.0, 1.0)

    return Triangle(vertices=new_vertices, color=(new_r, new_g, new_b, new_a))


def mutate_individual(individual: Individual, params: MutationParams) -> Individual:
    """
    Muta un individuo.

    Cada triángulo tiene una probabilidad de ser mutado.
    También puede ocurrir un swap de posiciones (cambio de Z-index).

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    # Decidir si mutar este individuo
    if random.random() > params.probability:
        return individual.copy()

    new_triangles = []

    for triangle in individual.triangles:
        if random.random() < params.gene_probability:
            # Mutar este triángulo
            new_triangles.append(mutate_triangle(triangle, params))
        else:
            # Copiar sin cambios
            new_triangles.append(triangle.copy())

    # Con cierta probabilidad, hacer swap de dos triángulos (cambio de Z-index)
    if len(new_triangles) >= 2 and random.random() < params.gene_probability:
        i, j = random.sample(range(len(new_triangles)), 2)
        new_triangles[i], new_triangles[j] = new_triangles[j], new_triangles[i]

    return Individual(triangles=new_triangles)


class Mutator:
    """
    Clase para aplicar mutaciones a individuos.

    Encapsula los parámetros de mutación y proporciona
    una interfaz simple.
    """

    def __init__(self, params: MutationParams | None = None):
        """
        Args:
            params: Parámetros de mutación. Si es None, usa defaults.
        """
        self.params = params or MutationParams()

    def mutate(self, individual: Individual) -> Individual:
        """
        Muta un individuo.

        Args:
            individual: Individuo a mutar.

        Returns:
            Nuevo individuo (posiblemente mutado).
        """
        return mutate_individual(individual, self.params)

    def mutate_population(self, population: list[Individual]) -> list[Individual]:
        """
        Muta toda una población.

        Args:
            population: Lista de individuos.

        Returns:
            Nueva lista con individuos mutados.
        """
        return [self.mutate(ind) for ind in population]
