"""
Operadores de mutación.

Implementa los 4 tipos de mutación según la teoría:
- SINGLE_GENE: Muta exactamente 1 gen (triángulo)
- LIMITED_MULTIGEN: Muta entre 1 y M genes
- UNIFORM_MULTIGEN: Cada gen tiene probabilidad independiente de mutar
- COMPLETE: Muta todos los genes del individuo

Alteración de coordenadas, colores, transparencia y orden Z-index
de los triángulos.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from src.genetic.individual import Individual, Triangle


class MutationType(Enum):
    """
    Tipos de mutación disponibles según la teoría.

    - SINGLE_GENE: Muta exactamente 1 gen (triángulo) con probabilidad Pm.
    - LIMITED_MULTIGEN: Muta entre [1, M] genes con probabilidad Pm.
    - UNIFORM_MULTIGEN: Cada gen tiene probabilidad Pm de ser mutado (independiente).
    - COMPLETE: Muta todos los genes con probabilidad Pm.
    """

    SINGLE_GENE = "single_gene"
    LIMITED_MULTIGEN = "limited_multigen"
    UNIFORM_MULTIGEN = "uniform_multigen"
    COMPLETE = "complete"


@dataclass
class MutationParams:
    """
    Parámetros de mutación.

    Attributes:
        mutation_type: Tipo de mutación a aplicar.
        probability: Probabilidad de mutar un individuo (Pm).
        gene_probability: Probabilidad de mutar cada gen (solo para UNIFORM_MULTIGEN).
        max_genes: Máximo de genes a mutar (M) para LIMITED_MULTIGEN.
        position_delta: Magnitud máxima de perturbación en coordenadas.
        color_delta: Magnitud máxima de perturbación en color (0-255).
        alpha_delta: Magnitud máxima de perturbación en alfa.
    """

    mutation_type: MutationType = MutationType.UNIFORM_MULTIGEN
    probability: float = 0.3
    gene_probability: float = 0.1
    max_genes: int = 3
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
        if self.max_genes < 1:
            raise ValueError("max_genes debe ser >= 1")


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


def _apply_zindex_swap(triangles: List[Triangle], probability: float) -> None:
    """
    Aplica swap de Z-index (intercambio de posiciones) in-place.

    Args:
        triangles: Lista de triángulos (se modifica in-place).
        probability: Probabilidad de realizar el swap.
    """
    if len(triangles) >= 2 and random.random() < probability:
        i, j = random.sample(range(len(triangles)), 2)
        triangles[i], triangles[j] = triangles[j], triangles[i]


def mutate_single_gene(individual: Individual, params: MutationParams) -> Individual:
    """
    Mutación de gen único.

    Con probabilidad Pm, muta exactamente UN gen (triángulo) elegido al azar.

    Cuándo sirve:
    - Cuando se quiere hacer cambios chicos
    - Cuando conviene no romper demasiado la estructura del individuo
    - Cuando el problema es sensible a cambios grandes

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    if random.random() > params.probability:
        return individual.copy()

    new_triangles = [t.copy() for t in individual.triangles]

    if len(new_triangles) > 0:
        # Elegir exactamente 1 gen al azar
        idx = random.randrange(len(new_triangles))
        new_triangles[idx] = mutate_triangle(new_triangles[idx], params)

    # Swap de Z-index con probabilidad reducida
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


def mutate_limited_multigen(
    individual: Individual, params: MutationParams
) -> Individual:
    """
    Mutación multigen limitada.

    Con probabilidad Pm, muta entre [1, M] genes elegidos al azar.
    M está definido por params.max_genes.

    Cuándo sirve:
    - Cuando una mutación de un solo gen resulta demasiado conservadora
    - Cuando se quiere explorar más sin llegar a modificar todo el individuo
    - Cuando tiene sentido controlar cuánto "ruido" se introduce

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    if random.random() > params.probability:
        return individual.copy()

    new_triangles = [t.copy() for t in individual.triangles]

    if len(new_triangles) > 0:
        # Elegir entre 1 y M genes (limitado por el tamaño del individuo)
        max_to_mutate = min(params.max_genes, len(new_triangles))
        num_to_mutate = random.randint(1, max_to_mutate)

        # Seleccionar índices únicos al azar
        indices_to_mutate = random.sample(range(len(new_triangles)), num_to_mutate)

        for idx in indices_to_mutate:
            new_triangles[idx] = mutate_triangle(new_triangles[idx], params)

    # Swap de Z-index
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


def mutate_uniform_multigen(
    individual: Individual, params: MutationParams
) -> Individual:
    """
    Mutación multigen uniforme.

    Cada gen tiene probabilidad independiente (gene_probability) de ser mutado.
    No hay decisión global de "si ocurre mutación o no", cada gen decide
    independientemente.

    Cuándo sirve:
    - Cuando no se quiere fijar de antemano cuántos genes cambiar
    - Cuando se busca una mutación distribuida a lo largo de todo el cromosoma
    - Cuando el cromosoma tiene muchos genes relativamente independientes

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    # En uniform_multigen, cada gen decide independientemente
    # Usamos gene_probability como la probabilidad de cada gen
    new_triangles = []

    for triangle in individual.triangles:
        if random.random() < params.gene_probability:
            new_triangles.append(mutate_triangle(triangle, params))
        else:
            new_triangles.append(triangle.copy())

    # Swap de Z-index
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


def mutate_complete(individual: Individual, params: MutationParams) -> Individual:
    """
    Mutación completa.

    Con probabilidad Pm, muta TODOS los genes del individuo.

    Cuándo sirve:
    - Para introducir un cambio muy fuerte
    - Para escapar de poblaciones demasiado estancadas
    - Como operador más agresivo de exploración

    Riesgo: puede romper estructuras buenas que ya se habían encontrado.

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    if random.random() > params.probability:
        return individual.copy()

    # Mutar TODOS los triángulos
    new_triangles = [mutate_triangle(t, params) for t in individual.triangles]

    # Swap de Z-index
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


# Dispatcher de métodos de mutación
_MUTATION_FUNCTIONS = {
    MutationType.SINGLE_GENE: mutate_single_gene,
    MutationType.LIMITED_MULTIGEN: mutate_limited_multigen,
    MutationType.UNIFORM_MULTIGEN: mutate_uniform_multigen,
    MutationType.COMPLETE: mutate_complete,
}


def mutate_individual(individual: Individual, params: MutationParams) -> Individual:
    """
    Muta un individuo usando el método especificado en params.

    Dispatcher que selecciona la función de mutación apropiada según
    params.mutation_type.

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación (incluye el tipo).

    Returns:
        Nuevo individuo mutado.
    """
    mutation_func = _MUTATION_FUNCTIONS.get(params.mutation_type)
    if mutation_func is None:
        raise ValueError(f"Tipo de mutación no soportado: {params.mutation_type}")

    return mutation_func(individual, params)


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

    @property
    def mutation_type(self) -> MutationType:
        """Retorna el tipo de mutación configurado."""
        return self.params.mutation_type

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


def create_mutation_params(
    mutation_method: str = "uniform_multigen",
    probability: float = 0.3,
    gene_probability: float = 0.1,
    max_genes: int = 3,
    position_delta: float = 0.1,
    color_delta: int = 30,
    alpha_delta: float = 0.1,
) -> MutationParams:
    """
    Factory para crear parámetros de mutación desde strings.

    Args:
        mutation_method: Nombre del método de mutación.
            Opciones: "single_gene", "limited_multigen", "uniform_multigen", "complete"
        probability: Probabilidad de mutar un individuo (Pm).
        gene_probability: Probabilidad por gen (para uniform_multigen).
        max_genes: Máximo de genes a mutar (M) para limited_multigen.
        position_delta: Delta de posición.
        color_delta: Delta de color.
        alpha_delta: Delta de alfa.

    Returns:
        MutationParams configurado.

    Raises:
        ValueError: Si el método no es válido.
    """
    method_map = {
        "single_gene": MutationType.SINGLE_GENE,
        "limited_multigen": MutationType.LIMITED_MULTIGEN,
        "uniform_multigen": MutationType.UNIFORM_MULTIGEN,
        "complete": MutationType.COMPLETE,
    }

    if mutation_method not in method_map:
        valid_methods = ", ".join(method_map.keys())
        raise ValueError(
            f"Método de mutación '{mutation_method}' no válido. "
            f"Opciones: {valid_methods}"
        )

    return MutationParams(
        mutation_type=method_map[mutation_method],
        probability=probability,
        gene_probability=gene_probability,
        max_genes=max_genes,
        position_delta=position_delta,
        color_delta=color_delta,
        alpha_delta=alpha_delta,
    )
