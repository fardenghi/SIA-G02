"""
Operadores de cruza (crossover).

Implementación de métodos de recombinación entre individuos para
generar descendencia.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.genetic.individual import Individual, Triangle


class CrossoverMethod(ABC):
    """Clase base abstracta para métodos de cruza."""

    def __init__(self, probability: float = 0.8):
        """
        Args:
            probability: Probabilidad de aplicar cruza (0-1).
        """
        if not 0 <= probability <= 1:
            raise ValueError("La probabilidad debe estar en [0, 1]")
        self.probability = probability

    @abstractmethod
    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Realiza la cruza entre dos padres.

        Args:
            parent1: Primer padre.
            parent2: Segundo padre.

        Returns:
            Tupla con dos hijos.
        """
        pass

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Aplica cruza con probabilidad.

        Si no se aplica, retorna copias de los padres.
        """
        if random.random() < self.probability:
            return self._crossover(parent1, parent2)
        else:
            return parent1.copy(), parent2.copy()


class SinglePointCrossover(CrossoverMethod):
    """
    Cruza de un punto.

    Se elige un punto de corte aleatorio y se intercambian
    las secuencias de triángulos.
    """

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Realiza cruza de un punto."""
        n = len(parent1)
        if n != len(parent2):
            raise ValueError("Los padres deben tener el mismo número de triángulos")

        if n <= 1:
            return parent1.copy(), parent2.copy()

        # Punto de corte (excluye extremos para asegurar intercambio)
        point = random.randint(1, n - 1)

        # Crear hijos intercambiando secuencias
        child1_triangles = [t.copy() for t in parent1.triangles[:point]]
        child1_triangles.extend([t.copy() for t in parent2.triangles[point:]])

        child2_triangles = [t.copy() for t in parent2.triangles[:point]]
        child2_triangles.extend([t.copy() for t in parent1.triangles[point:]])

        return Individual(child1_triangles), Individual(child2_triangles)


class TwoPointCrossover(CrossoverMethod):
    """
    Cruza de dos puntos.

    Se eligen dos puntos de corte y se intercambia la sección central.
    """

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Realiza cruza de dos puntos."""
        n = len(parent1)
        if n != len(parent2):
            raise ValueError("Los padres deben tener el mismo número de triángulos")

        if n <= 2:
            return parent1.copy(), parent2.copy()

        # Dos puntos de corte ordenados
        point1, point2 = sorted(random.sample(range(1, n), 2))

        # Crear hijos intercambiando sección central
        child1_triangles = (
            [t.copy() for t in parent1.triangles[:point1]]
            + [t.copy() for t in parent2.triangles[point1:point2]]
            + [t.copy() for t in parent1.triangles[point2:]]
        )

        child2_triangles = (
            [t.copy() for t in parent2.triangles[:point1]]
            + [t.copy() for t in parent1.triangles[point1:point2]]
            + [t.copy() for t in parent2.triangles[point2:]]
        )

        return Individual(child1_triangles), Individual(child2_triangles)


class UniformCrossover(CrossoverMethod):
    """
    Cruza uniforme.

    Cada triángulo se hereda de uno u otro padre con probabilidad 0.5.
    """

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Realiza cruza uniforme."""
        n = len(parent1)
        if n != len(parent2):
            raise ValueError("Los padres deben tener el mismo número de triángulos")

        child1_triangles = []
        child2_triangles = []

        for i in range(n):
            if random.random() < 0.5:
                child1_triangles.append(parent1[i].copy())
                child2_triangles.append(parent2[i].copy())
            else:
                child1_triangles.append(parent2[i].copy())
                child2_triangles.append(parent1[i].copy())

        return Individual(child1_triangles), Individual(child2_triangles)


class AnnularCrossover(CrossoverMethod):
    """
    Cruza anular (circular).

    Se elige un punto inicial P y una longitud L. El segmento de longitud L
    a partir de P se intercambia entre los padres, considerando el cromosoma
    como circular (puede envolver del final al inicio).
    """

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Realiza cruza anular."""
        n = len(parent1)
        if n != len(parent2):
            raise ValueError("Los padres deben tener el mismo número de triángulos")

        if n <= 1:
            return parent1.copy(), parent2.copy()

        # Punto inicial P ∈ [0, n-1]
        point = random.randint(0, n - 1)

        # Longitud L ∈ [0, ⌈n/2⌉]
        # Usamos -(-n // 2) para calcular ceil(n/2) sin importar math
        max_length = -(-n // 2)  # equivalente a math.ceil(n / 2)
        length = random.randint(0, max_length)

        # Crear copias de los padres para construir los hijos
        child1_triangles = [t.copy() for t in parent1.triangles]
        child2_triangles = [t.copy() for t in parent2.triangles]

        # Intercambiar el segmento circular de longitud L a partir de P
        for i in range(length):
            # Índice circular: (point + i) % n
            idx = (point + i) % n

            # Intercambiar triángulos en la posición idx
            child1_triangles[idx] = parent2.triangles[idx].copy()
            child2_triangles[idx] = parent1.triangles[idx].copy()

        return Individual(child1_triangles), Individual(child2_triangles)


class SpatialZIndexCrossover(CrossoverMethod):
    """
    Cruza espacial paralela con resolución de conflictos (preserva Z-Index y N).
    
    Verifica los centroides y cruza espacialmente la imagen a la mitad. 
    Resuelve los empates al azar garantizando el mantenimiento estricto 
    del largo del cromosoma.
    """
    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        n = len(parent1)
        if n != len(parent2):
            raise ValueError("Los padres deben tener el mismo número de triángulos")
            
        child1_triangles = []
        child2_triangles = []
        
        midpoint = 0.5  # Coordenadas normalizadas [0, 1]
        
        for i in range(n):
            t_A = parent1[i]
            t_B = parent2[i]
            
            # Centroides X
            cx_A = (t_A.vertices[0][0] + t_A.vertices[1][0] + t_A.vertices[2][0]) / 3.0
            cx_B = (t_B.vertices[0][0] + t_B.vertices[1][0] + t_B.vertices[2][0]) / 3.0
            
            # Regla Child1: Izquierda de A, Derecha de B
            A1_valid = cx_A < midpoint
            B1_valid = cx_B >= midpoint
            
            if A1_valid and not B1_valid:
                child1_triangles.append(t_A.copy())
            elif B1_valid and not A1_valid:
                child1_triangles.append(t_B.copy())
            else:
                child1_triangles.append(t_A.copy() if random.random() < 0.5 else t_B.copy())
                
            # Regla Child2: Izquierda de B, Derecha de A (simétrico)
            A2_valid = cx_A >= midpoint
            B2_valid = cx_B < midpoint
            
            if A2_valid and not B2_valid:
                child2_triangles.append(t_A.copy())
            elif B2_valid and not A2_valid:
                child2_triangles.append(t_B.copy())
            else:
                child2_triangles.append(t_A.copy() if random.random() < 0.5 else t_B.copy())

        return Individual(child1_triangles), Individual(child2_triangles)


class ArithmeticCrossover(CrossoverMethod):
    """
    Cruza aritmética (Blending Genético).
    
    Promedia los vértices y colores de los triángulos en cada capa Z para
    interpolarlos matemáticamente.
    """
    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        n = len(parent1)
        if n != len(parent2):
            raise ValueError("Los padres deben tener el mismo número de triángulos")
            
        child_triangles = []
        
        for i in range(n):
            t_A = parent1[i]
            t_B = parent2[i]
            
            new_vertices = []
            for j in range(3):
                vx = (t_A.vertices[j][0] + t_B.vertices[j][0]) / 2.0
                vy = (t_A.vertices[j][1] + t_B.vertices[j][1]) / 2.0
                new_vertices.append((vx, vy))
                
            r = int((t_A.color[0] + t_B.color[0]) / 2)
            g = int((t_A.color[1] + t_B.color[1]) / 2)
            b = int((t_A.color[2] + t_B.color[2]) / 2)
            a = (t_A.color[3] + t_B.color[3]) / 2.0
            
            child_triangles.append(Triangle(new_vertices, (r, g, b, a)))
            
        return Individual(child_triangles), Individual([t.copy() for t in child_triangles])




def create_crossover_method(method: str, probability: float = 0.8) -> CrossoverMethod:
    """
    Factory para crear métodos de cruza.

    Args:
        method: Nombre del método ("single_point", "two_point", "uniform", "annular").
        probability: Probabilidad de cruza.

    Returns:
        Instancia del método de cruza.
    """
    method = method.lower()

    if method == "single_point":
        return SinglePointCrossover(probability)
    elif method == "two_point":
        return TwoPointCrossover(probability)
    elif method == "uniform":
        return UniformCrossover(probability)
    elif method == "annular":
        return AnnularCrossover(probability)
    elif method == "spatial_zindex":
        return SpatialZIndexCrossover(probability)
    elif method == "arithmetic":
        return ArithmeticCrossover(probability)
    else:
        raise ValueError(f"Método de cruza desconocido: {method}")
