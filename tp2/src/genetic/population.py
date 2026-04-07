"""
Gestión de la población.

Manejo de colecciones de individuos y operaciones a nivel población.
"""

from __future__ import annotations

from typing import List, Callable
import random

import numpy as np

from src.genetic.individual import Individual


class Population:
    """
    Representa una población de individuos.

    Attributes:
        individuals: Lista de individuos.
        generation: Número de generación actual.
    """

    def __init__(self, individuals: List[Individual], generation: int = 0):
        """
        Args:
            individuals: Lista de individuos.
            generation: Número de generación.
        """
        self.individuals = individuals
        self.generation = generation

    def __len__(self) -> int:
        """Retorna el tamaño de la población."""
        return len(self.individuals)

    def __getitem__(self, index: int) -> Individual:
        """Accede a un individuo por índice."""
        return self.individuals[index]

    def __iter__(self):
        """Itera sobre los individuos."""
        return iter(self.individuals)

    @classmethod
    def random(
        cls,
        size: int,
        num_triangles: int,
        alpha_min: float = 0.1,
        alpha_max: float = 0.8,
    ) -> Population:
        """
        Genera una población aleatoria.

        Args:
            size: Número de individuos.
            num_triangles: Triángulos por individuo.
            alpha_min: Alfa mínimo para triángulos.
            alpha_max: Alfa máximo para triángulos.

        Returns:
            Nueva población aleatoria.
        """
        individuals = [
            Individual.random(
                num_triangles=num_triangles, alpha_min=alpha_min, alpha_max=alpha_max
            )
            for _ in range(size)
        ]
        return cls(individuals=individuals, generation=0)

    @property
    def best(self) -> Individual | None:
        """Retorna el mejor individuo (mayor fitness)."""
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda ind: ind.fitness)

    @property
    def worst(self) -> Individual | None:
        """Retorna el peor individuo (menor fitness)."""
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        if not evaluated:
            return None
        return min(evaluated, key=lambda ind: ind.fitness)

    @property
    def average_fitness(self) -> float | None:
        """Retorna el fitness promedio de la población."""
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        if not evaluated:
            return None
        return sum(ind.fitness for ind in evaluated) / len(evaluated)

    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas de la población.

        Usa numpy para calcular max/min/mean en un solo pass sobre el array
        de fitness, evitando tres iteraciones separadas sobre la población.

        Returns:
            Diccionario con best_fitness, worst_fitness, avg_fitness, generation.
        """
        fitnesses = np.fromiter(
            (ind.fitness for ind in self.individuals if ind.fitness is not None),
            dtype=np.float32,
        )
        if fitnesses.size == 0:
            return {
                "generation": self.generation,
                "best_fitness": None,
                "worst_fitness": None,
                "avg_fitness": None,
                "size": len(self),
            }
        return {
            "generation": self.generation,
            "best_fitness": float(fitnesses.max()),
            "worst_fitness": float(fitnesses.min()),
            "avg_fitness": float(fitnesses.mean()),
            "size": len(self),
        }

    def sorted_by_fitness(self) -> List[Individual]:
        """
        Retorna individuos ordenados por fitness (mejor primero).

        Los individuos sin fitness van al final.
        """
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        not_evaluated = [ind for ind in self.individuals if ind.fitness is None]

        evaluated.sort(key=lambda ind: ind.fitness, reverse=True)
        return evaluated + not_evaluated
