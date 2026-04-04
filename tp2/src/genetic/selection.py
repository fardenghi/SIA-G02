"""
Métodos de selección.

Implementación de operadores de selección para elegir individuos
que participarán en la reproducción.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

from src.genetic.individual import Individual


class SelectionMethod(ABC):
    """Clase base abstracta para métodos de selección."""

    @abstractmethod
    def select(
        self, population: List[Individual], num_parents: int
    ) -> List[Individual]:
        """
        Selecciona individuos de la población.

        Args:
            population: Lista de individuos (deben tener fitness calculado).
            num_parents: Cantidad de individuos a seleccionar.

        Returns:
            Lista de individuos seleccionados.
        """
        pass


class TournamentSelection(SelectionMethod):
    """
    Selección por torneo.

    Se eligen k individuos al azar y el mejor (menor fitness) gana.
    Se repite hasta obtener la cantidad deseada de padres.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Args:
            tournament_size: Cantidad de participantes por torneo.
        """
        if tournament_size < 2:
            raise ValueError("El tamaño del torneo debe ser al menos 2")
        self.tournament_size = tournament_size

    def select(
        self, population: List[Individual], num_parents: int
    ) -> List[Individual]:
        """Selecciona mediante torneos."""
        if len(population) < self.tournament_size:
            raise ValueError(
                f"Población ({len(population)}) menor que tamaño de torneo "
                f"({self.tournament_size})"
            )

        selected = []
        for _ in range(num_parents):
            # Elegir participantes al azar
            participants = random.sample(population, self.tournament_size)
            # El ganador es el de menor fitness
            winner = min(participants, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected


class RouletteSelection(SelectionMethod):
    """
    Selección por ruleta (proporcional al fitness).

    La probabilidad de selección es inversamente proporcional al fitness
    (menor fitness = mayor probabilidad).
    """

    def select(
        self, population: List[Individual], num_parents: int
    ) -> List[Individual]:
        """Selecciona mediante ruleta invertida."""
        if not population:
            raise ValueError("La población está vacía")

        # Calcular fitness invertido (mayor = mejor para ruleta)
        fitness_values = [ind.fitness for ind in population]
        max_fitness = max(fitness_values)

        # Invertir: fitness_invertido = max_fitness - fitness + epsilon
        # Agregar epsilon para evitar probabilidad 0
        epsilon = 1e-10
        inverted_fitness = [max_fitness - f + epsilon for f in fitness_values]

        total = sum(inverted_fitness)
        probabilities = [f / total for f in inverted_fitness]

        # Seleccionar con reemplazo según probabilidades
        selected = random.choices(population, weights=probabilities, k=num_parents)

        return selected


class RankSelection(SelectionMethod):
    """
    Selección por ranking.

    Los individuos se ordenan por fitness y la probabilidad de selección
    es proporcional a su posición en el ranking (mejor posición = mayor prob).
    """

    def select(
        self, population: List[Individual], num_parents: int
    ) -> List[Individual]:
        """Selecciona según ranking."""
        if not population:
            raise ValueError("La población está vacía")

        # Ordenar por fitness (menor primero = mejor)
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)

        # Asignar pesos según posición: mejor = mayor peso
        n = len(sorted_pop)
        weights = [n - i for i in range(n)]  # [n, n-1, ..., 1]

        selected = random.choices(sorted_pop, weights=weights, k=num_parents)

        return selected


class ElitistSelection(SelectionMethod):
    """
    Selección elitista.

    Garantiza que los mejores individuos pasen a la siguiente generación
    y usa otro método para el resto.
    """

    def __init__(self, elite_count: int, base_method: SelectionMethod):
        """
        Args:
            elite_count: Cantidad de élite que pasa directamente.
            base_method: Método de selección para el resto.
        """
        if elite_count < 0:
            raise ValueError("elite_count debe ser no negativo")
        self.elite_count = elite_count
        self.base_method = base_method

    def select(
        self, population: List[Individual], num_parents: int
    ) -> List[Individual]:
        """Selecciona élite + resto con método base."""
        if not population:
            raise ValueError("La población está vacía")

        # Ordenar por fitness
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)

        # Élite: los mejores pasan directamente
        actual_elite = min(self.elite_count, len(sorted_pop), num_parents)
        elite = sorted_pop[:actual_elite]

        # Resto: seleccionar con método base
        remaining = num_parents - actual_elite
        if remaining > 0:
            others = self.base_method.select(population, remaining)
        else:
            others = []

        return elite + others


def create_selection_method(
    method: str,
    tournament_size: int = 3,
    elite_ratio: float = 0.0,
    population_size: int = 100,
) -> SelectionMethod:
    """
    Factory para crear métodos de selección.

    Args:
        method: Nombre del método ("tournament", "roulette", "rank").
        tournament_size: Tamaño del torneo (solo para tournament).
        elite_ratio: Proporción de élite (0-1).
        population_size: Tamaño de la población (para calcular élite).

    Returns:
        Instancia del método de selección.
    """
    method = method.lower()

    if method == "tournament":
        base = TournamentSelection(tournament_size)
    elif method == "roulette":
        base = RouletteSelection()
    elif method == "rank":
        base = RankSelection()
    else:
        raise ValueError(f"Método de selección desconocido: {method}")

    # Agregar elitismo si se especifica
    if elite_ratio > 0:
        elite_count = max(1, int(population_size * elite_ratio))
        return ElitistSelection(elite_count, base)

    return base
