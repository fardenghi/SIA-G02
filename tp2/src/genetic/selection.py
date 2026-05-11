"""
Métodos de selección.

Implementación de operadores de selección para elegir individuos
que participarán en la reproducción.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.genetic.individual import Individual


class SelectionMethod(ABC):
    """Clase base abstracta para métodos de selección."""

    @abstractmethod
    def select(
        self, population: List[Individual], num_parents: int, generation: int = 0
    ) -> List[Individual]:
        """
        Selecciona individuos de la población.

        Args:
            population: Lista de individuos (deben tener fitness calculado).
            num_parents: Cantidad de individuos a seleccionar.
            generation: Número de generación actual (usado por métodos adaptativos).

        Returns:
            Lista de individuos seleccionados.
        """
        pass


class TournamentSelection(SelectionMethod):
    """
    Selección por torneo determinístico.

    Se eligen k individuos al azar y el mejor (mayor fitness) gana.
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
        self, population: List[Individual], num_parents: int, generation: int = 0
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
            # El ganador es el de mayor fitness
            winner = max(participants, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected


class ProbabilisticTournamentSelection(SelectionMethod):
    """
    Selección por torneo probabilístico.

    Se eligen 2 individuos al azar. Con probabilidad `threshold` gana
    el más apto; con probabilidad `1 - threshold` gana el menos apto.
    """

    def __init__(self, threshold: float = 0.75):
        """
        Args:
            threshold: Probabilidad de elegir al más apto. Debe estar en [0.5, 1].
        """
        if not (0.5 <= threshold <= 1.0):
            raise ValueError("threshold debe estar en [0.5, 1]")
        self.threshold = threshold

    def select(
        self, population: List[Individual], num_parents: int, generation: int = 0
    ) -> List[Individual]:
        """Selecciona mediante torneo probabilístico."""
        if len(population) < 2:
            raise ValueError("La población debe tener al menos 2 individuos")

        selected = []
        for _ in range(num_parents):
            a, b = random.sample(population, 2)
            best, worst = (a, b) if a.fitness >= b.fitness else (b, a)
            selected.append(best if random.random() < self.threshold else worst)

        return selected


class RouletteSelection(SelectionMethod):
    """
    Selección por ruleta (proporcional al fitness).

    La probabilidad de selección es proporcional al fitness
    (mayor fitness = mayor probabilidad).
    """

    def select(
        self, population: List[Individual], num_parents: int, generation: int = 0
    ) -> List[Individual]:
        """Selecciona mediante ruleta proporcional al fitness."""
        if not population:
            raise ValueError("La población está vacía")
        if num_parents <= 0:
            return []

        weights = np.array([ind.fitness for ind in population], dtype=np.float64)

        # Desplazar si hay fitness <= 0
        min_w = weights.min()
        if min_w <= 0:
            weights = weights - min_w + 1e-10

        total = weights.sum()
        if total <= 0:
            raise ValueError("No se pudieron construir pesos válidos para ruleta")

        # CDF normalizada; forzar último a 1.0 para evitar errores de punto flotante
        cumulative = np.cumsum(weights / total)
        cumulative[-1] = 1.0

        # K búsquedas en la CDF en una sola llamada vectorizada
        randoms = np.random.uniform(0.0, 1.0, num_parents)
        indices = np.searchsorted(cumulative, randoms)
        np.clip(indices, 0, len(population) - 1, out=indices)

        return [population[i] for i in indices]


class UniversalSelection(SelectionMethod):
    """
    Selección universal estocástica (SUS).

    Igual que ruleta, pero usa K punteros equiespaciados a partir de un
    único punto de inicio aleatorio, reduciendo la varianza del muestreo.

    Fórmula: r ~ Uniform[0,1), r_j = (r + j) / K  para j in [0, K-1]
    """

    def select(
        self, population: List[Individual], num_parents: int, generation: int = 0
    ) -> List[Individual]:
        """Selecciona mediante muestreo universal estocástico."""
        if not population:
            raise ValueError("La población está vacía")
        if num_parents <= 0:
            return []

        weights = np.array([ind.fitness for ind in population], dtype=np.float64)

        # Desplazar si hay fitness <= 0
        min_w = weights.min()
        if min_w <= 0:
            weights = weights - min_w + 1e-10

        total = weights.sum()
        if total <= 0:
            raise ValueError(
                "No se pudieron construir pesos válidos para selección universal"
            )

        cumulative = np.cumsum(weights / total)
        cumulative[-1] = 1.0

        # K punteros equiespaciados desde un único punto de inicio aleatorio
        r = np.random.uniform(0.0, 1.0)
        pointers = (r + np.arange(num_parents)) / num_parents
        indices = np.searchsorted(cumulative, pointers)
        np.clip(indices, 0, len(population) - 1, out=indices)

        return [population[i] for i in indices]


class BoltzmannSelection(SelectionMethod):
    """
    Selección entrópica de Boltzmann.

    La pseudo-aptitud de cada individuo depende de la temperatura T(t),
    que decrece con las generaciones:

        T(t) = Tc + (T0 - Tc) * exp(-k * t)

        ExpVal(i, g, T) = exp(f(i)/T) / mean(exp(f(x)/T))

    Los ExpVal se usan como pesos en Ruleta. A temperatura alta se favorece
    la exploración; a temperatura baja, la explotación.
    """

    def __init__(self, t0: float = 100.0, tc: float = 1.0, k: float = 0.005):
        """
        Args:
            t0: Temperatura inicial (alta → exploración).
            tc: Temperatura mínima asintótica.
            k:  Constante de decaimiento.
        """
        if t0 <= 0 or tc <= 0:
            raise ValueError("t0 y tc deben ser positivos")
        if k <= 0:
            raise ValueError("k debe ser positivo")
        self.t0 = t0
        self.tc = tc
        self.k = k

    def _temperature(self, generation: int) -> float:
        """Calcula T(t) = Tc + (T0 - Tc) * exp(-k * t)."""
        return self.tc + (self.t0 - self.tc) * math.exp(-self.k * generation)

    def select(
        self, population: List[Individual], num_parents: int, generation: int = 0
    ) -> List[Individual]:
        """Selecciona mediante Boltzmann con temperatura variable."""
        if not population:
            raise ValueError("La población está vacía")

        T = self._temperature(generation)

        # Estabilidad numérica: restar el máximo antes de exp (log-sum-exp trick)
        fitness_arr = np.array([ind.fitness for ind in population], dtype=np.float64)
        scaled = fitness_arr / T
        scaled -= scaled.max()
        exp_vals = np.exp(scaled)

        # Pseudo-aptitud normalizada por el promedio poblacional → pesos para ruleta
        weights = exp_vals / exp_vals.mean()
        probs = weights / weights.sum()

        indices = np.random.choice(len(population), size=num_parents, p=probs)
        return [population[i] for i in indices]


class RankSelection(SelectionMethod):
    """
    Selección por ranking.

    Los individuos se ordenan por fitness y la probabilidad de selección
    es proporcional a su posición en el ranking (mejor posición = mayor prob).
    """

    def select(
        self, population: List[Individual], num_parents: int, generation: int = 0
    ) -> List[Individual]:
        """Selecciona según ranking usando pseudo-aptitud teórica."""
        if not population:
            raise ValueError("La población está vacía")
        if num_parents <= 0:
            return []

        # Ordenar por fitness (mayor primero = mejor)
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        n = len(sorted_pop)

        # Caso degenerado: con N=1 la fórmula da 0/1
        if n == 1:
            return [sorted_pop[0]] * num_parents

        # Pseudo-aptitud teórica: f'(i) = (N - rank(i)) / N, rank(i) en [1, N]
        # Con índice 0-based: pseudo[i] = (n - 1 - i) / n
        pseudo = (np.arange(n - 1, -1, -1, dtype=np.float64)) / n
        total = pseudo.sum()

        cumulative = np.cumsum(pseudo / total)
        cumulative[-1] = 1.0

        randoms = np.random.uniform(0.0, 1.0, num_parents)
        indices = np.searchsorted(cumulative, randoms)
        np.clip(indices, 0, n - 1, out=indices)

        return [sorted_pop[i] for i in indices]


class EliteSelection(SelectionMethod):
    """
    Selección élite.

    Ordena la población por fitness y selecciona cada individuo n(i) veces
    según la fórmula teórica: n(i) = ceil((K - i) / N)

    Donde i es el rank (0 = mejor), K = num_parents, N = tamaño de población.
    Es determinístico y muy performante, aunque restrictivo en diversidad.
    """

    def select(
        self, population: List[Individual], num_parents: int, generation: int = 0
    ) -> List[Individual]:
        """Selecciona según fórmula élite."""
        if not population:
            raise ValueError("La población está vacía")
        if num_parents <= 0:
            return []

        N = len(population)
        K = num_parents
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        selected = []
        for i, ind in enumerate(sorted_pop):
            times = max(0, math.ceil((K - i) / N))
            if times > 0:
                selected.extend([ind] * times)

        # Con la fórmula teórica, la suma debe dar exactamente K.
        # Dejamos safeguards por robustez numérica.
        if len(selected) > K:
            selected = selected[:K]
        elif len(selected) < K:
            selected.extend([sorted_pop[0]] * (K - len(selected)))

        return selected


def create_selection_method(
    method: str,
    tournament_size: int = 3,
    threshold: float = 0.75,
    boltzmann_t0: float = 100.0,
    boltzmann_tc: float = 1.0,
    boltzmann_k: float = 0.005,
) -> SelectionMethod:
    """
    Factory para crear métodos de selección.

    Args:
        method: Nombre del método ("elite", "tournament", "probabilistic_tournament",
                "roulette", "universal", "boltzmann", "rank").
        tournament_size: Tamaño del torneo (solo para tournament).
        threshold: Umbral del torneo probabilístico (0.5-1).
        boltzmann_t0: Temperatura inicial de Boltzmann.
        boltzmann_tc: Temperatura mínima de Boltzmann.
        boltzmann_k: Constante de decaimiento de Boltzmann.

    Returns:
        Instancia del método de selección.
    """
    method = method.lower()

    if method == "elite":
        return EliteSelection()
    elif method == "tournament":
        return TournamentSelection(tournament_size)
    elif method == "probabilistic_tournament":
        return ProbabilisticTournamentSelection(threshold)
    elif method == "roulette":
        return RouletteSelection()
    elif method == "universal":
        return UniversalSelection()
    elif method == "boltzmann":
        return BoltzmannSelection(boltzmann_t0, boltzmann_tc, boltzmann_k)
    elif method in ("rank", "ranking"):
        return RankSelection()
    else:
        raise ValueError(f"Método de selección desconocido: {method}")
