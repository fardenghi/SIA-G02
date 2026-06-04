"""
Estrategias de supervivencia.

Implementación de operadores de supervivencia para definir cómo se forma
la nueva generación a partir de padres e hijos.

Según la teoría:
- Supervivencia Aditiva: Se seleccionan N individuos del pool de padres + hijos
- Supervivencia Exclusiva: Los hijos tienen prioridad; padres completan si K < N
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from src.genetic.individual import Individual
from src.genetic.selection import SelectionMethod, create_selection_method

if TYPE_CHECKING:
    pass


class SurvivalMethod(ABC):
    """Clase base abstracta para estrategias de supervivencia."""

    @abstractmethod
    def survive(
        self,
        parents: List[Individual],
        offspring: List[Individual],
        target_size: int,
    ) -> List[Individual]:
        """
        Determina qué individuos sobreviven para formar la nueva generación.

        Args:
            parents: Lista de individuos de la generación actual (N).
            offspring: Lista de hijos generados (K).
            target_size: Tamaño de la nueva población (N).

        Returns:
            Lista de individuos que forman la nueva generación (tamaño N).
        """
        pass


class AdditiveSurvival(SurvivalMethod):
    """
    Supervivencia aditiva.

    La nueva generación se forma seleccionando N individuos del conjunto
    compuesto por los N padres + K hijos.

    Es decir: Pool = padres(N) + hijos(K), seleccionar N del pool.

    Ventaja: Permite preservar buenos individuos de generaciones anteriores.
    """

    def __init__(self, selection_method: SelectionMethod):
        """
        Args:
            selection_method: Método de selección para elegir sobrevivientes.
        """
        self.selection_method = selection_method

    def survive(
        self,
        parents: List[Individual],
        offspring: List[Individual],
        target_size: int,
    ) -> List[Individual]:
        """
        Selecciona N individuos del pool de padres + hijos.

        Args:
            parents: Lista de individuos de la generación actual.
            offspring: Lista de hijos generados.
            target_size: Tamaño de la nueva población.

        Returns:
            Lista de N sobrevivientes seleccionados del pool combinado.
        """
        # Formar pool combinado: padres + hijos
        pool = parents + offspring

        if len(pool) <= target_size:
            # Si el pool es menor o igual al tamaño objetivo, todos sobreviven
            # y completamos con los mejores si hace falta
            if len(pool) < target_size:
                sorted_pool = sorted(pool, key=lambda x: x.fitness, reverse=True)
                extras_needed = target_size - len(pool)
                return pool + sorted_pool[:extras_needed]
            return pool

        # Seleccionar N sobrevivientes del pool usando el método configurado
        survivors = self.selection_method.select(
            population=pool,
            num_parents=target_size,
            generation=0,  # No usado en la mayoría de métodos
        )

        return survivors


class ExclusiveSurvival(SurvivalMethod):
    """
    Supervivencia exclusiva.

    La nueva generación se forma de manera diferente según la cantidad
    de hijos K comparada con el tamaño objetivo N:

    - Si K > N: Seleccionar N individuos solo de entre los K hijos.
    - Si K ≤ N: Los K hijos entran directamente + seleccionar (N-K) de los padres.

    Característica: Los hijos tienen prioridad sobre los padres.
    """

    def __init__(self, selection_method: SelectionMethod):
        """
        Args:
            selection_method: Método de selección para elegir sobrevivientes
                              (usado cuando K > N o para completar con padres).
        """
        self.selection_method = selection_method

    def survive(
        self,
        parents: List[Individual],
        offspring: List[Individual],
        target_size: int,
    ) -> List[Individual]:
        """
        Forma la nueva generación según la lógica exclusiva.

        Args:
            parents: Lista de individuos de la generación actual.
            offspring: Lista de hijos generados.
            target_size: Tamaño de la nueva población.

        Returns:
            Lista de N sobrevivientes.
        """
        k = len(offspring)
        n = target_size

        if k > n:
            # Caso 1: K > N → seleccionar N de los K hijos
            survivors = self.selection_method.select(
                population=offspring,
                num_parents=n,
                generation=0,
            )
        elif k == n:
            # Caso especial: K = N → todos los hijos pasan directamente
            survivors = offspring
        else:
            # Caso 2: K < N → K hijos + seleccionar (N-K) de los padres
            survivors = list(offspring)  # Copiar hijos
            remaining = n - k

            if remaining > 0 and parents:
                # Seleccionar (N-K) individuos de los padres
                selected_parents = self.selection_method.select(
                    population=parents,
                    num_parents=min(remaining, len(parents)),
                    generation=0,
                )
                survivors.extend(selected_parents)

            # Si aún faltan (porque había pocos padres), completar con los mejores
            if len(survivors) < n:
                all_individuals = parents + offspring
                sorted_all = sorted(
                    all_individuals, key=lambda x: x.fitness, reverse=True
                )
                for ind in sorted_all:
                    if ind not in survivors:
                        survivors.append(ind)
                        if len(survivors) >= n:
                            break

        return survivors


def create_survival_method(
    method: str = "exclusive",
    selection_method: str = "elite",
    tournament_size: int = 3,
    threshold: float = 0.75,
    boltzmann_t0: float = 100.0,
    boltzmann_tc: float = 1.0,
    boltzmann_k: float = 0.005,
) -> SurvivalMethod:
    """
    Factory para crear estrategias de supervivencia.

    Args:
        method: Estrategia de supervivencia ("additive" o "exclusive").
        selection_method: Método de selección para elegir sobrevivientes
            ("elite", "tournament", "roulette", etc.).
        tournament_size: Tamaño del torneo (solo para tournament).
        threshold: Umbral del torneo probabilístico.
        boltzmann_t0: Temperatura inicial de Boltzmann.
        boltzmann_tc: Temperatura mínima de Boltzmann.
        boltzmann_k: Constante de decaimiento de Boltzmann.

    Returns:
        Instancia de la estrategia de supervivencia.
    """
    # Crear el método de selección para la supervivencia
    selection = create_selection_method(
        method=selection_method,
        tournament_size=tournament_size,
        threshold=threshold,
        boltzmann_t0=boltzmann_t0,
        boltzmann_tc=boltzmann_tc,
        boltzmann_k=boltzmann_k,
    )

    method_lower = method.lower()

    if method_lower == "additive":
        return AdditiveSurvival(selection_method=selection)
    elif method_lower == "exclusive":
        return ExclusiveSurvival(selection_method=selection)
    else:
        raise ValueError(
            f"Estrategia de supervivencia desconocida: {method}. "
            f"Opciones válidas: 'additive', 'exclusive'"
        )
