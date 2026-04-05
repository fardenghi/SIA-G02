"""
Motor de algoritmo genético.

Orquesta todos los componentes para ejecutar el algoritmo evolutivo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from PIL import Image

from src.genetic.individual import Individual
from src.genetic.population import Population
from src.genetic.selection import SelectionMethod, create_selection_method
from src.genetic.crossover import CrossoverMethod, create_crossover_method
from src.genetic.mutation import Mutator, MutationParams, create_mutation_params
from src.fitness.mse import FitnessEvaluator


@dataclass
class EvolutionConfig:
    """
    Configuración del algoritmo evolutivo.

    Attributes:
        population_size: Tamaño de la población.
        num_triangles: Número de triángulos por individuo.
        max_generations: Número máximo de generaciones.
        fitness_threshold: Umbral de fitness para parada temprana
            (None = sin umbral).
        alpha_min: Alfa mínimo para triángulos.
        alpha_max: Alfa máximo para triángulos.
    """

    population_size: int = 100
    num_triangles: int = 50
    max_generations: int = 5000
    fitness_threshold: Optional[float] = None
    alpha_min: float = 0.1
    alpha_max: float = 0.8


@dataclass
class EvolutionResult:
    """
    Resultado de la evolución.

    Attributes:
        best_individual: Mejor individuo encontrado.
        best_fitness: Fitness del mejor individuo.
        generations: Número de generaciones ejecutadas.
        elapsed_time: Tiempo total de ejecución en segundos.
        history: Historial de estadísticas por generación.
        stopped_early: Si se detuvo por alcanzar el umbral de fitness.
    """

    best_individual: Individual
    best_fitness: float
    generations: int
    elapsed_time: float
    history: List[dict] = field(default_factory=list)
    stopped_early: bool = False


class GeneticEngine:
    """
    Motor principal del algoritmo genético.

    Coordina la inicialización, evaluación, selección, cruza y mutación
    para evolucionar una población hacia la imagen objetivo.
    """

    def __init__(
        self,
        target_image: Image.Image,
        config: EvolutionConfig,
        selection_method: SelectionMethod,
        crossover_method: CrossoverMethod,
        mutator: Mutator,
    ):
        """
        Args:
            target_image: Imagen objetivo a aproximar.
            config: Configuración del algoritmo.
            selection_method: Método de selección.
            crossover_method: Método de cruza.
            mutator: Operador de mutación.
        """
        self.config = config
        self.evaluator = FitnessEvaluator(target_image)
        self.selection = selection_method
        self.crossover = crossover_method
        self.mutator = mutator

        self.population: Optional[Population] = None
        self.history: List[dict] = []

        # Callbacks para eventos
        self._on_generation_callbacks: List[Callable] = []
        self._on_improvement_callbacks: List[Callable] = []

    @property
    def width(self) -> int:
        """Ancho de la imagen objetivo."""
        return self.evaluator.width

    @property
    def height(self) -> int:
        """Alto de la imagen objetivo."""
        return self.evaluator.height

    def on_generation(self, callback: Callable[[int, Population, dict], None]):
        """
        Registra un callback para cada generación.

        Args:
            callback: Función(generation, population, stats).
        """
        self._on_generation_callbacks.append(callback)

    def on_improvement(self, callback: Callable[[int, Individual, float], None]):
        """
        Registra un callback cuando hay mejora.

        Args:
            callback: Función(generation, best_individual, best_fitness).
        """
        self._on_improvement_callbacks.append(callback)

    def initialize_population(self) -> Population:
        """
        Inicializa una población aleatoria.

        Returns:
            Nueva población.
        """
        self.population = Population.random(
            size=self.config.population_size,
            num_triangles=self.config.num_triangles,
            alpha_min=self.config.alpha_min,
            alpha_max=self.config.alpha_max,
        )
        return self.population

    def evaluate_population(self, population: Population):
        """
        Evalúa el fitness de toda la población.

        Args:
            population: Población a evaluar.
        """
        self.evaluator.evaluate_population(population.individuals)

    def evolve_generation(self, population: Population) -> Population:
        """
        Evoluciona una generación.

        Args:
            population: Población actual.

        Returns:
            Nueva población (siguiente generación).
        """
        pop_size = len(population)
        new_individuals = []

        # Seleccionar padres (necesitamos pop_size padres para generar pop_size hijos)
        parents = self.selection.select(
            population.individuals,
            num_parents=pop_size,
            generation=population.generation,
        )

        # Cruza y mutación
        i = 0
        while len(new_individuals) < pop_size:
            parent1 = parents[i % len(parents)]
            parent2 = parents[(i + 1) % len(parents)]

            # Cruza
            child1, child2 = self.crossover.crossover(parent1, parent2)

            # Mutación
            child1 = self.mutator.mutate(child1)
            child2 = self.mutator.mutate(child2)

            new_individuals.append(child1)
            if len(new_individuals) < pop_size:
                new_individuals.append(child2)

            i += 2

        return Population(
            individuals=new_individuals, generation=population.generation + 1
        )

    def run(self) -> EvolutionResult:
        """
        Ejecuta el algoritmo genético completo.

        Returns:
            Resultado de la evolución.
        """
        start_time = time.time()
        self.history = []

        # Inicializar población
        if self.population is None:
            self.initialize_population()

        # Evaluar población inicial
        self.evaluate_population(self.population)

        best_ever = self.population.best.copy()
        best_fitness_ever = best_ever.fitness
        stopped_early = False

        # Registrar estadísticas iniciales
        stats = self.population.get_statistics()
        self.history.append(stats)
        self._notify_generation(0, self.population, stats)

        # Bucle evolutivo principal
        for gen in range(1, self.config.max_generations + 1):
            # Evolucionar
            self.population = self.evolve_generation(self.population)

            # Evaluar nueva generación
            self.evaluate_population(self.population)

            # Actualizar mejor global
            current_best = self.population.best
            if current_best.fitness > best_fitness_ever:
                best_ever = current_best.copy()
                best_fitness_ever = best_ever.fitness
                self._notify_improvement(gen, best_ever, best_fitness_ever)

            # Registrar estadísticas
            stats = self.population.get_statistics()
            self.history.append(stats)
            self._notify_generation(gen, self.population, stats)

            # Verificar criterio de parada
            if self.config.fitness_threshold is not None:
                if best_fitness_ever >= self.config.fitness_threshold:
                    stopped_early = True
                    break

        elapsed_time = time.time() - start_time

        return EvolutionResult(
            best_individual=best_ever,
            best_fitness=best_fitness_ever,
            generations=self.population.generation,
            elapsed_time=elapsed_time,
            history=self.history,
            stopped_early=stopped_early,
        )

    def _notify_generation(self, gen: int, population: Population, stats: dict):
        """Notifica a los callbacks de generación."""
        for callback in self._on_generation_callbacks:
            callback(gen, population, stats)

    def _notify_improvement(self, gen: int, individual: Individual, fitness: float):
        """Notifica a los callbacks de mejora."""
        for callback in self._on_improvement_callbacks:
            callback(gen, individual, fitness)


def create_engine(
    target_image: Image.Image,
    config: EvolutionConfig,
    selection_method: str = "tournament",
    tournament_size: int = 3,
    crossover_method: str = "single_point",
    crossover_probability: float = 0.8,
    mutation_params: Optional[MutationParams] = None,
    mutation_method: str = "uniform_multigen",
    mutation_probability: float = 0.3,
    mutation_gene_probability: float = 0.1,
    mutation_max_genes: int = 3,
    threshold: float = 0.75,
    boltzmann_t0: float = 100.0,
    boltzmann_tc: float = 1.0,
    boltzmann_k: float = 0.005,
) -> GeneticEngine:
    """
    Factory para crear un motor genético configurado.

    Args:
        target_image: Imagen objetivo.
        config: Configuración de evolución.
        selection_method: Método de selección.
        tournament_size: Tamaño de torneo.
        crossover_method: Método de cruza.
        crossover_probability: Probabilidad de cruza.
        mutation_params: Parámetros de mutación (si se provee, ignora los otros params de mutación).
        mutation_method: Método de mutación ("single_gene", "limited_multigen",
            "uniform_multigen", "complete").
        mutation_probability: Probabilidad de mutación (Pm).
        mutation_gene_probability: Probabilidad por gen (para uniform_multigen).
        mutation_max_genes: Máximo de genes a mutar (M) para limited_multigen.
        threshold: Umbral para torneo probabilístico.
        boltzmann_t0: Temperatura inicial de Boltzmann.
        boltzmann_tc: Temperatura mínima de Boltzmann.
        boltzmann_k: Constante de decaimiento de Boltzmann.

    Returns:
        Motor genético configurado.
    """
    selection = create_selection_method(
        method=selection_method,
        tournament_size=tournament_size,
        threshold=threshold,
        boltzmann_t0=boltzmann_t0,
        boltzmann_tc=boltzmann_tc,
        boltzmann_k=boltzmann_k,
    )

    crossover = create_crossover_method(
        method=crossover_method, probability=crossover_probability
    )

    # Si se proveen mutation_params, usarlos directamente
    # Si no, crear los params a partir de los argumentos individuales
    if mutation_params is not None:
        mutator = Mutator(mutation_params)
    else:
        params = create_mutation_params(
            mutation_method=mutation_method,
            probability=mutation_probability,
            gene_probability=mutation_gene_probability,
            max_genes=mutation_max_genes,
        )
        mutator = Mutator(params)

    return GeneticEngine(
        target_image=target_image,
        config=config,
        selection_method=selection,
        crossover_method=crossover,
        mutator=mutator,
    )
