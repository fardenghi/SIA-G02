"""
Motor de algoritmo genético.

Orquesta todos los componentes para ejecutar el algoritmo evolutivo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from src.genetic.individual import Individual
from src.genetic.population import Population
from src.genetic.selection import SelectionMethod, create_selection_method
from src.genetic.crossover import CrossoverMethod, create_crossover_method
from src.genetic.mutation import (
    Mutator,
    MutationParams,
    MutationType,
    AdaptiveSigma,
    create_mutation_params,
)
from src.genetic.survival import SurvivalMethod, create_survival_method
from src.fitness.mse import FitnessEvaluator


@dataclass
class EvolutionConfig:
    """
    Configuración del algoritmo evolutivo.

    Attributes:
        population_size: Tamaño de la población.
        num_triangles: Número de triángulos por individuo.
        shape_type: Familia de formas por individuo.
        max_generations: Número máximo de generaciones.
        fitness_threshold: Umbral de fitness para parada temprana
            (None = sin umbral).
        alpha_min: Alfa mínimo para triángulos.
        alpha_max: Alfa máximo para triángulos.
    """

    population_size: int = 100
    num_triangles: int = 50
    shape_type: str = "triangle"
    max_generations: int = 5000
    fitness_threshold: Optional[float] = None
    alpha_min: float = 0.1
    alpha_max: float = 0.8
    elite_count: int = 0
    stagnation_threshold: float = 0.0005
    max_patience: int = 20
    transition_methods: Optional[List[str]] = None


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
        survival_method: Optional[SurvivalMethod] = None,
        offspring_ratio: float = 1.0,
        fitness_method: str = "linear",
        fitness_scale: float = 0.1,
        fitness_detail_weight_base: float = 0.3,
        fitness_composite_alpha: float = 0.5,
        fitness_composite_beta: float = 0.2,
        fitness_composite_gamma: float = 0.3,
        renderer: str = "cpu",
    ):
        """
        Args:
            target_image: Imagen objetivo a aproximar.
            config: Configuración del algoritmo.
            selection_method: Método de selección.
            crossover_method: Método de cruza.
            mutator: Operador de mutación.
            survival_method: Estrategia de supervivencia (opcional).
                Si es None, se usa supervivencia exclusiva por defecto.
            offspring_ratio: Ratio de hijos a generar respecto al tamaño
                de la población (K = N * offspring_ratio).
            fitness_method: Función de fitness a usar.
            fitness_scale: Escala para el método exponencial.
            fitness_detail_weight_base: Peso base para regiones lisas en detail_weighted.
            fitness_composite_alpha: Peso de (1-SSIM) en el fitness compuesto.
            fitness_composite_beta: Peso de MSE_norm en el fitness compuesto.
            fitness_composite_gamma: Peso de EdgeLoss en el fitness compuesto.
            renderer: Backend de renderizado ("cpu" o "gpu").
        """
        self.config = config
        self.evaluator = FitnessEvaluator(
            target_image,
            method=fitness_method,
            exponential_scale=fitness_scale,
            renderer=renderer,
            shape_type=config.shape_type,
            detail_weight_base=fitness_detail_weight_base,
            composite_alpha=fitness_composite_alpha,
            composite_beta=fitness_composite_beta,
            composite_gamma=fitness_composite_gamma,
        )
        self.selection = selection_method
        self.crossover = crossover_method
        self.base_crossover = (
            crossover_method  # Guardar el operador configurado en YAML
        )

        from src.genetic.selection import TournamentSelection

        self.base_tournament_size = (
            getattr(self.selection, "tournament_size", 3)
            if isinstance(self.selection, TournamentSelection)
            else 3
        )

        self.mutator = mutator
        self.survival = survival_method
        self.offspring_ratio = offspring_ratio

        self.base_fitness_method = fitness_method

        self.cv_methods: List[str] = []
        if self.config.transition_methods:
            self.cv_methods = list(self.config.transition_methods)
            if self.base_fitness_method not in self.cv_methods:
                self.cv_methods.insert(0, self.base_fitness_method)

        self.cv_idx = 0
        if self.cv_methods and self.base_fitness_method in self.cv_methods:
            self.cv_idx = self.cv_methods.index(self.base_fitness_method)

        # Precompute maps for dynamic transitions (AOS optimization)
        if self.cv_methods:
            self.evaluator.preload_maps(
                self.cv_methods, detail_weight_base=fitness_detail_weight_base
            )

        self.population: Optional[Population] = None
        self.history: List[dict] = []

        # --- Computación Evolutiva Avanzada ---
        self.active_fitness_method = fitness_method
        self.hall_of_fame = {}  # type: dict[str, Individual]

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
            shape_type=self.config.shape_type,
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
        elite_count = min(self.config.elite_count, pop_size)

        # Elitismo: copiar los mejores individuos antes de cualquier operación
        if elite_count > 0:
            sorted_parents = sorted(
                population.individuals, key=lambda ind: ind.fitness or 0.0, reverse=True
            )
            elites = [ind.copy() for ind in sorted_parents[:elite_count]]
        else:
            elites = []

        # Spots que deben llenarse con individuos no-élite
        effective_size = pop_size - elite_count

        # Actualizar error map para mutación guiada (una vez por generación)
        if self.mutator.mutation_type in {
            MutationType.ERROR_MAP_GUIDED,
            MutationType.SSIM_MAP_GUIDED,
        }:
            best = max(population.individuals, key=lambda ind: ind.fitness or 0.0)
            rendered = self.evaluator.canvas.render_to_array(best)

            if self.mutator.mutation_type == MutationType.ERROR_MAP_GUIDED:
                diff = rendered.astype(np.float32) - self.evaluator.target.astype(
                    np.float32
                )
                error_map = (diff**2).mean(axis=2)  # (H, W)
            else:
                # SSIM_MAP_GUIDED: Extraemos el mapa local de skimage, donde valores bajos indican rotura
                from skimage.metrics import structural_similarity

                _, ssim_map = structural_similarity(
                    self.evaluator.target,
                    rendered,
                    channel_axis=-1,
                    data_range=255,
                    full=True,
                )
                if ssim_map.ndim == 3:
                    ssim_map = ssim_map.mean(axis=2)
                # Invertimos (1.0 - SSIM) para que el Mutator (que maximiza sampling en errores altos) ataque estas zonas.
                error_map = 1.0 - ssim_map

            self.mutator.set_error_map(error_map)

        if effective_size == 0:
            return Population(individuals=elites, generation=population.generation + 1)

        # Calcular cantidad de hijos a generar (K = N * offspring_ratio)
        # Se genera siempre sobre pop_size completo para mantener diversidad
        num_offspring = max(pop_size, int(pop_size * self.offspring_ratio))

        # Seleccionar padres (necesitamos suficientes para generar num_offspring hijos)
        num_parents_needed = num_offspring  # Un padre por hijo aproximadamente
        parents = self.selection.select(
            population.individuals,
            num_parents=num_parents_needed,
            generation=population.generation,
        )

        # Cruza y mutación para generar hijos
        offspring = []

        # --- ELITISMO MULTIOBJETIVO (SALÓN DE LA FAMA) ---
        # Inyectamos incondicionalmente a los campeones históricos.
        # Al poner fitness=None, obligamos al motor a re-evaluarlos con la moneda local
        # ("El Pasaporte Genético") evitando el engaño de Cross-Metric Fitness.
        for champ in self.hall_of_fame.values():
            inmigrante = champ.copy()
            inmigrante.fitness = None
            offspring.append(inmigrante)

        i = 0
        while len(offspring) < num_offspring:
            parent1 = parents[i % len(parents)]
            parent2 = parents[(i + 1) % len(parents)]

            # Cruza
            child1, child2 = self.crossover.crossover(parent1, parent2)

            # Mutación
            child1 = self.mutator.mutate(child1)
            child2 = self.mutator.mutate(child2)

            offspring.append(child1)
            if len(offspring) < num_offspring:
                offspring.append(child2)

            i += 2

        # Evaluar fitness de los hijos
        self.evaluator.evaluate_population(offspring)

        # Aplicar estrategia de supervivencia sobre los spots no-élite
        if self.survival is not None:
            new_individuals = self.survival.survive(
                parents=population.individuals,
                offspring=offspring,
                target_size=effective_size,
            )
        else:
            # Comportamiento por defecto: supervivencia exclusiva simple
            if len(offspring) > effective_size:
                sorted_offspring = sorted(
                    offspring, key=lambda x: x.fitness, reverse=True
                )
                new_individuals = sorted_offspring[:effective_size]
            else:
                new_individuals = offspring[:effective_size]

        return Population(
            individuals=elites + new_individuals, generation=population.generation + 1
        )

    def _apply_crossover_profile(self, fitness_method: str) -> None:
        """Cambia dinámicamente el método de cruza según la fase."""
        from src.genetic.crossover import create_crossover_method

        if fitness_method in {"ssim", "detail_weighted"}:
            self.crossover = create_crossover_method(
                "spatial_zindex", probability=self.base_crossover.probability
            )
        else:
            self.crossover = self.base_crossover

    def _apply_selection_profile(self, fitness_method: str) -> None:
        """Torneo Dinámico: Ajusta la Presión Selectiva acoplando el tamaño del torneo."""
        from src.genetic.selection import TournamentSelection

        if isinstance(self.selection, TournamentSelection):
            if fitness_method in {"ssim", "detail_weighted"}:
                # Presión Brutal para convergencia fina: 5 veces la configurada por el usuario
                self.selection.tournament_size = self.base_tournament_size * 5
            else:
                # Presión Baja para exploración (default del yaml)
                self.selection.tournament_size = self.base_tournament_size

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

        # Acoplar el perfil de mutación y cruza inicial a la fase de fitness correspondiente
        self.mutator.apply_profile(self.base_fitness_method)
        self._apply_crossover_profile(self.base_fitness_method)
        self._apply_selection_profile(self.base_fitness_method)

        # Evaluar población inicial
        self.evaluate_population(self.population)

        best_ever = self.population.best.copy()
        best_fitness_ever = best_ever.fitness
        stopped_early = False

        # Registrar estadísticas iniciales
        stats = self.population.get_statistics()
        self.history.append(stats)
        self._notify_generation(0, self.population, stats)

        patience_counter = 0

        # Bucle evolutivo principal
        for gen in range(1, self.config.max_generations + 1):
            previous_gen_best_fitness = self.population.best.fitness

            # Evolucionar
            self.population = self.evolve_generation(self.population)

            # --- Curriculum Learning (Dinámico y Genérico) ---
            if self.cv_methods and len(self.cv_methods) > 1:
                current_gen_best_fitness = self.population.best.fitness
                improvement = current_gen_best_fitness - previous_gen_best_fitness

                if improvement < self.config.stagnation_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0

                # --- El Gatillo de Transición (Hot Restart) ---
                if patience_counter >= self.config.max_patience:
                    self.cv_idx = (self.cv_idx + 1) % len(self.cv_methods)
                    next_method = self.cv_methods[self.cv_idx]

                    print(
                        f"\n[Engine] Estancamiento detectado en gen {gen}: Cambiando métrica de fitness a '{next_method}'."
                    )
                    print(
                        "[Engine] Realizando 'Hot Restart': invalidando y reconectando la población base y genotipo Élite al vuelo..."
                    )
                    self.evaluator.set_method(next_method)
                    self.active_fitness_method = next_method
                    self.mutator.apply_profile(next_method)
                    self._apply_crossover_profile(next_method)
                    self._apply_selection_profile(next_method)
                    patience_counter = 0

                    # Reinicio en Caliente (Hot Restart)
                    for ind in self.population.individuals:
                        ind.fitness = None

                    self.evaluate_population(self.population)

                    best_ever.fitness = None
                    best_fitness_ever = self.evaluator.evaluate(best_ever)

            # Actualizar mejor global y Salón de la Fama
            current_best = self.population.best

            # Actualizamos el Salón de la Fama con el rey local actual
            self.hall_of_fame[self.active_fitness_method] = current_best.copy()

            if current_best.fitness > best_fitness_ever:
                best_ever = current_best.copy()
                best_fitness_ever = best_ever.fitness
                self._notify_improvement(gen, best_ever, best_fitness_ever)

            # Actualizar sigma adaptativo con el mejor fitness de esta generación
            self.mutator.update(best_fitness_ever)

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
    survival_method: str = "exclusive",
    survival_selection_method: str = "elite",
    offspring_ratio: float = 1.0,
    fitness_method: str = "linear",
    fitness_scale: float = 0.1,
    fitness_detail_weight_base: float = 0.3,
    fitness_composite_alpha: float = 0.5,
    fitness_composite_beta: float = 0.2,
    fitness_composite_gamma: float = 0.3,
    renderer: str = "cpu",
    adaptive_sigma: Optional[AdaptiveSigma] = None,
) -> GeneticEngine:
    """
    Factory para crear un motor genético configurado.

    Args:
        target_image: Imagen objetivo.
        config: Configuración de evolución.
        selection_method: Método de selección de padres.
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
        survival_method: Estrategia de supervivencia ("additive" o "exclusive").
        survival_selection_method: Método de selección para supervivientes.
        offspring_ratio: Ratio de hijos a generar (K = N * offspring_ratio).

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
        mutator = Mutator(mutation_params, adaptive_sigma=adaptive_sigma)
    else:
        params = create_mutation_params(
            mutation_method=mutation_method,
            probability=mutation_probability,
            gene_probability=mutation_gene_probability,
            max_genes=mutation_max_genes,
        )
        mutator = Mutator(params, adaptive_sigma=adaptive_sigma)

    # Crear estrategia de supervivencia
    survival = create_survival_method(
        method=survival_method,
        selection_method=survival_selection_method,
        tournament_size=tournament_size,
        threshold=threshold,
        boltzmann_t0=boltzmann_t0,
        boltzmann_tc=boltzmann_tc,
        boltzmann_k=boltzmann_k,
    )

    return GeneticEngine(
        target_image=target_image,
        config=config,
        selection_method=selection,
        crossover_method=crossover,
        mutator=mutator,
        survival_method=survival,
        offspring_ratio=offspring_ratio,
        fitness_method=fitness_method,
        fitness_scale=fitness_scale,
        fitness_detail_weight_base=fitness_detail_weight_base,
        fitness_composite_alpha=fitness_composite_alpha,
        fitness_composite_beta=fitness_composite_beta,
        fitness_composite_gamma=fitness_composite_gamma,
        renderer=renderer,
    )
