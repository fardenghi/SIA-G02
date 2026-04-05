"""Tests para la población y motor evolutivo."""

import pytest
from PIL import Image
import numpy as np

from src.genetic.individual import Individual
from src.genetic.population import Population
from src.genetic.engine import (
    GeneticEngine,
    EvolutionConfig,
    EvolutionResult,
    create_engine,
)
from src.genetic.selection import TournamentSelection
from src.genetic.crossover import SinglePointCrossover
from src.genetic.mutation import Mutator, MutationParams


class TestPopulation:
    """Tests para la clase Population."""

    def test_create_population(self):
        """Debe crear una población con individuos."""
        individuals = [Individual.random(num_triangles=5) for _ in range(10)]
        pop = Population(individuals=individuals)

        assert len(pop) == 10
        assert pop.generation == 0

    def test_random_population(self):
        """Debe generar una población aleatoria."""
        pop = Population.random(size=20, num_triangles=10)

        assert len(pop) == 20
        assert all(len(ind) == 10 for ind in pop)

    def test_best_individual(self):
        """Debe retornar el individuo con mayor fitness."""
        individuals = [Individual.random(num_triangles=5) for _ in range(5)]
        for i, ind in enumerate(individuals):
            ind.fitness = float(i * 100)  # 0, 100, 200, 300, 400

        pop = Population(individuals=individuals)

        assert pop.best.fitness == 400.0

    def test_worst_individual(self):
        """Debe retornar el individuo con menor fitness."""
        individuals = [Individual.random(num_triangles=5) for _ in range(5)]
        for i, ind in enumerate(individuals):
            ind.fitness = float(i * 100)

        pop = Population(individuals=individuals)

        assert pop.worst.fitness == 0.0

    def test_average_fitness(self):
        """Debe calcular el fitness promedio correctamente."""
        individuals = [Individual.random(num_triangles=5) for _ in range(4)]
        for i, ind in enumerate(individuals):
            ind.fitness = float(i * 100)  # 0, 100, 200, 300 -> avg = 150

        pop = Population(individuals=individuals)

        assert pop.average_fitness == 150.0

    def test_statistics(self):
        """Debe retornar estadísticas correctas."""
        individuals = [Individual.random(num_triangles=5) for _ in range(3)]
        individuals[0].fitness = 100.0
        individuals[1].fitness = 200.0
        individuals[2].fitness = 300.0

        pop = Population(individuals=individuals, generation=5)
        stats = pop.get_statistics()

        assert stats["generation"] == 5
        assert stats["best_fitness"] == 300.0
        assert stats["worst_fitness"] == 100.0
        assert stats["avg_fitness"] == 200.0
        assert stats["size"] == 3

    def test_sorted_by_fitness(self):
        """Debe ordenar por fitness (mejor primero)."""
        individuals = [Individual.random(num_triangles=5) for _ in range(4)]
        individuals[0].fitness = 300.0
        individuals[1].fitness = 100.0
        individuals[2].fitness = 200.0
        individuals[3].fitness = 50.0

        pop = Population(individuals=individuals)
        sorted_inds = pop.sorted_by_fitness()

        assert sorted_inds[0].fitness == 300.0
        assert sorted_inds[1].fitness == 200.0
        assert sorted_inds[2].fitness == 100.0
        assert sorted_inds[3].fitness == 50.0

    def test_iteration(self):
        """Debe permitir iteración sobre individuos."""
        individuals = [Individual.random(num_triangles=3) for _ in range(5)]
        pop = Population(individuals=individuals)

        count = 0
        for ind in pop:
            assert isinstance(ind, Individual)
            count += 1

        assert count == 5


class TestEvolutionConfig:
    """Tests para la configuración de evolución."""

    def test_default_config(self):
        """Debe crear configuración con valores por defecto."""
        config = EvolutionConfig()

        assert config.population_size == 100
        assert config.num_triangles == 50
        assert config.max_generations == 5000

    def test_custom_config(self):
        """Debe aceptar valores personalizados."""
        config = EvolutionConfig(
            population_size=50,
            num_triangles=20,
            max_generations=100,
            fitness_threshold=0.5,
        )

        assert config.population_size == 50
        assert config.num_triangles == 20
        assert config.max_generations == 100
        assert config.fitness_threshold == 0.5


class TestGeneticEngine:
    """Tests para el motor genético."""

    @pytest.fixture
    def simple_engine(self):
        """Crea un motor simple para tests."""
        # Imagen pequeña para tests rápidos
        target = Image.new("RGB", (20, 20), color=(128, 128, 128))
        config = EvolutionConfig(population_size=10, num_triangles=5, max_generations=5)
        return create_engine(target, config)

    def test_initialize_population(self, simple_engine):
        """Debe inicializar población correctamente."""
        pop = simple_engine.initialize_population()

        assert len(pop) == 10
        assert all(len(ind) == 5 for ind in pop)

    def test_evaluate_population(self, simple_engine):
        """Debe evaluar fitness de la población."""
        pop = simple_engine.initialize_population()

        # Antes de evaluar, fitness es None
        assert all(ind.fitness is None for ind in pop)

        simple_engine.evaluate_population(pop)

        # Después de evaluar, todos tienen fitness
        assert all(ind.fitness is not None for ind in pop)
        assert all(0 < ind.fitness <= 1 for ind in pop)

    def test_evolve_generation(self, simple_engine):
        """Debe crear una nueva generación."""
        pop = simple_engine.initialize_population()
        simple_engine.evaluate_population(pop)

        new_pop = simple_engine.evolve_generation(pop)

        assert len(new_pop) == len(pop)
        assert new_pop.generation == pop.generation + 1

    def test_run_evolution(self, simple_engine):
        """Debe ejecutar el algoritmo completo."""
        result = simple_engine.run()

        assert isinstance(result, EvolutionResult)
        assert result.best_individual is not None
        assert 0 < result.best_fitness <= 1
        assert result.generations == 5
        assert result.elapsed_time > 0
        assert len(result.history) > 0

    def test_early_stopping(self):
        """Debe detenerse si alcanza el umbral de fitness."""
        # Imagen blanca como objetivo
        target = Image.new("RGB", (10, 10), color=(255, 255, 255))
        config = EvolutionConfig(
            population_size=20,
            num_triangles=3,
            max_generations=1000,
            fitness_threshold=1e-10,  # Umbral muy bajo, debería parar rápido
        )
        engine = create_engine(target, config)

        result = engine.run()

        # Debería parar antes de las 1000 generaciones
        # (con umbral muy bajo, cualquier fitness válido debería alcanzarlo)
        assert result.generations < 1000 or result.stopped_early

    def test_callbacks(self, simple_engine):
        """Debe llamar a los callbacks correctamente."""
        generation_calls = []
        improvement_calls = []

        def on_gen(gen, pop, stats):
            generation_calls.append(gen)

        def on_improve(gen, ind, fitness):
            improvement_calls.append((gen, fitness))

        simple_engine.on_generation(on_gen)
        simple_engine.on_improvement(on_improve)

        simple_engine.run()

        # Debe llamar on_generation para cada generación (0 a max)
        assert len(generation_calls) == 6  # 0, 1, 2, 3, 4, 5
        assert 0 in generation_calls

    def test_fitness_improves_or_stays(self, simple_engine):
        """El mejor fitness global debe mejorar o mantenerse."""
        result = simple_engine.run()

        # El resultado final debe tener un fitness razonable
        assert 0 < result.best_fitness <= 1

        # El fitness final debe ser mayor o igual al inicial
        initial_best = result.history[0]["best_fitness"]
        assert result.best_fitness >= initial_best


class TestCreateEngine:
    """Tests para el factory create_engine."""

    def test_create_with_defaults(self):
        """Debe crear motor con configuración por defecto."""
        target = Image.new("RGB", (30, 30))
        config = EvolutionConfig()

        engine = create_engine(target, config)

        assert engine is not None
        assert engine.width == 30
        assert engine.height == 30

    def test_create_with_custom_params(self):
        """Debe crear motor con parámetros personalizados."""
        target = Image.new("RGB", (20, 20))
        config = EvolutionConfig(population_size=50)

        engine = create_engine(
            target,
            config,
            selection_method="roulette",
            crossover_method="uniform",
            mutation_params=MutationParams(probability=0.5),
        )

        assert engine is not None
        assert engine.config.population_size == 50
