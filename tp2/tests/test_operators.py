"""Tests para los operadores genéticos (selección, cruza, mutación)."""

import pytest
import random
from src.genetic.individual import Individual, Triangle
from src.genetic.selection import (
    TournamentSelection,
    RouletteSelection,
    RankSelection,
    ElitistSelection,
    create_selection_method,
)
from src.genetic.crossover import (
    SinglePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    create_crossover_method,
)
from src.genetic.mutation import (
    MutationParams,
    mutate_triangle,
    mutate_individual,
    Mutator,
)


def create_population_with_fitness(n: int = 10) -> list[Individual]:
    """Crea una población de prueba con fitness asignado."""
    population = []
    for i in range(n):
        ind = Individual.random(num_triangles=5)
        ind.fitness = float((i + 1) * 100)  # 100, 200, 300, ...
        population.append(ind)
    return population


class TestTournamentSelection:
    """Tests para selección por torneo."""

    def test_basic_selection(self):
        """Debe seleccionar la cantidad correcta de padres."""
        pop = create_population_with_fitness(20)
        selector = TournamentSelection(tournament_size=3)

        selected = selector.select(pop, num_parents=10)

        assert len(selected) == 10
        assert all(isinstance(ind, Individual) for ind in selected)

    def test_winner_is_best(self):
        """El ganador del torneo debe ser el de mayor fitness."""
        random.seed(42)
        pop = create_population_with_fitness(10)
        selector = TournamentSelection(tournament_size=5)

        # Seleccionar muchos y verificar tendencia hacia mejores
        selected = selector.select(pop, num_parents=100)
        avg_fitness = sum(ind.fitness for ind in selected) / len(selected)

        # El promedio debería ser mayor que el promedio de toda la población
        pop_avg = sum(ind.fitness for ind in pop) / len(pop)
        assert avg_fitness > pop_avg

    def test_invalid_tournament_size(self):
        """Debe fallar con tamaño de torneo inválido."""
        with pytest.raises(ValueError):
            TournamentSelection(tournament_size=1)


class TestRouletteSelection:
    """Tests para selección por ruleta."""

    def test_basic_selection(self):
        """Debe seleccionar la cantidad correcta de padres."""
        pop = create_population_with_fitness(20)
        selector = RouletteSelection()

        selected = selector.select(pop, num_parents=10)

        assert len(selected) == 10

    def test_favors_high_fitness(self):
        """Debe favorecer individuos con mayor fitness."""
        random.seed(42)
        pop = create_population_with_fitness(10)
        selector = RouletteSelection()

        selected = selector.select(pop, num_parents=200)
        avg_fitness = sum(ind.fitness for ind in selected) / len(selected)
        pop_avg = sum(ind.fitness for ind in pop) / len(pop)

        assert avg_fitness > pop_avg


class TestRankSelection:
    """Tests para selección por ranking."""

    def test_basic_selection(self):
        """Debe seleccionar la cantidad correcta de padres."""
        pop = create_population_with_fitness(20)
        selector = RankSelection()

        selected = selector.select(pop, num_parents=10)

        assert len(selected) == 10

    def test_favors_better_ranked(self):
        """Debe favorecer individuos mejor rankeados."""
        random.seed(42)
        pop = create_population_with_fitness(10)
        selector = RankSelection()

        selected = selector.select(pop, num_parents=200)
        avg_fitness = sum(ind.fitness for ind in selected) / len(selected)
        pop_avg = sum(ind.fitness for ind in pop) / len(pop)

        assert avg_fitness > pop_avg


class TestElitistSelection:
    """Tests para selección elitista."""

    def test_elite_passes(self):
        """Los mejores deben pasar directamente."""
        pop = create_population_with_fitness(10)
        base = TournamentSelection(tournament_size=2)
        selector = ElitistSelection(elite_count=3, base_method=base)

        selected = selector.select(pop, num_parents=5)

        # Los 3 mejores (fitness más altos) deben estar
        best_fitnesses = sorted((ind.fitness for ind in pop), reverse=True)[:3]
        selected_fitnesses = [ind.fitness for ind in selected[:3]]

        for f in best_fitnesses:
            assert f in selected_fitnesses


class TestSelectionFactory:
    """Tests para el factory de selección."""

    def test_create_tournament(self):
        """Debe crear selección por torneo."""
        selector = create_selection_method("tournament", tournament_size=5)
        assert isinstance(selector, TournamentSelection)

    def test_create_with_elite(self):
        """Debe crear selección con elitismo."""
        selector = create_selection_method(
            "tournament", elite_ratio=0.1, population_size=100
        )
        assert isinstance(selector, ElitistSelection)

    def test_invalid_method(self):
        """Debe fallar con método desconocido."""
        with pytest.raises(ValueError):
            create_selection_method("unknown")


class TestSinglePointCrossover:
    """Tests para cruza de un punto."""

    def test_creates_two_children(self):
        """Debe crear dos hijos."""
        parent1 = Individual.random(num_triangles=10)
        parent2 = Individual.random(num_triangles=10)
        crossover = SinglePointCrossover(probability=1.0)

        child1, child2 = crossover.crossover(parent1, parent2)

        assert len(child1) == 10
        assert len(child2) == 10

    def test_children_differ_from_parents(self):
        """Los hijos deben ser diferentes de los padres."""
        random.seed(42)
        parent1 = Individual.random(num_triangles=10)
        parent2 = Individual.random(num_triangles=10)
        parent1.fitness = 1.0
        parent2.fitness = 2.0
        crossover = SinglePointCrossover(probability=1.0)

        child1, child2 = crossover.crossover(parent1, parent2)

        # Los hijos no deben tener fitness (son nuevos)
        assert child1.fitness is None
        assert child2.fitness is None

    def test_no_crossover_when_probability_zero(self):
        """No debe cruzar si probabilidad es 0."""
        parent1 = Individual.random(num_triangles=5)
        parent2 = Individual.random(num_triangles=5)
        crossover = SinglePointCrossover(probability=0.0)

        child1, child2 = crossover.crossover(parent1, parent2)

        # Deben ser copias de los padres
        assert child1.triangles[0].vertices == parent1.triangles[0].vertices
        assert child2.triangles[0].vertices == parent2.triangles[0].vertices


class TestTwoPointCrossover:
    """Tests para cruza de dos puntos."""

    def test_creates_two_children(self):
        """Debe crear dos hijos."""
        parent1 = Individual.random(num_triangles=10)
        parent2 = Individual.random(num_triangles=10)
        crossover = TwoPointCrossover(probability=1.0)

        child1, child2 = crossover.crossover(parent1, parent2)

        assert len(child1) == 10
        assert len(child2) == 10


class TestUniformCrossover:
    """Tests para cruza uniforme."""

    def test_creates_two_children(self):
        """Debe crear dos hijos."""
        parent1 = Individual.random(num_triangles=10)
        parent2 = Individual.random(num_triangles=10)
        crossover = UniformCrossover(probability=1.0)

        child1, child2 = crossover.crossover(parent1, parent2)

        assert len(child1) == 10
        assert len(child2) == 10


class TestCrossoverFactory:
    """Tests para el factory de cruza."""

    def test_create_single_point(self):
        """Debe crear cruza de un punto."""
        crossover = create_crossover_method("single_point")
        assert isinstance(crossover, SinglePointCrossover)

    def test_create_two_point(self):
        """Debe crear cruza de dos puntos."""
        crossover = create_crossover_method("two_point")
        assert isinstance(crossover, TwoPointCrossover)

    def test_create_uniform(self):
        """Debe crear cruza uniforme."""
        crossover = create_crossover_method("uniform")
        assert isinstance(crossover, UniformCrossover)

    def test_invalid_method(self):
        """Debe fallar con método desconocido."""
        with pytest.raises(ValueError):
            create_crossover_method("unknown")


class TestMutationParams:
    """Tests para parámetros de mutación."""

    def test_default_params(self):
        """Debe crear parámetros por defecto."""
        params = MutationParams()
        assert params.probability == 0.3
        assert params.gene_probability == 0.1

    def test_invalid_probability(self):
        """Debe fallar con probabilidad inválida."""
        with pytest.raises(ValueError):
            MutationParams(probability=1.5)


class TestMutateTriangle:
    """Tests para mutación de triángulos."""

    def test_mutated_triangle_is_valid(self):
        """El triángulo mutado debe ser válido."""
        triangle = Triangle.random()
        params = MutationParams(position_delta=0.2, color_delta=50, alpha_delta=0.2)

        mutated = mutate_triangle(triangle, params)

        # Debe ser un triángulo válido (no lanza excepción)
        assert len(mutated.vertices) == 3
        assert len(mutated.color) == 4

        # Valores dentro de rango
        for x, y in mutated.vertices:
            assert 0 <= x <= 1
            assert 0 <= y <= 1

        r, g, b, a = mutated.color
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255
        assert 0 <= a <= 1


class TestMutateIndividual:
    """Tests para mutación de individuos."""

    def test_mutated_individual_same_length(self):
        """El individuo mutado debe tener el mismo número de triángulos."""
        individual = Individual.random(num_triangles=10)
        params = MutationParams(probability=1.0, gene_probability=1.0)

        mutated = mutate_individual(individual, params)

        assert len(mutated) == len(individual)

    def test_no_mutation_when_probability_zero(self):
        """No debe mutar si probabilidad es 0."""
        individual = Individual.random(num_triangles=5)
        params = MutationParams(probability=0.0)

        mutated = mutate_individual(individual, params)

        # Debe ser una copia exacta
        for i in range(len(individual)):
            assert mutated[i].vertices == individual[i].vertices
            assert mutated[i].color == individual[i].color

    def test_fitness_invalidated(self):
        """El fitness debe invalidarse después de mutar."""
        individual = Individual.random(num_triangles=5)
        individual.fitness = 100.0
        params = MutationParams(probability=1.0, gene_probability=1.0)

        mutated = mutate_individual(individual, params)

        assert mutated.fitness is None


class TestMutator:
    """Tests para la clase Mutator."""

    def test_mutate_individual(self):
        """Debe mutar un individuo."""
        mutator = Mutator(MutationParams(probability=1.0))
        individual = Individual.random(num_triangles=5)

        mutated = mutator.mutate(individual)

        assert isinstance(mutated, Individual)
        assert len(mutated) == len(individual)

    def test_mutate_population(self):
        """Debe mutar toda la población."""
        mutator = Mutator()
        population = [Individual.random(num_triangles=5) for _ in range(10)]

        mutated_pop = mutator.mutate_population(population)

        assert len(mutated_pop) == 10
        assert all(isinstance(ind, Individual) for ind in mutated_pop)
