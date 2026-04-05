"""Tests para los operadores genéticos (selección, cruza, mutación)."""

import pytest
import random
from src.genetic.individual import Individual, Triangle
from src.genetic.selection import (
    EliteSelection,
    TournamentSelection,
    ProbabilisticTournamentSelection,
    RouletteSelection,
    UniversalSelection,
    BoltzmannSelection,
    RankSelection,
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


class TestEliteSelection:
    """Tests para selección élite."""

    def test_basic_selection(self):
        """Debe seleccionar la cantidad correcta de padres."""
        pop = create_population_with_fitness(10)
        selector = EliteSelection()

        selected = selector.select(pop, num_parents=10)

        assert len(selected) == 10

    def test_best_individual_always_included(self):
        """El mejor individuo siempre debe aparecer en la selección."""
        pop = create_population_with_fitness(10)
        best_fitness = max(ind.fitness for ind in pop)
        selector = EliteSelection()

        selected = selector.select(pop, num_parents=10)

        assert any(ind.fitness == best_fitness for ind in selected)

    def test_deterministic(self):
        """Dos llamadas con la misma población deben dar el mismo resultado."""
        pop = create_population_with_fitness(10)
        selector = EliteSelection()

        selected1 = selector.select(pop, num_parents=10)
        selected2 = selector.select(pop, num_parents=10)

        assert [ind.fitness for ind in selected1] == [ind.fitness for ind in selected2]

    def test_favors_high_fitness(self):
        """El promedio de fitness seleccionado debe ser mayor al de la población."""
        pop = create_population_with_fitness(10)
        selector = EliteSelection()

        selected = selector.select(pop, num_parents=10)
        avg_selected = sum(ind.fitness for ind in selected) / len(selected)
        avg_pop = sum(ind.fitness for ind in pop) / len(pop)

        assert avg_selected > avg_pop


class TestProbabilisticTournamentSelection:
    """Tests para selección por torneo probabilístico."""

    def test_basic_selection(self):
        """Debe seleccionar la cantidad correcta de padres."""
        pop = create_population_with_fitness(20)
        selector = ProbabilisticTournamentSelection(threshold=0.75)

        selected = selector.select(pop, num_parents=10)

        assert len(selected) == 10
        assert all(isinstance(ind, Individual) for ind in selected)

    def test_favors_high_fitness(self):
        """Con threshold alto debe favorecer a los más aptos."""
        random.seed(42)
        pop = create_population_with_fitness(10)
        selector = ProbabilisticTournamentSelection(threshold=0.9)

        selected = selector.select(pop, num_parents=200)
        avg_fitness = sum(ind.fitness for ind in selected) / len(selected)
        pop_avg = sum(ind.fitness for ind in pop) / len(pop)

        assert avg_fitness > pop_avg

    def test_invalid_threshold(self):
        """Debe fallar con threshold fuera de [0.5, 1]."""
        with pytest.raises(ValueError):
            ProbabilisticTournamentSelection(threshold=0.3)

        with pytest.raises(ValueError):
            ProbabilisticTournamentSelection(threshold=1.1)


class TestUniversalSelection:
    """Tests para selección universal estocástica (SUS)."""

    def test_basic_selection(self):
        """Debe seleccionar la cantidad correcta de padres."""
        pop = create_population_with_fitness(20)
        selector = UniversalSelection()

        selected = selector.select(pop, num_parents=10)

        assert len(selected) == 10
        assert all(isinstance(ind, Individual) for ind in selected)

    def test_favors_high_fitness(self):
        """Debe favorecer individuos con mayor fitness."""
        random.seed(42)
        pop = create_population_with_fitness(10)
        selector = UniversalSelection()

        selected = selector.select(pop, num_parents=200)
        avg_fitness = sum(ind.fitness for ind in selected) / len(selected)
        pop_avg = sum(ind.fitness for ind in pop) / len(pop)

        assert avg_fitness > pop_avg

    def test_empty_population(self):
        """Debe fallar con población vacía."""
        selector = UniversalSelection()
        with pytest.raises(ValueError):
            selector.select([], num_parents=5)


class TestBoltzmannSelection:
    """Tests para selección entrópica de Boltzmann."""

    def test_basic_selection(self):
        """Debe seleccionar la cantidad correcta de padres."""
        pop = create_population_with_fitness(20)
        selector = BoltzmannSelection(t0=100.0, tc=1.0, k=0.005)

        selected = selector.select(pop, num_parents=10, generation=0)

        assert len(selected) == 10
        assert all(isinstance(ind, Individual) for ind in selected)

    def test_favors_high_fitness_at_low_temperature(self):
        """A temperatura baja debe favorecer fuertemente a los más aptos."""
        random.seed(42)
        pop = create_population_with_fitness(10)
        # Temperatura baja: generación muy avanzada
        selector = BoltzmannSelection(t0=100.0, tc=1.0, k=1.0)

        selected = selector.select(pop, num_parents=200, generation=100)
        avg_fitness = sum(ind.fitness for ind in selected) / len(selected)
        pop_avg = sum(ind.fitness for ind in pop) / len(pop)

        assert avg_fitness > pop_avg

    def test_temperature_decreases_with_generation(self):
        """T(t) debe decrecer con generaciones crecientes."""
        selector = BoltzmannSelection(t0=100.0, tc=1.0, k=0.1)

        t0 = selector._temperature(0)
        t10 = selector._temperature(10)
        t100 = selector._temperature(100)

        assert t0 > t10 > t100
        assert abs(t0 - 100.0) < 1e-9  # T(0) = T0

    def test_invalid_params(self):
        """Debe fallar con parámetros inválidos."""
        with pytest.raises(ValueError):
            BoltzmannSelection(t0=-1.0)
        with pytest.raises(ValueError):
            BoltzmannSelection(k=0.0)


class TestSelectionFactory:
    """Tests para el factory de selección."""

    def test_create_elite(self):
        """Debe crear selección élite."""
        selector = create_selection_method("elite")
        assert isinstance(selector, EliteSelection)

    def test_create_tournament(self):
        """Debe crear selección por torneo."""
        selector = create_selection_method("tournament", tournament_size=5)
        assert isinstance(selector, TournamentSelection)

    def test_create_probabilistic_tournament(self):
        """Debe crear selección por torneo probabilístico."""
        selector = create_selection_method("probabilistic_tournament", threshold=0.8)
        assert isinstance(selector, ProbabilisticTournamentSelection)

    def test_create_universal(self):
        """Debe crear selección universal."""
        selector = create_selection_method("universal")
        assert isinstance(selector, UniversalSelection)

    def test_create_boltzmann(self):
        """Debe crear selección de Boltzmann."""
        selector = create_selection_method("boltzmann")
        assert isinstance(selector, BoltzmannSelection)

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
