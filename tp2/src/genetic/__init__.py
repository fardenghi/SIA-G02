"""Motor de algoritmos genéticos."""

from src.genetic.individual import Triangle, Individual
from src.genetic.population import Population
from src.genetic.selection import (
    SelectionMethod,
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
    CrossoverMethod,
    SinglePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    create_crossover_method,
)
from src.genetic.mutation import (
    MutationParams,
    Mutator,
    mutate_triangle,
    mutate_individual,
)
from src.genetic.engine import (
    GeneticEngine,
    EvolutionConfig,
    EvolutionResult,
    create_engine,
)

__all__ = [
    # Estructuras de datos
    "Triangle",
    "Individual",
    "Population",
    # Selección
    "SelectionMethod",
    "EliteSelection",
    "TournamentSelection",
    "ProbabilisticTournamentSelection",
    "RouletteSelection",
    "UniversalSelection",
    "BoltzmannSelection",
    "RankSelection",
    "create_selection_method",
    # Cruza
    "CrossoverMethod",
    "SinglePointCrossover",
    "TwoPointCrossover",
    "UniformCrossover",
    "create_crossover_method",
    # Mutación
    "MutationParams",
    "Mutator",
    "mutate_triangle",
    "mutate_individual",
    # Motor
    "GeneticEngine",
    "EvolutionConfig",
    "EvolutionResult",
    "create_engine",
]
