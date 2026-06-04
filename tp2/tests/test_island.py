"""Tests para el modelo de islas (IMGA)."""

import random
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.genetic.engine import (
    EvolutionConfig,
    EvolutionResult,
    StepResult,
    create_engine,
)
from src.genetic.individual import Individual
from src.genetic.island import (
    IslandEngine,
    build_ring_topology,
    migrate_sequential,
)
from src.genetic.population import Population
from src.utils.config import Config, IslandConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_image(size=(20, 20), color=(128, 128, 128)):
    return Image.new("RGB", size, color=color)


def _small_config(**overrides):
    """Config mínima para tests rápidos."""
    defaults = {
        "target_path": "/tmp/test.png",
        "population_size": 10,
        "num_triangles": 5,
        "max_generations": 5,
        "fitness_threshold": None,
    }
    defaults.update(overrides)
    return Config(**defaults)


def _create_population_with_fitness(n, num_triangles=5):
    """Crea una población con fitness asignado."""
    individuals = [Individual.random(num_triangles=num_triangles) for _ in range(n)]
    for i, ind in enumerate(individuals):
        ind.fitness = float((i + 1) * 100)
    return Population(individuals=individuals)


# ---------------------------------------------------------------------------
# TestIslandConfig
# ---------------------------------------------------------------------------

class TestIslandConfig:
    def test_default_disabled(self):
        config = IslandConfig()
        assert config.enabled is False
        assert config.num_islands == 4
        assert config.migration_size == 5
        assert config.migration_interval == 10
        assert config.topology == "ring"
        assert config.parallel is True

    def test_from_dict_with_island_section(self):
        data = {
            "target_path": "/tmp/test.png",
            "island": {
                "enabled": True,
                "num_islands": 8,
                "migration_size": 10,
                "migration_interval": 5,
                "topology": "ring",
                "parallel": False,
            },
        }
        config = Config.from_dict(data)
        assert config.island.enabled is True
        assert config.island.num_islands == 8
        assert config.island.migration_size == 10
        assert config.island.migration_interval == 5
        assert config.island.parallel is False

    def test_from_dict_without_island_section(self):
        data = {"target_path": "/tmp/test.png"}
        config = Config.from_dict(data)
        assert config.island.enabled is False

    def test_cli_override_islands(self):
        config = Config(target_path="/tmp/test.png")
        merged = config.merge_cli_args(islands=4)
        assert merged.island.enabled is True
        assert merged.island.num_islands == 4

    def test_cli_override_islands_one_disables(self):
        config = Config(target_path="/tmp/test.png")
        merged = config.merge_cli_args(islands=1)
        assert merged.island.enabled is False
        assert merged.island.num_islands == 1

    def test_cli_override_migration_params(self):
        config = Config(target_path="/tmp/test.png")
        merged = config.merge_cli_args(migration_size=20, migration_interval=3)
        assert merged.island.migration_size == 20
        assert merged.island.migration_interval == 3

    def test_validate_migration_size_too_large(self):
        config = Config(
            target_path="/tmp/test.png",
            population_size=10,
            island=IslandConfig(enabled=True, num_islands=2, migration_size=6),
        )
        errors = config.validate()
        assert any("migration_size" in e for e in errors)

    def test_validate_num_islands_too_small(self):
        config = Config(
            target_path="/tmp/test.png",
            island=IslandConfig(enabled=True, num_islands=1),
        )
        errors = config.validate()
        assert any("num_islands" in e for e in errors)


# ---------------------------------------------------------------------------
# TestRingTopology
# ---------------------------------------------------------------------------

class TestRingTopology:
    def test_ring_2_islands(self):
        topo = build_ring_topology(2)
        assert topo == [(0, 1), (1, 0)]

    def test_ring_3_islands(self):
        topo = build_ring_topology(3)
        assert topo == [(0, 1), (1, 2), (2, 0)]

    def test_ring_4_islands(self):
        topo = build_ring_topology(4)
        assert topo == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_ring_1_island(self):
        topo = build_ring_topology(1)
        assert topo == [(0, 0)]


# ---------------------------------------------------------------------------
# TestMigration
# ---------------------------------------------------------------------------

class TestMigration:
    def _make_engines_with_populations(self, k=2, pop_size=10, num_triangles=5):
        """Crea K engines con poblaciones evaluadas."""
        target = _small_image()
        engines = []
        for _ in range(k):
            config = EvolutionConfig(
                population_size=pop_size,
                num_triangles=num_triangles,
                max_generations=5,
            )
            eng = create_engine(target, config)
            eng.initialize_and_evaluate()
            engines.append(eng)
        return engines

    def test_migration_preserves_population_size(self):
        random.seed(42)
        engines = self._make_engines_with_populations(k=2)
        topology = build_ring_topology(2)
        pop_sizes_before = [len(e.population) for e in engines]

        migrate_sequential(engines, topology, migration_size=2, elite_count=0)

        pop_sizes_after = [len(e.population) for e in engines]
        assert pop_sizes_before == pop_sizes_after

    def test_migrant_fitness_invalidated(self):
        random.seed(42)
        engines = self._make_engines_with_populations(k=2)
        topology = build_ring_topology(2)

        # Antes de migrar, todos tienen fitness
        for eng in engines:
            assert all(ind.fitness is not None for ind in eng.population.individuals)

        migrate_sequential(engines, topology, migration_size=2, elite_count=0)

        # Después de migrar, algunos tienen fitness=None (los migrantes)
        for eng in engines:
            none_count = sum(
                1 for ind in eng.population.individuals if ind.fitness is None
            )
            # Deberían haber 2*M=4 migrantes con fitness invalidado
            # (pueden ser menos si hay overlap en indices)
            assert none_count > 0

    def test_migration_selects_best_and_worst(self):
        random.seed(42)
        engines = self._make_engines_with_populations(k=2)
        topology = [(0, 1)]  # solo 0 -> 1

        # Recordar fitness del source
        src_pop = engines[0].population
        sorted_src = sorted(
            src_pop.individuals, key=lambda i: i.fitness or 0.0, reverse=True
        )
        best_fitness = sorted_src[0].fitness
        worst_fitness = sorted_src[-1].fitness

        migrate_sequential(engines, topology, migration_size=1, elite_count=0)

        # El destino debería tener individuos con fitness=None
        # (migrantes invalidados)
        dst_pop = engines[1].population
        none_inds = [ind for ind in dst_pop.individuals if ind.fitness is None]
        assert len(none_inds) == 2  # 1 best + 1 worst

    def test_migration_respects_elite_count(self):
        """Migrantes no deberían reemplazar a los elite del destino."""
        random.seed(42)
        engines = self._make_engines_with_populations(k=2, pop_size=10)
        topology = [(0, 1)]

        dst_pop = engines[1].population
        sorted_dst = sorted(
            range(len(dst_pop)),
            key=lambda i: dst_pop.individuals[i].fitness or 0.0,
            reverse=True,
        )
        top_2_fitnesses = {
            dst_pop.individuals[sorted_dst[0]].fitness,
            dst_pop.individuals[sorted_dst[1]].fitness,
        }

        migrate_sequential(engines, topology, migration_size=2, elite_count=2)

        # Los top-2 del destino no deberían haber sido reemplazados
        remaining_fitnesses = {
            ind.fitness for ind in dst_pop.individuals if ind.fitness is not None
        }
        # Al menos los top 2 originales deben seguir presentes
        assert top_2_fitnesses.issubset(remaining_fitnesses)


# ---------------------------------------------------------------------------
# TestEngineStepRefactor
# ---------------------------------------------------------------------------

class TestEngineStepRefactor:
    @pytest.fixture
    def engine(self):
        target = _small_image()
        config = EvolutionConfig(
            population_size=10, num_triangles=5, max_generations=5
        )
        return create_engine(target, config)

    def test_step_returns_step_result(self, engine):
        engine.initialize_and_evaluate()
        result = engine.step()
        assert isinstance(result, StepResult)
        assert result.generation == 1
        assert isinstance(result.best_fitness, float)
        assert isinstance(result.stats, dict)
        assert isinstance(result.improved, bool)
        assert isinstance(result.stopped_early, bool)

    def test_initialize_and_evaluate_sets_state(self, engine):
        engine.initialize_and_evaluate()
        assert engine.population is not None
        assert engine.best_ever is not None
        assert engine.best_fitness_ever is not None
        assert len(engine.history) == 1  # gen 0

    def test_run_produces_evolution_result(self, engine):
        result = engine.run()
        assert isinstance(result, EvolutionResult)
        assert result.generations == 5
        assert result.best_fitness > 0

    def test_step_increments_generation(self, engine):
        engine.initialize_and_evaluate()
        for expected_gen in range(1, 4):
            result = engine.step()
            assert result.generation == expected_gen

    def test_best_ever_property(self, engine):
        engine.initialize_and_evaluate()
        assert isinstance(engine.best_ever, Individual)
        assert engine.best_fitness_ever == engine.best_ever.fitness


# ---------------------------------------------------------------------------
# TestIslandEngineSequential
# ---------------------------------------------------------------------------

class TestIslandEngineSequential:
    def _make_config(self, **island_overrides):
        island_defaults = {
            "enabled": True,
            "num_islands": 2,
            "migration_size": 2,
            "migration_interval": 2,
            "topology": "ring",
            "parallel": False,
        }
        island_defaults.update(island_overrides)
        return Config(
            target_path="/tmp/test.png",
            population_size=10,
            num_triangles=5,
            max_generations=5,
            island=IslandConfig(**island_defaults),
        )

    def test_run_returns_evolution_result(self):
        target = _small_image()
        config = self._make_config()
        engine = IslandEngine(target_image=target, config=config)
        result = engine.run()
        assert isinstance(result, EvolutionResult)
        assert result.best_fitness > 0
        assert result.generations > 0

    def test_callbacks_fire(self):
        target = _small_image()
        config = self._make_config()
        engine = IslandEngine(target_image=target, config=config)

        gen_calls = []
        imp_calls = []
        engine.on_generation(lambda g, p, s: gen_calls.append(g))
        engine.on_improvement(lambda g, i, f: imp_calls.append(g))

        engine.run()

        # gen 0 + 5 generaciones
        assert len(gen_calls) == 6
        assert gen_calls[0] == 0
        assert gen_calls[-1] == 5

    def test_global_best_improves_or_stays(self):
        target = _small_image()
        config = self._make_config()
        engine = IslandEngine(target_image=target, config=config)

        fitness_history = []
        engine.on_generation(lambda g, p, s: fitness_history.append(s["best_fitness"]))

        engine.run()

        # El global best nunca debería empeorar
        for i in range(1, len(fitness_history)):
            assert fitness_history[i] >= fitness_history[i - 1]

    def test_width_height_properties(self):
        target = _small_image(size=(30, 20))
        config = self._make_config()
        engine = IslandEngine(target_image=target, config=config)
        assert engine.width == 30
        assert engine.height == 20


# ---------------------------------------------------------------------------
# TestIslandEngineParallel
# ---------------------------------------------------------------------------

class TestIslandEngineParallel:
    def test_parallel_run_returns_result(self):
        """Smoke test con multiprocessing real."""
        target = _small_image()
        config = Config(
            target_path="/tmp/test.png",
            population_size=10,
            num_triangles=5,
            max_generations=3,
            island=IslandConfig(
                enabled=True,
                num_islands=2,
                migration_size=1,
                migration_interval=2,
                topology="ring",
                parallel=True,
            ),
        )
        engine = IslandEngine(target_image=target, config=config)
        result = engine.run()
        assert isinstance(result, EvolutionResult)
        assert result.best_fitness > 0
        assert result.generations > 0

    def test_parallel_callbacks_fire(self):
        target = _small_image()
        config = Config(
            target_path="/tmp/test.png",
            population_size=10,
            num_triangles=5,
            max_generations=3,
            island=IslandConfig(
                enabled=True,
                num_islands=2,
                migration_size=1,
                migration_interval=2,
                parallel=True,
            ),
        )
        engine = IslandEngine(target_image=target, config=config)

        gen_calls = []
        engine.on_generation(lambda g, p, s: gen_calls.append(g))

        result = engine.run()

        assert len(gen_calls) == 4  # gen 0 + 3 generations
