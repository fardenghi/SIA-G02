"""Tests para configuración y CLI."""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.utils.config import (
    Config,
    SelectionConfig,
    CrossoverConfig,
    MutationConfig,
    OutputConfig,
    load_config,
)


class TestSelectionConfig:
    """Tests para SelectionConfig."""

    def test_default_values(self):
        """Debe tener valores por defecto."""
        config = SelectionConfig()

        assert config.method == "tournament"
        assert config.tournament_size == 3
        assert config.threshold == 0.75
        assert config.boltzmann_t0 == 100.0
        assert config.boltzmann_tc == 1.0
        assert config.boltzmann_k == 0.005


class TestCrossoverConfig:
    """Tests para CrossoverConfig."""

    def test_default_values(self):
        """Debe tener valores por defecto."""
        config = CrossoverConfig()

        assert config.method == "single_point"
        assert config.probability == 0.8


class TestMutationConfig:
    """Tests para MutationConfig."""

    def test_default_values(self):
        """Debe tener valores por defecto."""
        config = MutationConfig()

        assert config.probability == 0.3
        assert config.gene_probability == 0.1

    def test_to_params(self):
        """Debe convertir a MutationParams."""
        config = MutationConfig(probability=0.5, color_delta=50)
        params = config.to_params()

        assert params.probability == 0.5
        assert params.color_delta == 50


class TestConfig:
    """Tests para Config."""

    def test_default_config(self):
        """Debe crear configuración con valores por defecto."""
        config = Config()

        assert config.num_triangles == 50
        assert config.population_size == 100
        assert config.max_generations == 5000

    def test_from_dict(self):
        """Debe crear desde diccionario."""
        data = {
            "num_triangles": 30,
            "population_size": 50,
            "selection": {"method": "roulette"},
            "crossover": {"method": "uniform", "probability": 0.9},
        }

        config = Config.from_dict(data)

        assert config.num_triangles == 30
        assert config.population_size == 50
        assert config.selection.method == "roulette"
        assert config.crossover.method == "uniform"
        assert config.crossover.probability == 0.9

    def test_from_dict_legacy_format(self):
        """Debe soportar formato legacy del YAML."""
        data = {
            "image": {"target_path": "/path/to/image.png"},
            "genotype": {"num_triangles": 40, "alpha_min": 0.2},
            "genetic": {
                "population_size": 80,
                "max_generations": 1000,
                "error_threshold": 3500,
            },
        }

        config = Config.from_dict(data)

        assert config.target_path == "/path/to/image.png"
        assert config.num_triangles == 40
        assert config.alpha_min == 0.2
        assert config.population_size == 80
        assert config.max_generations == 1000
        assert config.fitness_threshold == pytest.approx(1.0 / 3501.0)

    def test_from_yaml(self):
        """Debe cargar desde archivo YAML."""
        yaml_content = """
        num_triangles: 25
        population_size: 60
        selection:
          method: rank
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = Config.from_yaml(path)

            assert config.num_triangles == 25
            assert config.population_size == 60
            assert config.selection.method == "rank"
        finally:
            Path(path).unlink()

    def test_from_yaml_not_found(self):
        """Debe fallar si el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path.yaml")

    def test_merge_cli_args(self):
        """Debe combinar con argumentos CLI."""
        config = Config(num_triangles=50, population_size=100)

        merged = config.merge_cli_args(
            triangles=75, generations=2000, output="custom_output"
        )

        assert merged.num_triangles == 75
        assert merged.max_generations == 2000
        assert merged.output.directory == "custom_output"
        # El original no debe cambiar
        assert config.num_triangles == 50

    def test_to_evolution_config(self):
        """Debe convertir a EvolutionConfig."""
        config = Config(population_size=80, num_triangles=40, max_generations=3000)

        evo_config = config.to_evolution_config()

        assert evo_config.population_size == 80
        assert evo_config.num_triangles == 40
        assert evo_config.max_generations == 3000

    def test_validate_missing_image(self):
        """Debe detectar imagen faltante."""
        config = Config(target_path=None)

        errors = config.validate()

        assert any("imagen objetivo" in e for e in errors)

    def test_validate_invalid_triangles(self):
        """Debe detectar triángulos inválidos."""
        config = Config(target_path="/dummy/path.png", num_triangles=0)

        errors = config.validate()

        assert any("num_triangles" in e for e in errors)

    def test_validate_alpha_range(self):
        """Debe detectar rango de alpha inválido."""
        config = Config(target_path="/dummy/path.png", alpha_min=0.8, alpha_max=0.2)

        errors = config.validate()

        assert any("alpha_min" in e for e in errors)

    def test_validate_fitness_threshold_range(self):
        """Debe detectar fitness_threshold fuera de rango."""
        config = Config(target_path="/dummy/path.png", fitness_threshold=2.0)

        errors = config.validate()

        assert any("fitness_threshold" in e for e in errors)

    def test_validate_invalid_fitness_method(self):
        """Debe detectar métodos de fitness removidos o inválidos."""
        config = Config(target_path="/dummy/path.png")
        config.fitness.method = "unknown_method"

        errors = config.validate()

        assert any("fitness.method" in e for e in errors)


class TestLoadConfig:
    """Tests para la función load_config."""

    def test_load_without_file(self):
        """Debe cargar configuración por defecto sin archivo."""
        config = load_config()

        assert isinstance(config, Config)
        assert config.num_triangles == 50

    def test_load_with_cli_override(self):
        """Debe aplicar overrides de CLI."""
        config = load_config(triangles=100, population=200)

        assert config.num_triangles == 100
        assert config.population_size == 200

    def test_load_from_file_with_override(self):
        """Debe combinar archivo y CLI."""
        yaml_content = """
        num_triangles: 30
        population_size: 50
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            config = load_config(path, triangles=60)

            # triangles viene del CLI (override)
            assert config.num_triangles == 60
            # population_size viene del archivo
            assert config.population_size == 50
        finally:
            Path(path).unlink()
