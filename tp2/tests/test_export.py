"""Tests para exportación de resultados."""

import pytest
import json
import tempfile
from pathlib import Path
from PIL import Image

from src.genetic.individual import Individual, Triangle
from src.utils.export import (
    save_result_image,
    export_triangles_json,
    load_triangles_json,
    save_fitness_plot,
    save_metrics_csv,
)


class TestSaveResultImage:
    """Tests para save_result_image."""

    def test_save_image(self):
        """Debe guardar imagen correctamente."""
        individual = Individual.random(num_triangles=5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            save_result_image(individual, 100, 100, path)

            assert Path(path).exists()
            img = Image.open(path)
            assert img.size == (100, 100)
        finally:
            Path(path).unlink(missing_ok=True)


class TestExportTriangles:
    """Tests para exportación de triángulos."""

    def test_export_and_load(self):
        """Debe exportar y cargar triángulos correctamente."""
        original = Individual.random(num_triangles=10)
        original.fitness = 123.45

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            export_triangles_json(original, path)

            assert Path(path).exists()

            # Verificar contenido JSON
            with open(path) as f:
                data = json.load(f)

            assert data["num_triangles"] == 10
            assert data["fitness"] == 123.45
            assert len(data["triangles"]) == 10

            # Cargar de vuelta
            loaded = load_triangles_json(path)

            assert len(loaded) == len(original)
            assert loaded.fitness == original.fitness

            for i in range(len(original)):
                assert loaded[i].vertices == original[i].vertices
                assert loaded[i].color == original[i].color
        finally:
            Path(path).unlink(missing_ok=True)


class TestSaveFitnessPlot:
    """Tests para save_fitness_plot."""

    def test_save_plot(self):
        """Debe guardar gráfico de fitness."""
        history = [
            {"generation": 0, "best_fitness": 10000, "avg_fitness": 15000},
            {"generation": 1, "best_fitness": 9000, "avg_fitness": 12000},
            {"generation": 2, "best_fitness": 8000, "avg_fitness": 10000},
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            save_fitness_plot(history, path)

            assert Path(path).exists()
            img = Image.open(path)
            assert img.size[0] > 0
            assert img.size[1] > 0
        finally:
            Path(path).unlink(missing_ok=True)


class TestSaveMetricsCsv:
    """Tests para save_metrics_csv."""

    def test_save_csv(self):
        """Debe guardar CSV correctamente."""
        history = [
            {"generation": 0, "best_fitness": 10000, "avg_fitness": 15000},
            {"generation": 1, "best_fitness": 9000, "avg_fitness": 12000},
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name

        try:
            save_metrics_csv(history, path)

            assert Path(path).exists()

            with open(path) as f:
                content = f.read()

            assert "generation" in content
            assert "best_fitness" in content
            assert "10000" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_history(self):
        """Debe manejar historial vacío."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name

        try:
            save_metrics_csv([], path)
            # No debe lanzar excepción
        finally:
            Path(path).unlink(missing_ok=True)
