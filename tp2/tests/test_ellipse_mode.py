"""Tests específicos para el modo ellipse."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.genetic.engine import EvolutionConfig, create_engine
from src.genetic.individual import Ellipse, Individual
from src.rendering.ellipse_canvas import EllipseCanvas
from src.rendering.factory import create_renderer
from src.rendering.gpu_ellipse_canvas import EllipseGPUCanvas, MODERNGL_AVAILABLE
from src.utils.export import export_shapes_json, load_shapes_json
from visualize import EvolutionVisualizer


ROOT = Path(__file__).resolve().parents[1]


class TestEllipse:
    def test_create_valid_ellipse(self):
        ellipse = Ellipse(
            center=(0.5, 0.5),
            radii=(0.2, 0.1),
            angle=0.3,
            color=(255, 0, 0, 0.5),
        )

        assert ellipse.center == (0.5, 0.5)
        assert ellipse.radii == (0.2, 0.1)
        assert ellipse.color == (255, 0, 0, 0.5)

    def test_random_ellipse(self):
        ellipse = Ellipse.random(alpha_min=0.2, alpha_max=0.7)

        assert 0 <= ellipse.center[0] <= 1
        assert 0 <= ellipse.center[1] <= 1
        assert 0 < ellipse.radii[0] <= 1
        assert 0 < ellipse.radii[1] <= 1
        assert -np.pi <= ellipse.angle <= np.pi
        assert 0.2 <= ellipse.color[3] <= 0.7

    def test_serialization_roundtrip(self):
        original = Ellipse(
            center=(0.25, 0.75),
            radii=(0.12, 0.2),
            angle=1.2,
            color=(10, 20, 30, 0.4),
        )

        restored = Ellipse.from_dict(original.to_dict())

        assert restored.center == original.center
        assert restored.radii == original.radii
        assert restored.angle == pytest.approx(original.angle)
        assert restored.color == original.color


class TestEllipseRendering:
    def test_render_single_ellipse_cpu(self):
        canvas = EllipseCanvas(width=100, height=100)
        ellipse = Ellipse(
            center=(0.5, 0.5),
            radii=(0.25, 0.15),
            angle=0.0,
            color=(255, 0, 0, 1.0),
        )
        individual = Individual(triangles=[ellipse], shape_type="ellipse")

        arr = canvas.render_to_array(individual)

        center = arr[50, 50]
        assert center[0] == 255
        assert center[1] == 0
        assert center[2] == 0

    def test_renderer_factory_returns_ellipse_cpu(self):
        renderer = create_renderer(64, 64, backend="cpu", shape_type="ellipse")
        assert isinstance(renderer, EllipseCanvas)

    @pytest.mark.skipif(not MODERNGL_AVAILABLE, reason="moderngl no instalado")
    def test_renderer_factory_returns_ellipse_gpu(self):
        renderer = create_renderer(64, 64, backend="gpu", shape_type="ellipse")
        assert isinstance(renderer, EllipseGPUCanvas)


class TestEllipseModeIO:
    def test_export_and_load_shapes_json(self, tmp_path: Path):
        individual = Individual.random(num_triangles=4, shape_type="ellipse")
        individual.fitness = 0.77
        path = tmp_path / "shapes.json"

        export_shapes_json(individual, path)
        loaded = load_shapes_json(path)

        assert loaded.shape_type == "ellipse"
        assert len(loaded) == len(individual)
        assert loaded.fitness == individual.fitness

    def test_reconstruct_script_with_shapes(self, tmp_path: Path):
        target = tmp_path / "target.png"
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(target)
        individual = Individual.random(num_triangles=3, shape_type="ellipse")
        shapes_path = tmp_path / "shapes.json"
        output_path = tmp_path / "reconstructed.png"
        export_shapes_json(individual, shapes_path)

        subprocess.run(
            [
                sys.executable,
                "reconstruct.py",
                str(shapes_path),
                "--output",
                str(output_path),
                "--width",
                "64",
                "--height",
                "64",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        assert output_path.exists()


class TestEllipseModeEngine:
    def test_engine_runs_with_ellipses(self):
        target = Image.new("RGB", (32, 32), color=(255, 255, 255))
        config = EvolutionConfig(
            population_size=6,
            num_triangles=4,
            shape_type="ellipse",
            max_generations=2,
        )

        engine = create_engine(target, config)
        result = engine.run()

        assert result.best_individual.shape_type == "ellipse"
        assert 0 < result.best_fitness <= 1
        assert result.generations == 2


class TestEllipseModeCli:
    def test_main_cli_smoke_ellipse(self, tmp_path: Path):
        input_path = tmp_path / "target.png"
        output_dir = tmp_path / "run"
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(input_path)

        subprocess.run(
            [
                sys.executable,
                "main.py",
                "--image",
                str(input_path),
                "--triangles",
                "4",
                "--population",
                "4",
                "--generations",
                "1",
                "--shape",
                "ellipse",
                "--renderer",
                "cpu",
                "--output",
                str(output_dir),
                "--quiet",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        assert (output_dir / "result.png").exists()
        assert (output_dir / "shapes.json").exists()
        assert not (output_dir / "triangles.json").exists()

    def test_visualizer_loads_shapes_output(self, tmp_path: Path):
        output_dir = tmp_path / "viz"
        output_dir.mkdir()
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(
            output_dir / "result.png"
        )
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(
            output_dir / "gen_00001.png"
        )

        individual = Individual.random(num_triangles=3, shape_type="ellipse")
        export_shapes_json(individual, output_dir / "shapes.json")

        visualizer = EvolutionVisualizer(output_dir)

        assert visualizer.shapes_data is not None
        assert visualizer.shapes_data["shape_type"] == "ellipse"
        assert len(visualizer.frames) >= 1
