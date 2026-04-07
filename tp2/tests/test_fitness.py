"""Tests para cálculo de error (MSE) y fitness."""

import pytest
import numpy as np
from PIL import Image

from src.genetic.individual import Triangle, Individual
from src.fitness.mse import (
    calculate_mse,
    calculate_normalized_mse,
    error_to_fitness,
    calculate_fitness,
    compute_fitness,
    FitnessEvaluator,
)


class TestMSE:
    """Tests para las funciones de cálculo de MSE."""

    def test_identical_images(self):
        """MSE debe ser 0 para imágenes idénticas."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        mse = calculate_mse(img, img.copy())

        assert mse == 0.0

    def test_completely_different(self):
        """MSE debe ser máximo para imágenes completamente opuestas."""
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        white = np.full((10, 10, 3), 255, dtype=np.uint8)

        mse = calculate_mse(black, white)

        # MSE máximo = 255² = 65025
        assert mse == 65025.0

    def test_partial_difference(self):
        """MSE debe estar entre 0 y máximo para diferencias parciales."""
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 150, dtype=np.uint8)

        mse = calculate_mse(img1, img2)

        # Diferencia de 50, MSE = 50² = 2500
        assert mse == 2500.0

    def test_dimension_mismatch(self):
        """Debe fallar si las dimensiones no coinciden."""
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.zeros((20, 20, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="dimensiones no coinciden"):
            calculate_mse(img1, img2)

    def test_normalized_mse_range(self):
        """MSE normalizado debe estar en [0, 1]."""
        img1 = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)

        nmse = calculate_normalized_mse(img1, img2)

        assert 0 <= nmse <= 1

    def test_normalized_mse_identical(self):
        """MSE normalizado debe ser 0 para imágenes idénticas."""
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)

        nmse = calculate_normalized_mse(img, img.copy())

        assert nmse == 0.0

    def test_normalized_mse_opposite(self):
        """MSE normalizado debe ser 1 para imágenes opuestas."""
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        white = np.full((10, 10, 3), 255, dtype=np.uint8)

        nmse = calculate_normalized_mse(black, white)

        assert nmse == 1.0


class TestFitnessTransform:
    """Tests para la transformación de error a fitness."""

    def test_error_zero_is_max_fitness(self):
        """Error 0 debe mapear a fitness máximo (1.0)."""
        assert error_to_fitness(0.0) == 1.0

    def test_fitness_decreases_with_error(self):
        """A mayor error, menor fitness."""
        low_error = error_to_fitness(100.0)
        high_error = error_to_fitness(1000.0)
        assert low_error > high_error

    def test_calculate_fitness_range(self):
        """El fitness calculado debe estar en (0, 1]."""
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        white = np.full((10, 10, 3), 255, dtype=np.uint8)

        fitness = calculate_fitness(black, white)

        assert 0 < fitness <= 1

    def test_ssim_is_not_supported_anymore(self):
        """Debe rechazar ssim porque fue removido del proyecto."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Método de fitness desconocido"):
            compute_fitness(img, img, method="ssim")


class TestFitnessEvaluator:
    """Tests para la clase FitnessEvaluator."""

    def test_create_from_pil_image(self):
        """Debe crear evaluador desde imagen PIL."""
        img = Image.new("RGB", (100, 80), color=(128, 128, 128))

        evaluator = FitnessEvaluator(img)

        assert evaluator.width == 100
        assert evaluator.height == 80
        assert evaluator.evaluations == 0

    def test_create_from_numpy(self):
        """Debe crear evaluador desde array NumPy."""
        arr = np.zeros((60, 70, 3), dtype=np.uint8)

        evaluator = FitnessEvaluator(arr)

        assert evaluator.width == 70
        assert evaluator.height == 60

    def test_evaluate_individual(self):
        """Debe calcular fitness y almacenarlo en el individuo."""
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        evaluator = FitnessEvaluator(img)

        individual = Individual.random(num_triangles=5)
        assert individual.fitness is None

        fitness = evaluator.evaluate(individual)

        assert 0 < fitness <= 1
        assert individual.fitness == fitness
        assert evaluator.evaluations == 1

    def test_cached_fitness(self):
        """No debe recalcular si el fitness ya está calculado."""
        img = Image.new("RGB", (50, 50))
        evaluator = FitnessEvaluator(img)

        individual = Individual.random(num_triangles=5)

        # Primera evaluación
        fitness1 = evaluator.evaluate(individual)
        # Segunda evaluación (debe usar cache)
        fitness2 = evaluator.evaluate(individual)

        assert fitness1 == fitness2
        assert evaluator.evaluations == 1  # Solo una evaluación real

    def test_evaluate_population(self):
        """Debe evaluar toda la población."""
        img = Image.new("RGB", (30, 30))
        evaluator = FitnessEvaluator(img)

        population = [Individual.random(num_triangles=3) for _ in range(10)]

        fitness_values = evaluator.evaluate_population(population)

        assert len(fitness_values) == 10
        assert evaluator.evaluations == 10
        assert all(ind.fitness is not None for ind in population)

    def test_normalized_evaluator(self):
        """Debe normalizar fitness cuando se solicita (normalize=True → inverse_normalized)."""
        img = Image.new("RGB", (20, 20))
        evaluator = FitnessEvaluator(img, normalize=True)

        assert evaluator.method == "inverse_normalized"

        individual = Individual.random(num_triangles=5)
        fitness = evaluator.evaluate(individual)

        assert 0 < fitness <= 1

    def test_white_canvas_on_white_target(self):
        """Individuo vacío sobre imagen blanca debe tener fitness 1."""
        white_img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        evaluator = FitnessEvaluator(white_img)

        # Individuo sin triángulos = canvas blanco
        empty_individual = Individual(triangles=[])
        fitness = evaluator.evaluate(empty_individual)

        assert fitness == 1.0

    def test_reset_counter(self):
        """Debe reiniciar el contador de evaluaciones."""
        img = Image.new("RGB", (20, 20))
        evaluator = FitnessEvaluator(img)

        evaluator.evaluate(Individual.random(num_triangles=2))
        evaluator.evaluate(Individual.random(num_triangles=2))
        assert evaluator.evaluations == 2

        evaluator.reset_counter()
        assert evaluator.evaluations == 0
