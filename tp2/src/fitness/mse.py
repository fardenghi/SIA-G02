"""
Cálculo de error (MSE) y transformación a fitness.

El MSE se usa como señal de error base (menor es mejor), y luego se transforma
a fitness con `fitness = 1 / (1 + error)` para trabajar con la convención
clásica de algoritmos genéticos (mayor es mejor).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.genetic.individual import Individual
from src.rendering.canvas import Canvas


def calculate_mse(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    Calcula el Error Cuadrático Medio entre dos imágenes.

    MSE = (1/n) × Σ(pixel_rendered - pixel_target)²

    Args:
        rendered: Array de la imagen renderizada (height, width, 3).
        target: Array de la imagen objetivo (height, width, 3).

    Returns:
        Valor MSE (menor es mejor, 0 = imágenes idénticas).

    Raises:
        ValueError: Si las dimensiones no coinciden.
    """
    if rendered.shape != target.shape:
        raise ValueError(
            f"Las dimensiones no coinciden: "
            f"rendered={rendered.shape}, target={target.shape}"
        )

    # Convertir a float64 para evitar overflow en la resta
    rendered_f = rendered.astype(np.float64)
    target_f = target.astype(np.float64)

    # Calcular diferencias al cuadrado
    diff_squared = (rendered_f - target_f) ** 2

    # MSE promedio sobre todos los píxeles y canales
    mse = np.mean(diff_squared)

    return float(mse)


def calculate_normalized_mse(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    Calcula el MSE normalizado en el rango [0, 1].

    Útil para comparar fitness entre imágenes de diferentes tamaños.
    El valor máximo posible de MSE es 255² = 65025.

    Args:
        rendered: Array de la imagen renderizada.
        target: Array de la imagen objetivo.

    Returns:
        Valor MSE normalizado (0 = idénticas, 1 = máxima diferencia).
    """
    mse = calculate_mse(rendered, target)
    max_mse = 255.0**2  # Máxima diferencia posible
    return mse / max_mse


def error_to_fitness(error: float) -> float:
    """
    Convierte una métrica de error a fitness.

    Aplica la transformación:

        fitness = 1 / (1 + error)

    Args:
        error: Error no negativo (0 = ajuste perfecto).

    Returns:
        Fitness en el rango (0, 1], donde valores más altos son mejores.

    Raises:
        ValueError: Si el error es negativo.
    """
    if error < 0:
        raise ValueError(f"El error debe ser no negativo, recibido: {error}")

    return 1.0 / (1.0 + error)


def calculate_fitness(
    rendered: np.ndarray, target: np.ndarray, normalize_error: bool = False
) -> float:
    """
    Calcula fitness entre una imagen renderizada y la imagen objetivo.

    Si `normalize_error=True`, primero normaliza el MSE a [0, 1] y luego aplica
    la transformación a fitness.

    Args:
        rendered: Array de la imagen renderizada.
        target: Array de la imagen objetivo.
        normalize_error: Si True, usa MSE normalizado antes de convertir.

    Returns:
        Valor de fitness (mayor es mejor).
    """
    if normalize_error:
        error = calculate_normalized_mse(rendered, target)
    else:
        error = calculate_mse(rendered, target)

    return error_to_fitness(error)


class FitnessEvaluator:
    """
    Evaluador de fitness para individuos.

    Encapsula la imagen objetivo y el canvas para calcular
    el fitness de múltiples individuos de forma eficiente.

    Attributes:
        target: Array NumPy de la imagen objetivo.
        canvas: Canvas para renderizar individuos.
        evaluations: Contador de evaluaciones realizadas.
    """

    def __init__(self, target_image: Image.Image | np.ndarray, normalize: bool = False):
        """
        Inicializa el evaluador.

        Args:
            target_image: Imagen objetivo (PIL Image o NumPy array).
            normalize: Si True, normaliza el error MSE a [0, 1] antes de
                convertir a fitness.
        """
        if isinstance(target_image, Image.Image):
            target_image = target_image.convert("RGB")
            self.target = np.array(target_image, dtype=np.uint8)
            width, height = target_image.size
        else:
            self.target = target_image.astype(np.uint8)
            height, width = target_image.shape[:2]

        self.canvas = Canvas(width=width, height=height)
        self.normalize = normalize
        self.evaluations = 0

    @property
    def width(self) -> int:
        """Ancho de la imagen objetivo."""
        return self.canvas.width

    @property
    def height(self) -> int:
        """Alto de la imagen objetivo."""
        return self.canvas.height

    def evaluate(self, individual: Individual) -> float:
        """
        Evalúa el fitness de un individuo.

        Renderiza el individuo y calcula fitness contra la imagen objetivo.
        El resultado se almacena en individual.fitness.

        Args:
            individual: Individuo a evaluar.

        Returns:
            Valor de fitness (mayor es mejor).
        """
        # Si ya tiene fitness calculado, devolverlo
        if individual.fitness is not None:
            return individual.fitness

        # Renderizar y calcular fitness
        rendered = self.canvas.render_to_array(individual)

        fitness = calculate_fitness(
            rendered, self.target, normalize_error=self.normalize
        )

        individual.fitness = fitness
        self.evaluations += 1

        return fitness

    def evaluate_population(self, population: list[Individual]) -> list[float]:
        """
        Evalúa el fitness de toda una población.

        Args:
            population: Lista de individuos.

        Returns:
            Lista de valores de fitness.
        """
        return [self.evaluate(ind) for ind in population]

    def reset_counter(self):
        """Reinicia el contador de evaluaciones."""
        self.evaluations = 0
