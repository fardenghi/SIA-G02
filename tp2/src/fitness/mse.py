"""
Cálculo de error (MSE) y transformación a fitness.

Métodos de fitness disponibles:

- ``"linear"``              1 - MSE/65025           rango [0, 1], lineal, recomendado
- ``"rmse"``                1 - sqrt(MSE)/255        rango [0, 1], penaliza menos errores grandes
- ``"inverse_normalized"``  1 / (1 + MSE/65025)      rango (0.5, 1], más discriminativo cerca de 1
- ``"exponential"``         exp(-MSE_norm / scale)   rango (0, 1], escala ajustable
- ``"ssim"``                SSIM mapeado a [0, 1]    requiere scikit-image, métrica perceptual
- ``"inverse_mse"``         1 / (1 + MSE)            rango (0, 0.003] en la práctica, NO recomendado
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from src.genetic.individual import Individual
from src.rendering.canvas import Canvas
from src.rendering.gpu_canvas import GPUCanvas, MODERNGL_AVAILABLE

FITNESS_METHODS = frozenset(
    {"linear", "rmse", "inverse_normalized", "exponential", "ssim", "inverse_mse"}
)


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

    rendered_f = rendered.astype(np.float64)
    target_f = target.astype(np.float64)
    diff_squared = (rendered_f - target_f) ** 2
    return float(np.mean(diff_squared))


def calculate_normalized_mse(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    Calcula el MSE normalizado en el rango [0, 1].

    El valor máximo posible de MSE es 255² = 65025.

    Args:
        rendered: Array de la imagen renderizada.
        target: Array de la imagen objetivo.

    Returns:
        Valor MSE normalizado (0 = idénticas, 1 = máxima diferencia).
    """
    return calculate_mse(rendered, target) / (255.0 ** 2)


def error_to_fitness(error: float) -> float:
    """
    Convierte una métrica de error a fitness con ``1 / (1 + error)``.

    Args:
        error: Error no negativo (0 = ajuste perfecto).

    Returns:
        Fitness en el rango (0, 1].
    """
    if error < 0:
        raise ValueError(f"El error debe ser no negativo, recibido: {error}")
    return 1.0 / (1.0 + error)


def calculate_fitness(
    rendered: np.ndarray,
    target: np.ndarray,
    normalize_error: bool = False,
) -> float:
    """Calcula fitness con ``inverse_mse`` o ``inverse_normalized`` según ``normalize_error``."""
    if normalize_error:
        return error_to_fitness(calculate_normalized_mse(rendered, target))
    return error_to_fitness(calculate_mse(rendered, target))


def compute_fitness(
    rendered: np.ndarray,
    target: np.ndarray,
    method: str = "linear",
    exponential_scale: float = 0.1,
) -> float:
    """
    Calcula el fitness entre dos imágenes con el método especificado.

    Args:
        rendered: Imagen renderizada como array uint8 (H, W, 3).
        target:   Imagen objetivo como array uint8 (H, W, 3).
        method:   Uno de los métodos en :data:`FITNESS_METHODS`.
        exponential_scale: Escala para el método ``"exponential"``.
            Valores menores = curva más pronunciada (mayor presión selectiva).

    Returns:
        Valor de fitness en (0, 1] (mayor es mejor).

    Raises:
        ValueError: Si ``method`` no es reconocido.
        ImportError: Si ``method="ssim"`` y scikit-image no está instalado.
    """
    if method not in FITNESS_METHODS:
        raise ValueError(
            f"Método de fitness desconocido: {method!r}. "
            f"Válidos: {sorted(FITNESS_METHODS)}"
        )

    if method == "inverse_mse":
        return error_to_fitness(calculate_mse(rendered, target))

    nmse = calculate_normalized_mse(rendered, target)

    if method == "linear":
        return 1.0 - nmse

    if method == "rmse":
        return 1.0 - math.sqrt(nmse)

    if method == "inverse_normalized":
        return error_to_fitness(nmse)

    if method == "exponential":
        return math.exp(-nmse / max(exponential_scale, 1e-9))

    # method == "ssim"
    try:
        from skimage.metrics import structural_similarity as ssim_fn  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "El método 'ssim' requiere scikit-image: pip install scikit-image"
        ) from exc
    score = ssim_fn(rendered, target, channel_axis=2, data_range=255)
    # ssim ∈ [-1, 1] → mapeamos a [0, 1]
    return float((score + 1.0) / 2.0)


class FitnessEvaluator:
    """
    Evaluador de fitness para individuos.

    Encapsula la imagen objetivo y el canvas para calcular
    el fitness de múltiples individuos de forma eficiente.

    Attributes:
        target:            Array NumPy de la imagen objetivo.
        canvas:            Canvas para renderizar individuos.
        method:            Método de fitness activo.
        exponential_scale: Escala para el método exponencial.
        evaluations:       Contador de evaluaciones realizadas.
    """

    def __init__(
        self,
        target_image: Image.Image | np.ndarray,
        method: str = "linear",
        exponential_scale: float = 0.1,
        # normalize mantenido para compatibilidad, ignorado si method != "linear"
        normalize: bool = False,
        renderer: str = "cpu",
    ):
        """
        Inicializa el evaluador.

        Args:
            target_image:      Imagen objetivo (PIL Image o NumPy array).
            method:            Método de fitness. Ver :data:`FITNESS_METHODS`.
            exponential_scale: Escala para método ``"exponential"``.
            normalize:         Obsoleto. Si True y method="linear",
                               usa ``"inverse_normalized"`` (compatibilidad).
        """
        if isinstance(target_image, Image.Image):
            target_image = target_image.convert("RGB")
            self.target = np.array(target_image, dtype=np.uint8)
            width, height = target_image.size
        else:
            self.target = target_image.astype(np.uint8)
            height, width = target_image.shape[:2]

        # Backward-compat: normalize=True equivale a inverse_normalized
        if normalize and method == "linear":
            method = "inverse_normalized"

        if method not in FITNESS_METHODS:
            raise ValueError(
                f"Método de fitness desconocido: {method!r}. "
                f"Válidos: {sorted(FITNESS_METHODS)}"
            )

        # Cache float32 del target: evita reconversión en cada evaluación.
        # El target nunca cambia; rendered sí se recalcula cada vez.
        self._target_f32 = self.target.astype(np.float32)

        if renderer == "gpu":
            if MODERNGL_AVAILABLE:
                self.canvas = GPUCanvas(width=width, height=height)
            else:
                import warnings
                warnings.warn(
                    "moderngl no instalado; usando CPU. "
                    "Instalar con: pip install moderngl  o  uv sync --extra gpu",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.canvas = Canvas(width=width, height=height)
        else:
            self.canvas = Canvas(width=width, height=height)
        self.method = method
        self.exponential_scale = exponential_scale
        self.evaluations = 0

    @property
    def width(self) -> int:
        """Ancho de la imagen objetivo."""
        return self.canvas.width

    @property
    def height(self) -> int:
        """Alto de la imagen objetivo."""
        return self.canvas.height

    def _compute_fitness_fast(self, rendered: np.ndarray) -> float:
        """
        Calcula fitness usando el target pre-convertido a float32.

        Evita reconvertir self.target en cada llamada. SSIM delega al path
        original porque skimage espera uint8.

        Args:
            rendered: Array uint8 (H, W, 3) recién renderizado.

        Returns:
            Valor de fitness en (0, 1].
        """
        if self.method == "ssim":
            return compute_fitness(
                rendered, self.target, self.method, self.exponential_scale
            )

        diff = rendered.astype(np.float32) - self._target_f32
        # nmse = MSE / 255² ∈ [0, 1]; dot product evita array intermedio de cuadrados
        nmse = float(np.dot(diff.ravel(), diff.ravel())) / (diff.size * 65025.0)

        if self.method == "linear":
            return 1.0 - nmse
        if self.method == "rmse":
            return 1.0 - math.sqrt(max(nmse, 0.0))
        if self.method == "inverse_normalized":
            return 1.0 / (1.0 + nmse)
        if self.method == "exponential":
            return math.exp(-nmse / max(self.exponential_scale, 1e-9))
        # inverse_mse: usa MSE crudo, no normalizado
        return 1.0 / (1.0 + nmse * 65025.0)

    def evaluate(self, individual: Individual) -> float:
        """
        Evalúa el fitness de un individuo.

        Renderiza el individuo y calcula fitness contra la imagen objetivo.
        El resultado se almacena en ``individual.fitness``.

        Args:
            individual: Individuo a evaluar.

        Returns:
            Valor de fitness (mayor es mejor).
        """
        if individual.fitness is not None:
            return individual.fitness

        rendered = self.canvas.render_to_array(individual)
        fitness = self._compute_fitness_fast(rendered)
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
