"""
Cálculo de error (MSE) y transformación a fitness.

Métodos de fitness disponibles:

- ``"linear"``           1 - MSE/65025           rango [0, 1], lineal, recomendado
- ``"rmse"``             1 - sqrt(MSE)/255        rango [0, 1], penaliza menos errores grandes
- ``"inverse_normalized" 1 / (1 + MSE/65025)      rango (0.5, 1], más discriminativo cerca de 1
- ``"exponential"``      exp(-MSE_norm / scale)   rango (0, 1], escala ajustable
- ``"inverse_mse"``      1 / (1 + MSE)            rango (0, 0.003] en la práctica, NO recomendado
- ``"detail_weighted"``  MSE ponderado por bordes  refuerza regiones con detalle/contraste
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from src.genetic.individual import Individual
from src.rendering.factory import create_renderer

FITNESS_METHODS = frozenset(
    {
        "linear",
        "rmse",
        "inverse_normalized",
        "exponential",
        "inverse_mse",
        "detail_weighted",
        "composite",
        "ssim",
        "edge_loss",
    }
)

# Magnitud máxima de gradiente para uint8: cada canal puede cambiar 255 unidades
# entre píxeles adyacentes; np.gradient da diferencias centradas, máximo ~255.
_MAX_GRADIENT = 255.0 * math.sqrt(2.0)


def compute_detail_weight_map(target: np.ndarray, base: float = 0.3) -> np.ndarray:
    """
    Construye un mapa de pesos basado en el gradiente del target.

    Píxeles con alto contraste/borde reciben mayor peso; regiones lisas
    reciben el peso base. El mapa se normaliza para que su media sea 1,
    de forma que el weighted-MSE sea comparable en escala al MSE ordinario.

    Algoritmo:
        1. Convertir target a grises con coeficientes perceptuales.
        2. Calcular magnitud del gradiente con np.gradient (diferencias centradas).
        3. Normalizar al rango [0, 1].
        4. Mezclar con el peso base: weight = base + (1 - base) * grad_norm.
        5. Dividir por la media para que sum(w) / n = 1.

    Args:
        target: Array uint8 (H, W, 3) de la imagen objetivo.
        base:   Fracción de peso mínimo para regiones lisas. 0.0 = solo bordes
                pesan; 1.0 = equivalente a MSE uniforme. Valor recomendado: 0.3.

    Returns:
        Array float32 (H, W) de pesos, con media ≈ 1.
    """
    gray = (
        0.299 * target[:, :, 0].astype(np.float32)
        + 0.587 * target[:, :, 1].astype(np.float32)
        + 0.114 * target[:, :, 2].astype(np.float32)
    )
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)  # [0, 1]
    weights = base + (1.0 - base) * grad_norm  # [base, 1]
    weights /= weights.mean() + 1e-8  # media ≈ 1
    return weights.astype(np.float32)


def compute_ssim(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    SSIM global (sin ventana deslizante) entre dos imágenes RGB.

    Calcula el índice de similitud estructural usando estadísticas globales
    (media, varianza y covarianza sobre todo el canal). Es más rápido que el
    SSIM ventaneado estándar y suficiente como señal para el AG.

    Args:
        rendered: Array uint8 (H, W, 3) de la imagen renderizada.
        target:   Array uint8 (H, W, 3) de la imagen objetivo.

    Returns:
        SSIM ∈ [-1, 1]; en la práctica ∈ [0, 1] para imágenes naturales.
    """
    C1 = (0.01 * 255.0) ** 2  # ≈ 6.50
    C2 = (0.03 * 255.0) ** 2  # ≈ 58.52

    rendered_f = rendered.astype(np.float64)
    target_f = target.astype(np.float64)

    ssim_channels = []
    for c in range(3):
        x = rendered_f[:, :, c]
        y = target_f[:, :, c]
        mu_x = x.mean()
        mu_y = y.mean()
        sigma_x2 = float(x.var())
        sigma_y2 = float(y.var())
        sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

        num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_channels.append(num / den)

    return float(np.mean(ssim_channels))


def compute_target_edge_map(target: np.ndarray) -> np.ndarray:
    """
    Precomputa el mapa de bordes del target (magnitud del gradiente).

    Convierte a escala de grises con coeficientes perceptuales, aplica
    np.gradient y normaliza por la magnitud máxima teórica (_MAX_GRADIENT)
    para obtener valores en [0, 1].

    Args:
        target: Array uint8 (H, W, 3) de la imagen objetivo.

    Returns:
        Array float32 (H, W) con valores en [0, 1].
    """
    gray = (
        0.299 * target[:, :, 0].astype(np.float32)
        + 0.587 * target[:, :, 1].astype(np.float32)
        + 0.114 * target[:, :, 2].astype(np.float32)
    )
    gy, gx = np.gradient(gray)
    edges = np.sqrt(gx**2 + gy**2) / _MAX_GRADIENT
    return edges.astype(np.float32)


def compute_edge_loss(rendered: np.ndarray, target_edges: np.ndarray) -> float:
    """
    MSE entre el mapa de bordes del rendered y el del target.

    Ambos mapas se normalizan por _MAX_GRADIENT para obtener valores en [0, 1],
    por lo que el MSE resultante también está en [0, 1].

    Args:
        rendered:      Array uint8 (H, W, 3) de la imagen renderizada.
        target_edges:  Mapa de bordes precomputado del target, float32 [0, 1].

    Returns:
        EdgeLoss ∈ [0, 1].
    """
    gray = (
        0.299 * rendered[:, :, 0].astype(np.float32)
        + 0.587 * rendered[:, :, 1].astype(np.float32)
        + 0.114 * rendered[:, :, 2].astype(np.float32)
    )
    gy, gx = np.gradient(gray)
    rendered_edges = np.sqrt(gx**2 + gy**2) / _MAX_GRADIENT
    diff = rendered_edges - target_edges
    return float(np.mean(diff**2))


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
    return calculate_mse(rendered, target) / (255.0**2)


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
    weight_map: np.ndarray | None = None,
    target_edges: np.ndarray | None = None,
    composite_alpha: float = 0.5,
    composite_beta: float = 0.2,
    composite_gamma: float = 0.3,
) -> float:
    """
    Calcula el fitness entre dos imágenes con el método especificado.

    Args:
        rendered:         Imagen renderizada como array uint8 (H, W, 3).
        target:           Imagen objetivo como array uint8 (H, W, 3).
        method:           Uno de los métodos en :data:`FITNESS_METHODS`.
        exponential_scale: Escala para el método ``"exponential"``.
        weight_map:       Mapa de pesos precomputado para ``"detail_weighted"``.
        target_edges:     Mapa de bordes precomputado para ``"composite"``.
        composite_alpha:  Peso de (1 - SSIM) en el fitness compuesto.
        composite_beta:   Peso de MSE normalizado en el fitness compuesto.
        composite_gamma:  Peso de EdgeLoss en el fitness compuesto.

    Returns:
        Valor de fitness en (0, 1] (mayor es mejor).

    Raises:
        ValueError: Si ``method`` no es reconocido.
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

    if method == "detail_weighted":
        w = weight_map if weight_map is not None else compute_detail_weight_map(target)
        diff = rendered.astype(np.float32) - target.astype(np.float32)
        sq = diff**2  # (H, W, 3)
        per_pixel = sq.mean(axis=2)  # (H, W)  — media de canales
        weighted_mse = float(np.mean(w * per_pixel))
        weighted_nmse = weighted_mse / 65025.0
        return 1.0 - min(weighted_nmse, 1.0)

    if method == "composite":
        # Fitness = 1 - (α·(1−SSIM) + β·MSE_norm + γ·EdgeLoss) / (α+β+γ)
        ssim_val = compute_ssim(rendered, target)
        t_edges = (
            target_edges
            if target_edges is not None
            else compute_target_edge_map(target)
        )
        edge_loss = compute_edge_loss(rendered, t_edges)
        total_w = composite_alpha + composite_beta + composite_gamma
        if total_w < 1e-9:
            return 1.0 - nmse  # fallback a linear
        loss = (
            composite_alpha * (1.0 - ssim_val)
            + composite_beta * nmse
            + composite_gamma * edge_loss
        ) / total_w
        return max(0.0, 1.0 - loss)

    if method == "ssim":
        return compute_ssim(rendered, target)

    if method == "edge_loss":
        t_edges = (
            target_edges
            if target_edges is not None
            else compute_target_edge_map(target)
        )
        loss = compute_edge_loss(rendered, t_edges)
        return max(0.0, 1.0 - loss)

    raise AssertionError(f"Método de fitness no manejado: {method}")


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
        shape_type: str = "triangle",
        detail_weight_base: float = 0.3,
        composite_alpha: float = 0.5,
        composite_beta: float = 0.2,
        composite_gamma: float = 0.3,
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

        # Mapa de pesos para detail_weighted: precomputado una sola vez.
        self._weight_map: np.ndarray | None = None
        if method == "detail_weighted":
            self._weight_map = compute_detail_weight_map(
                self.target, base=detail_weight_base
            )

        # Mapa de bordes para composite y edge_loss: precomputado una sola vez.
        self._target_edges: np.ndarray | None = None
        if method in {"composite", "edge_loss"}:
            self._target_edges = compute_target_edge_map(self.target)

        self.composite_alpha = composite_alpha
        self.composite_beta = composite_beta
        self.composite_gamma = composite_gamma

        self.canvas = create_renderer(
            width=width,
            height=height,
            backend=renderer,
            shape_type=shape_type,
        )
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

        Evita reconvertir self.target en cada llamada.

        Args:
            rendered: Array uint8 (H, W, 3) recién renderizado.

        Returns:
            Valor de fitness en (0, 1].
        """
        if self.method == "detail_weighted":
            return compute_fitness(
                rendered,
                self.target,
                self.method,
                self.exponential_scale,
                weight_map=self._weight_map,
            )

        if self.method == "composite":
            return compute_fitness(
                rendered,
                self.target,
                self.method,
                target_edges=self._target_edges,
                composite_alpha=self.composite_alpha,
                composite_beta=self.composite_beta,
                composite_gamma=self.composite_gamma,
            )

        if self.method == "edge_loss":
            return compute_fitness(
                rendered,
                self.target,
                self.method,
                target_edges=self._target_edges,
            )

        if self.method == "ssim":
            return compute_ssim(rendered, self.target)

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
        """
        Evalúa el fitness de un individuo aisalado.
        """
        if individual.fitness is not None:
            return individual.fitness

        fitness = self._evaluate_stateless(individual)
        individual.fitness = fitness
        self.evaluations += 1
        return fitness

    def _evaluate_stateless(self, individual: Individual) -> float:
        """Rutina de renderizado sin tocar contadores globales para hilos."""
        rendered = self.canvas.render_to_array(individual)
        return self._compute_fitness_fast(rendered)

    def evaluate_population(self, population: list[Individual]) -> list[float]:
        """
        Evalúa el fitness de toda una población de manera secuencial.
        """
        return [self.evaluate(ind) for ind in population]

    def reset_counter(self):
        """Reinicia el contador de evaluaciones."""
        self.evaluations = 0

    def preload_maps(self, methods: list[str], detail_weight_base: float = 0.3) -> None:
        """Precomputa mapas pesados en caso de que vayan a ser utilizados luego en transiciones dinámicas."""
        if (
            any(m in {"composite", "edge_loss"} for m in methods)
            and self._target_edges is None
        ):
            self._target_edges = compute_target_edge_map(self.target)
        if "detail_weighted" in methods and getattr(self, "_weight_map", None) is None:
            self._weight_map = compute_detail_weight_map(
                self.target, base=detail_weight_base
            )

    def set_method(self, method: str):
        """Cambia la métrica dinámica de fitness."""
        if method not in FITNESS_METHODS:
            raise ValueError(f"Método desconocido: {method}")
        self.method = method
