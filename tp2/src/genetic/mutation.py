"""
Operadores de mutación.

Implementa los 4 tipos de mutación según la teoría:
- SINGLE_GENE: Muta exactamente 1 gen (triángulo)
- LIMITED_MULTIGEN: Muta entre 1 y M genes
- UNIFORM_MULTIGEN: Cada gen tiene probabilidad independiente de mutar
- COMPLETE: Muta todos los genes del individuo

Alteración de coordenadas, colores, transparencia y orden Z-index
de los triángulos.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from src.genetic.individual import Individual, Triangle


class MutationType(Enum):
    """
    Tipos de mutación disponibles según la teoría.

    - SINGLE_GENE: Muta exactamente 1 gen (triángulo) con probabilidad Pm.
    - LIMITED_MULTIGEN: Muta entre [1, M] genes con probabilidad Pm.
    - UNIFORM_MULTIGEN: Cada gen tiene probabilidad Pm de ser mutado (independiente).
    - COMPLETE: Muta todos los genes con probabilidad Pm.
    """

    SINGLE_GENE = "single_gene"
    LIMITED_MULTIGEN = "limited_multigen"
    UNIFORM_MULTIGEN = "uniform_multigen"
    COMPLETE = "complete"
    ERROR_MAP_GUIDED = "error_map_guided"


@dataclass
class MutationParams:
    """
    Parámetros de mutación.

    Attributes:
        mutation_type: Tipo de mutación a aplicar.
        probability: Probabilidad de mutar un individuo (Pm).
        gene_probability: Probabilidad de mutar cada gen (solo para UNIFORM_MULTIGEN).
        max_genes: Máximo de genes a mutar (M) para LIMITED_MULTIGEN.
        position_delta: Magnitud máxima de perturbación en coordenadas.
        color_delta: Magnitud máxima de perturbación en color (0-255).
        alpha_delta: Magnitud máxima de perturbación en alfa.
        field_probability: Probabilidad de mutar cada float individual dentro de
            un triángulo seleccionado (vértices, R, G, B, alpha).
            1.0 = todos los campos mutan siempre (comportamiento original).
            < 1.0 = mutación per-float; en promedio se tocan field_probability * 10
            valores por triángulo.
        guided_ratio: Fracción de mutaciones guiadas por error map vs uniformes aleatorias.
            Solo aplica a ERROR_MAP_GUIDED: 0.75 → 75% guiadas, 25% aleatorias.
    """

    mutation_type: MutationType = MutationType.UNIFORM_MULTIGEN
    probability: float = 0.3
    gene_probability: float = 0.1
    max_genes: int = 3
    position_delta: float = 0.1
    color_delta: int = 30
    alpha_delta: float = 0.1
    field_probability: float = 1.0
    guided_ratio: float = 0.75

    def __post_init__(self):
        """Valida los parámetros."""
        if not 0 <= self.probability <= 1:
            raise ValueError("probability debe estar en [0, 1]")
        if not 0 <= self.gene_probability <= 1:
            raise ValueError("gene_probability debe estar en [0, 1]")
        if not 0 <= self.position_delta <= 1:
            raise ValueError("position_delta debe estar en [0, 1]")
        if not 0 <= self.color_delta <= 255:
            raise ValueError("color_delta debe estar en [0, 255]")
        if not 0 <= self.alpha_delta <= 1:
            raise ValueError("alpha_delta debe estar en [0, 1]")
        if self.max_genes < 1:
            raise ValueError("max_genes debe ser >= 1")
        if not 0 <= self.field_probability <= 1:
            raise ValueError("field_probability debe estar en [0, 1]")
        if not 0 <= self.guided_ratio <= 1:
            raise ValueError("guided_ratio debe estar en [0, 1]")


def mutate_value(value: float, delta: float, min_val: float, max_val: float) -> float:
    """
    Muta un valor con perturbación gaussiana.

    Args:
        value: Valor actual.
        delta: Desviación estándar de la perturbación.
        min_val: Valor mínimo permitido.
        max_val: Valor máximo permitido.

    Returns:
        Valor mutado dentro del rango.
    """
    perturbation = random.gauss(0, delta)
    new_value = value + perturbation
    return max(min_val, min(max_val, new_value))


def mutate_int_value(value: int, delta: int, min_val: int, max_val: int) -> int:
    """
    Muta un valor entero con perturbación.

    Args:
        value: Valor actual.
        delta: Magnitud máxima de la perturbación.
        min_val: Valor mínimo permitido.
        max_val: Valor máximo permitido.

    Returns:
        Valor mutado dentro del rango.
    """
    perturbation = random.randint(-delta, delta)
    new_value = value + perturbation
    return max(min_val, min(max_val, new_value))


def mutate_triangle(triangle: Triangle, params: MutationParams) -> Triangle:
    """
    Muta un triángulo.

    Con field_probability < 1.0 opera en modo per-float: cada valor individual
    (coordenadas de vértices, R, G, B, alpha) se muta de forma independiente
    con probabilidad field_probability. Con field_probability = 1.0 (default)
    todos los campos mutan siempre, igual que el comportamiento original.

    Args:
        triangle: Triángulo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo triángulo mutado.
    """
    p = params.field_probability

    # Mutar vértices
    new_vertices = []
    for x, y in triangle.vertices:
        new_x = mutate_value(x, params.position_delta, 0.0, 1.0) if random.random() < p else x
        new_y = mutate_value(y, params.position_delta, 0.0, 1.0) if random.random() < p else y
        new_vertices.append((new_x, new_y))

    # Mutar color
    r, g, b, a = triangle.color
    new_r = mutate_int_value(r, params.color_delta, 0, 255) if random.random() < p else r
    new_g = mutate_int_value(g, params.color_delta, 0, 255) if random.random() < p else g
    new_b = mutate_int_value(b, params.color_delta, 0, 255) if random.random() < p else b
    new_a = mutate_value(a, params.alpha_delta, 0.0, 1.0) if random.random() < p else a

    return Triangle(vertices=new_vertices, color=(new_r, new_g, new_b, new_a))


def _apply_zindex_swap(triangles: List[Triangle], probability: float) -> None:
    """
    Aplica swap de Z-index (intercambio de posiciones) in-place.

    Args:
        triangles: Lista de triángulos (se modifica in-place).
        probability: Probabilidad de realizar el swap.
    """
    if len(triangles) >= 2 and random.random() < probability:
        i, j = random.sample(range(len(triangles)), 2)
        triangles[i], triangles[j] = triangles[j], triangles[i]


def mutate_single_gene(individual: Individual, params: MutationParams) -> Individual:
    """
    Mutación de gen único.

    Con probabilidad Pm, muta exactamente UN gen (triángulo) elegido al azar.

    Cuándo sirve:
    - Cuando se quiere hacer cambios chicos
    - Cuando conviene no romper demasiado la estructura del individuo
    - Cuando el problema es sensible a cambios grandes

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    if random.random() > params.probability:
        return individual.copy()

    new_triangles = [t.copy() for t in individual.triangles]

    if len(new_triangles) > 0:
        # Elegir exactamente 1 gen al azar
        idx = random.randrange(len(new_triangles))
        new_triangles[idx] = mutate_triangle(new_triangles[idx], params)

    # Swap de Z-index con probabilidad reducida
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


def mutate_limited_multigen(
    individual: Individual, params: MutationParams
) -> Individual:
    """
    Mutación multigen limitada.

    Con probabilidad Pm, muta entre [1, M] genes elegidos al azar.
    M está definido por params.max_genes.

    Cuándo sirve:
    - Cuando una mutación de un solo gen resulta demasiado conservadora
    - Cuando se quiere explorar más sin llegar a modificar todo el individuo
    - Cuando tiene sentido controlar cuánto "ruido" se introduce

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    if random.random() > params.probability:
        return individual.copy()

    new_triangles = [t.copy() for t in individual.triangles]

    if len(new_triangles) > 0:
        # Elegir entre 1 y M genes (limitado por el tamaño del individuo)
        max_to_mutate = min(params.max_genes, len(new_triangles))
        num_to_mutate = random.randint(1, max_to_mutate)

        # Seleccionar índices únicos al azar
        indices_to_mutate = random.sample(range(len(new_triangles)), num_to_mutate)

        for idx in indices_to_mutate:
            new_triangles[idx] = mutate_triangle(new_triangles[idx], params)

    # Swap de Z-index
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


def mutate_uniform_multigen(
    individual: Individual, params: MutationParams
) -> Individual:
    """
    Mutación multigen uniforme.

    Cada gen tiene probabilidad independiente (gene_probability) de ser mutado.
    No hay decisión global de "si ocurre mutación o no", cada gen decide
    independientemente.

    Cuándo sirve:
    - Cuando no se quiere fijar de antemano cuántos genes cambiar
    - Cuando se busca una mutación distribuida a lo largo de todo el cromosoma
    - Cuando el cromosoma tiene muchos genes relativamente independientes

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    # En uniform_multigen, cada gen decide independientemente
    # Usamos gene_probability como la probabilidad de cada gen
    new_triangles = []

    for triangle in individual.triangles:
        if random.random() < params.gene_probability:
            new_triangles.append(mutate_triangle(triangle, params))
        else:
            new_triangles.append(triangle.copy())

    # Swap de Z-index
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


def mutate_complete(individual: Individual, params: MutationParams) -> Individual:
    """
    Mutación completa.

    Con probabilidad Pm, muta TODOS los genes del individuo.

    Cuándo sirve:
    - Para introducir un cambio muy fuerte
    - Para escapar de poblaciones demasiado estancadas
    - Como operador más agresivo de exploración

    Riesgo: puede romper estructuras buenas que ya se habían encontrado.

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación.

    Returns:
        Nuevo individuo mutado.
    """
    if random.random() > params.probability:
        return individual.copy()

    # Mutar TODOS los triángulos
    new_triangles = [mutate_triangle(t, params) for t in individual.triangles]

    # Swap de Z-index
    _apply_zindex_swap(new_triangles, params.gene_probability)

    return Individual(triangles=new_triangles)


def _mutate_triangle_guided(
    triangle: Triangle,
    params: MutationParams,
    sample_cdf: np.ndarray,
    sample_shape: Tuple[int, int],
) -> Triangle:
    """
    Muta un triángulo sesgando sus vértices hacia zonas de alto error.

    Usa ``np.searchsorted`` sobre una CDF precomputada en lugar de
    ``np.random.choice(p=…)``, lo que reduce el costo de cada muestra de
    O(n) a O(log n).  Además genera los 3 uniformes de golpe para
    amortizar el overhead de numpy.

    Args:
        triangle: Triángulo original.
        params: Parámetros de mutación.
        sample_cdf: CDF aplanada del error map, precomputada en set_error_map.
        sample_shape: Forma (H, W) del mapa de muestreo.

    Returns:
        Nuevo triángulo mutado.
    """
    sh, sw = sample_shape
    # Decidir qué vértices son guiados de una sola vez (sin loop de random.random)
    guided_mask = np.random.random(3) < params.guided_ratio
    n_guided = int(guided_mask.sum())

    new_vertices: List[Tuple[float, float]] = list(triangle.vertices)

    if n_guided > 0:
        # Muestrear todos los píxeles guiados de golpe: O(n_guided * log |CDF|)
        u = np.random.random(n_guided)
        pixel_indices = np.searchsorted(sample_cdf, u).clip(0, len(sample_cdf) - 1)
        pys, pxs = np.divmod(pixel_indices, sw)
        g = 0
        for i in range(3):
            if guided_mask[i]:
                px, py = int(pxs[g]), int(pys[g])
                new_vertices[i] = (
                    max(0.0, min(1.0, px / sw + random.gauss(0, params.position_delta))),
                    max(0.0, min(1.0, py / sh + random.gauss(0, params.position_delta))),
                )
                g += 1
            else:
                x, y = triangle.vertices[i]
                new_vertices[i] = (
                    mutate_value(x, params.position_delta, 0.0, 1.0),
                    mutate_value(y, params.position_delta, 0.0, 1.0),
                )
    else:
        for i, (x, y) in enumerate(triangle.vertices):
            new_vertices[i] = (
                mutate_value(x, params.position_delta, 0.0, 1.0),
                mutate_value(y, params.position_delta, 0.0, 1.0),
            )

    p = params.field_probability
    r, g_c, b, a = triangle.color
    new_r = mutate_int_value(r, params.color_delta, 0, 255) if random.random() < p else r
    new_g_c = mutate_int_value(g_c, params.color_delta, 0, 255) if random.random() < p else g_c
    new_b = mutate_int_value(b, params.color_delta, 0, 255) if random.random() < p else b
    new_a = mutate_value(a, params.alpha_delta, 0.0, 1.0) if random.random() < p else a

    return Triangle(
        vertices=new_vertices,
        color=(new_r, new_g_c, new_b, new_a),
    )


def mutate_error_map_guided(
    individual: Individual,
    params: MutationParams,
    integral_map: np.ndarray,
    sample_cdf: np.ndarray,
    sample_shape: Tuple[int, int],
) -> Individual:
    """
    Mutación guiada por mapa de error espacial.

    Sesga la selección de triángulos a mutar hacia aquellos que cubren
    regiones con mayor error (bounding box). La fracción ``guided_ratio``
    de las mutaciones se hace de forma guiada; el resto es aleatoria
    uniforme (exploración).

    Implementación eficiente:
    - Los pesos por triángulo se calculan con imagen integral (O(1) por caja),
      totalmente vectorizado sin bucle Python.
    - El muestreo de posiciones usa CDF precomputada + ``searchsorted``
      (O(log n) por muestra vs O(n) de ``np.random.choice(p=…)``).

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación (usa guided_ratio).
        integral_map: Imagen integral (H+1, W+1) precomputada en set_error_map.
        sample_cdf: CDF aplanada del error map para muestreo de posiciones.
        sample_shape: Forma (H, W) del mapa de muestreo.

    Returns:
        Nuevo individuo mutado.
    """
    if random.random() > params.probability:
        return individual.copy()

    n = len(individual.triangles)
    if n == 0:
        return individual.copy()

    H = integral_map.shape[0] - 1
    W = integral_map.shape[1] - 1

    # --- Pesos por triángulo via imagen integral (vectorizado) ---
    # verts: (n, 3, 2)  →  xs: (n, 3), ys: (n, 3)
    verts = np.array([t.vertices for t in individual.triangles], dtype=np.float64)

    xs, ys = verts[:, :, 0], verts[:, :, 1]
    x0s = np.maximum(0, (xs.min(axis=1) * W).astype(np.int32))
    x1s = np.minimum(W, (xs.max(axis=1) * W).astype(np.int32) + 1)
    y0s = np.maximum(0, (ys.min(axis=1) * H).astype(np.int32))
    y1s = np.minimum(H, (ys.max(axis=1) * H).astype(np.int32) + 1)

    areas = (x1s - x0s) * (y1s - y0s)
    box_sums = (
        integral_map[y1s, x1s]
        - integral_map[y0s, x1s]
        - integral_map[y1s, x0s]
        + integral_map[y0s, x0s]
    )
    global_mean = integral_map[-1, -1] / max(H * W, 1)
    weights = np.where(areas > 0, box_sums / np.maximum(areas, 1), global_mean)

    w_sum = weights.sum()
    tri_probs = weights / w_sum if w_sum > 1e-12 else np.ones(n) / n

    # --- Reparto guiadas / uniformes ---
    expected = max(1, int(round(params.gene_probability * n)))
    num_guided = max(0, int(round(expected * params.guided_ratio)))
    num_uniform = expected - num_guided

    new_triangles = [t.copy() for t in individual.triangles]
    mutated: set = set()

    if num_guided > 0:
        k = min(num_guided, n)
        guided_idx = np.random.choice(n, size=k, replace=False, p=tri_probs)
        for idx in guided_idx:
            new_triangles[idx] = _mutate_triangle_guided(
                new_triangles[idx], params, sample_cdf, sample_shape
            )
            mutated.add(int(idx))

    remaining = [i for i in range(n) if i not in mutated]
    if num_uniform > 0 and remaining:
        uni_idx = random.sample(remaining, min(num_uniform, len(remaining)))
        for idx in uni_idx:
            new_triangles[idx] = mutate_triangle(new_triangles[idx], params)

    _apply_zindex_swap(new_triangles, params.gene_probability)
    return Individual(triangles=new_triangles)


# Dispatcher de métodos de mutación
_MUTATION_FUNCTIONS = {
    MutationType.SINGLE_GENE: mutate_single_gene,
    MutationType.LIMITED_MULTIGEN: mutate_limited_multigen,
    MutationType.UNIFORM_MULTIGEN: mutate_uniform_multigen,
    MutationType.COMPLETE: mutate_complete,
}


def mutate_individual(individual: Individual, params: MutationParams) -> Individual:
    """
    Muta un individuo usando el método especificado en params.

    Dispatcher que selecciona la función de mutación apropiada según
    params.mutation_type.

    Args:
        individual: Individuo a mutar.
        params: Parámetros de mutación (incluye el tipo).

    Returns:
        Nuevo individuo mutado.
    """
    mutation_func = _MUTATION_FUNCTIONS.get(params.mutation_type)
    if mutation_func is None:
        raise ValueError(f"Tipo de mutación no soportado: {params.mutation_type}")

    return mutation_func(individual, params)


@dataclass
class AdaptiveSigmaConfig:
    """
    Configuración del sigma adaptativo.

    El scale multiplica position_delta, alpha_delta y color_delta del Mutator.
    Decae cuando hay estancamiento (refinamiento) y sube cuando hay mejoras (exploración).

    Attributes:
        scale_min: Escala mínima permitida (piso de refinamiento).
        scale_max: Escala máxima permitida (techo de exploración). Valor inicial.
        decay_factor: Factor de decaimiento por generación estancada (< 1).
        recovery_factor: Factor de recuperación por generación con mejora (> 1).
        stagnation_window: Generaciones a observar para detectar estancamiento.
        min_improvement: Mejora mínima en la ventana para no considerarse estancamiento.
    """

    scale_min: float = 0.1
    scale_max: float = 1.0
    decay_factor: float = 0.9
    recovery_factor: float = 1.1
    stagnation_window: int = 15
    min_improvement: float = 1e-4

    def __post_init__(self):
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("scale_min debe estar en (0, scale_max]")
        if not 0 < self.decay_factor < 1:
            raise ValueError("decay_factor debe estar en (0, 1)")
        if self.recovery_factor <= 1:
            raise ValueError("recovery_factor debe ser > 1")
        if self.stagnation_window < 2:
            raise ValueError("stagnation_window debe ser >= 2")


class AdaptiveSigma:
    """
    Sigma adaptativo basado en el progreso real del fitness.

    Mantiene un historial de los últimos `stagnation_window` valores del mejor
    fitness. Si la mejora acumulada en esa ventana es menor que `min_improvement`,
    decae el scale para refinar; si hay mejoras significativas, lo recupera para
    explorar más.

    Esto contrasta con el sigma no-uniforme clásico (que decae con el tiempo),
    ya que aquí la adaptación responde al estado real de la búsqueda.
    """

    def __init__(self, config: AdaptiveSigmaConfig | None = None):
        cfg = config or AdaptiveSigmaConfig()
        self._scale_min = cfg.scale_min
        self._scale_max = cfg.scale_max
        self._decay = cfg.decay_factor
        self._recovery = cfg.recovery_factor
        self._window = cfg.stagnation_window
        self._min_improvement = cfg.min_improvement
        self._scale: float = cfg.scale_max  # arranca en máximo (exploración)
        self._history: list[float] = []

    @property
    def scale(self) -> float:
        """Escala actual de los deltas de mutación."""
        return self._scale

    def update(self, best_fitness: float) -> None:
        """
        Actualiza el scale con el mejor fitness de la generación actual.

        Args:
            best_fitness: Mejor fitness de la generación (tp2 maximiza).
        """
        self._history.append(best_fitness)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) < self._window:
            return

        # Mejora acumulada en la ventana (positivo = mejoró, tp2 maximiza)
        improvement = self._history[-1] - self._history[0]

        if improvement < self._min_improvement:
            # Estancamiento: reducir sigma para refinar
            self._scale = max(self._scale_min, self._scale * self._decay)
        else:
            # Mejora real: recuperar sigma para seguir explorando
            self._scale = min(self._scale_max, self._scale * self._recovery)


class Mutator:
    """
    Clase para aplicar mutaciones a individuos.

    Encapsula los parámetros de mutación y proporciona
    una interfaz simple. Soporta sigma adaptativo opcional.
    """

    def __init__(
        self,
        params: MutationParams | None = None,
        adaptive_sigma: AdaptiveSigma | None = None,
    ):
        """
        Args:
            params: Parámetros de mutación. Si es None, usa defaults.
            adaptive_sigma: Sigma adaptativo opcional. Si se provee, escala
                los deltas de mutación en función del progreso del fitness.
        """
        self.params = params or MutationParams()
        self._adaptive_sigma = adaptive_sigma
        # Estado precomputado para mutación guiada (actualizado por set_error_map)
        self._integral_map: Optional[np.ndarray] = None   # imagen integral (H+1, W+1)
        self._sample_cdf: Optional[np.ndarray] = None     # CDF aplanada para searchsorted
        self._sample_shape: Tuple[int, int] = (0, 0)      # (H, W) del mapa de muestreo

    @property
    def mutation_type(self) -> MutationType:
        """Retorna el tipo de mutación configurado."""
        return self.params.mutation_type

    @property
    def sigma_scale(self) -> float:
        """Escala actual del sigma adaptativo (1.0 si no está configurado)."""
        if self._adaptive_sigma is not None:
            return self._adaptive_sigma.scale
        return 1.0

    def set_error_map(self, error_map: np.ndarray) -> None:
        """
        Actualiza el mapa de error y precomputa las estructuras de consulta.

        Construye dos estructuras una sola vez por generación:
        - **Imagen integral** (summed area table): permite calcular el error
          promedio de cualquier bounding box en O(1), sin bucle Python.
        - **CDF aplanada**: permite muestrear posiciones proporcionales al
          error en O(log n) con ``searchsorted``, vs O(n) de
          ``np.random.choice(p=…)``.

        Args:
            error_map: Mapa de error (H, W) float32/float64.
                Típicamente ``((rendered - target)**2).mean(axis=2)``.
        """
        H, W = error_map.shape[:2]
        self._sample_shape = (H, W)

        # Imagen integral para pesos vectorizados de bounding box
        em = error_map.astype(np.float64)
        integral = np.zeros((H + 1, W + 1), dtype=np.float64)
        integral[1:, 1:] = em.cumsum(axis=0).cumsum(axis=1)
        self._integral_map = integral

        # CDF para muestreo de posiciones en O(log n)
        flat = em.ravel()
        s = flat.sum()
        probs = flat / s if s > 1e-12 else np.ones(len(flat)) / len(flat)
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0  # seguridad numérica
        self._sample_cdf = cdf

    def update(self, best_fitness: float) -> None:
        """
        Actualiza el sigma adaptativo con el mejor fitness de la generación.

        Args:
            best_fitness: Mejor fitness de la generación actual.
        """
        if self._adaptive_sigma is not None:
            self._adaptive_sigma.update(best_fitness)

    def mutate(self, individual: Individual) -> Individual:
        """
        Muta un individuo.

        Si hay sigma adaptativo, escala position_delta, alpha_delta y color_delta
        por el scale actual antes de mutar. Si el tipo es ERROR_MAP_GUIDED y hay
        un error map cargado, delega a mutate_error_map_guided; de lo contrario
        cae a uniform_multigen como fallback.

        Args:
            individual: Individuo a mutar.

        Returns:
            Nuevo individuo (posiblemente mutado).
        """
        if self._adaptive_sigma is not None:
            scale = self._adaptive_sigma.scale
            effective_params = MutationParams(
                mutation_type=self.params.mutation_type,
                probability=self.params.probability,
                gene_probability=self.params.gene_probability,
                max_genes=self.params.max_genes,
                position_delta=self.params.position_delta * scale,
                color_delta=max(1, int(self.params.color_delta * scale)),
                alpha_delta=self.params.alpha_delta * scale,
                field_probability=self.params.field_probability,
                guided_ratio=self.params.guided_ratio,
            )
        else:
            effective_params = self.params

        if effective_params.mutation_type == MutationType.ERROR_MAP_GUIDED:
            if self._integral_map is None:
                # Sin error map todavía (primera generación): fallback uniforme
                return mutate_uniform_multigen(individual, effective_params)
            return mutate_error_map_guided(
                individual, effective_params,
                self._integral_map, self._sample_cdf, self._sample_shape,
            )

        return mutate_individual(individual, effective_params)

    def mutate_population(self, population: list[Individual]) -> list[Individual]:
        """
        Muta toda una población.

        Args:
            population: Lista de individuos.

        Returns:
            Nueva lista con individuos mutados.
        """
        return [self.mutate(ind) for ind in population]


def create_mutation_params(
    mutation_method: str = "uniform_multigen",
    probability: float = 0.3,
    gene_probability: float = 0.1,
    max_genes: int = 3,
    position_delta: float = 0.1,
    color_delta: int = 30,
    alpha_delta: float = 0.1,
    field_probability: float = 1.0,
    guided_ratio: float = 0.75,
) -> MutationParams:
    """
    Factory para crear parámetros de mutación desde strings.

    Args:
        mutation_method: Nombre del método de mutación.
            Opciones: "single_gene", "limited_multigen", "uniform_multigen",
            "complete", "error_map_guided"
        probability: Probabilidad de mutar un individuo (Pm).
        gene_probability: Probabilidad por gen (para uniform_multigen y error_map_guided).
        max_genes: Máximo de genes a mutar (M) para limited_multigen.
        position_delta: Delta de posición.
        color_delta: Delta de color.
        alpha_delta: Delta de alfa.
        field_probability: Probabilidad de mutar cada float individual dentro
            del triángulo seleccionado. 1.0 = todos los campos (default).
        guided_ratio: Fracción guiada por error map (solo para error_map_guided).

    Returns:
        MutationParams configurado.

    Raises:
        ValueError: Si el método no es válido.
    """
    method_map = {
        "single_gene": MutationType.SINGLE_GENE,
        "limited_multigen": MutationType.LIMITED_MULTIGEN,
        "uniform_multigen": MutationType.UNIFORM_MULTIGEN,
        "complete": MutationType.COMPLETE,
        "error_map_guided": MutationType.ERROR_MAP_GUIDED,
    }

    if mutation_method not in method_map:
        valid_methods = ", ".join(method_map.keys())
        raise ValueError(
            f"Método de mutación '{mutation_method}' no válido. "
            f"Opciones: {valid_methods}"
        )

    return MutationParams(
        mutation_type=method_map[mutation_method],
        probability=probability,
        gene_probability=gene_probability,
        max_genes=max_genes,
        position_delta=position_delta,
        color_delta=color_delta,
        alpha_delta=alpha_delta,
        field_probability=field_probability,
        guided_ratio=guided_ratio,
    )
