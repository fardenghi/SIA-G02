"""
Configuración e hiperparámetros.

Manejo de parámetros del algoritmo genético y configuración general.
Soporta archivos YAML con override por argumentos CLI.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict

from src.genetic.engine import EvolutionConfig
from src.genetic.mutation import (
    MutationParams,
    create_mutation_params,
    AdaptiveSigma,
    AdaptiveSigmaConfig,
)


def _legacy_error_threshold_to_fitness(error_threshold: Any) -> Optional[float]:
    """Convierte umbral legacy de error (MSE) a umbral de fitness."""
    if error_threshold is None:
        return None

    value = float(error_threshold)
    return 1.0 / (1.0 + value)


@dataclass
class FitnessConfig:
    """Configuración de la función de fitness."""

    # Método: "linear", "rmse", "inverse_normalized", "exponential",
    #         "inverse_mse", "detail_weighted", "composite"
    method: str = "linear"
    # Escala para método exponencial: fitness = exp(-MSE_norm / scale)
    exponential_scale: float = 0.1
    # Peso base para regiones lisas en detail_weighted (0.0–1.0).
    # 0.0 = solo pesan los bordes; 1.0 = equivalente a MSE uniforme.
    detail_weight_base: float = 0.3
    # Pesos para fitness compuesto: 1 - (α·(1−SSIM) + β·MSE_norm + γ·EdgeLoss) / (α+β+γ)
    composite_alpha: float = 0.5  # peso de (1 - SSIM)
    composite_beta: float = 0.2  # peso de MSE normalizado
    composite_gamma: float = 0.3  # peso de EdgeLoss


@dataclass
class SelectionConfig:
    """Configuración de selección."""

    method: str = "tournament"
    tournament_size: int = 3
    threshold: float = 0.75
    boltzmann_t0: float = 100.0
    boltzmann_tc: float = 1.0
    boltzmann_k: float = 0.005


@dataclass
class CrossoverConfig:
    """Configuración de cruza."""

    method: str = "single_point"
    probability: float = 0.8
    phased_enabled: bool = False
    early_method: Optional[str] = None
    late_method: str = "uniform"
    switch_ratio: float = 0.6


@dataclass
class MutationConfig:
    """Configuración de mutación."""

    method: str = "uniform_multigen"
    probability: float = 0.3
    gene_probability: float = 0.1
    max_genes: int = 3
    position_delta: float = 0.1
    color_delta: int = 30
    alpha_delta: float = 0.1
    # Probabilidad de mutar cada float individual dentro del triángulo.
    # 1.0 = todos los campos mutan siempre (default). < 1.0 = per-float.
    field_probability: float = 1.0
    # Fracción de mutaciones guiadas (0-1). Solo para error_map_guided.
    # El resto (1 - guided_ratio) son mutaciones uniformes aleatorias.
    guided_ratio: float = 0.75

    # Sigma adaptativo: escala los deltas según el progreso real del fitness
    adaptive_sigma_enabled: bool = False
    adaptive_sigma_scale_min: float = 0.1
    adaptive_sigma_scale_max: float = 1.0
    adaptive_sigma_decay_factor: float = 0.9
    adaptive_sigma_recovery_factor: float = 1.1
    adaptive_sigma_window: int = 15
    adaptive_sigma_min_improvement: float = 1e-4

    def to_params(self) -> MutationParams:
        """Convierte a MutationParams."""
        return create_mutation_params(
            mutation_method=self.method,
            probability=self.probability,
            gene_probability=self.gene_probability,
            max_genes=self.max_genes,
            position_delta=self.position_delta,
            color_delta=self.color_delta,
            alpha_delta=self.alpha_delta,
            field_probability=self.field_probability,
            guided_ratio=self.guided_ratio,
        )

    def to_adaptive_sigma(self) -> AdaptiveSigma | None:
        """Crea un AdaptiveSigma si está habilitado, None si no."""
        if not self.adaptive_sigma_enabled:
            return None
        cfg = AdaptiveSigmaConfig(
            scale_min=self.adaptive_sigma_scale_min,
            scale_max=self.adaptive_sigma_scale_max,
            decay_factor=self.adaptive_sigma_decay_factor,
            recovery_factor=self.adaptive_sigma_recovery_factor,
            stagnation_window=self.adaptive_sigma_window,
            min_improvement=self.adaptive_sigma_min_improvement,
        )
        return AdaptiveSigma(cfg)


@dataclass
class OutputConfig:
    """Configuración de salida."""

    directory: str = "output"
    save_interval: int = 100
    log_interval: int = 10
    export_triangles: bool = True
    plot_fitness: bool = True
    export_metrics_csv: bool = True  # métricas por generación (pandas)
    export_triangles_csv: bool = False  # enumeración de triángulos en CSV


@dataclass
class SurvivalConfig:
    """Configuración de supervivencia."""

    method: str = "exclusive"  # "additive" o "exclusive"
    selection_method: str = "elite"  # método para seleccionar sobrevivientes
    offspring_ratio: float = 1.0  # K = N * offspring_ratio
    elite_count: int = 0  # individuos élite que siempre pasan a la siguiente gen


@dataclass
class IslandConfig:
    """Configuración del modelo de islas (IMGA)."""

    enabled: bool = False
    num_islands: int = 4  # K: cantidad de islas
    migration_size: int = 5  # M: mejores M + peores M migran
    migration_interval: int = 10  # T: migrar cada T generaciones
    topology: str = "ring"  # topología de migración
    parallel: bool = True  # True = multiprocessing, False = secuencial


@dataclass
class RenderingConfig:
    """Backend de renderizado: 'cpu' (Pillow) o 'gpu' (moderngl/OpenGL)."""

    backend: str = "cpu"


@dataclass
class Config:
    """
    Configuración completa del sistema.

    Agrupa todas las configuraciones parciales y proporciona
    métodos para cargar desde archivo y combinar con CLI.
    """

    # Imagen objetivo
    target_path: Optional[str] = None

    # Parámetros del genotipo
    num_triangles: int = 50
    shape_type: str = "triangle"
    alpha_min: float = 0.1
    alpha_max: float = 0.8

    # Algoritmo genético
    population_size: int = 100
    max_generations: int = 5000
    fitness_threshold: Optional[float] = None
    stagnation_threshold: float = 0.0005
    max_patience: int = 20
    transition_methods: Optional[list] = None
    seed_ratio: float = 0.0  # Fracción [0–1] de la pob. inicial sembrada por grilla

    # Fitness
    fitness: FitnessConfig = field(default_factory=FitnessConfig)

    # Operadores
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    crossover: CrossoverConfig = field(default_factory=CrossoverConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)

    # Salida
    output: OutputConfig = field(default_factory=OutputConfig)

    # Islas (IMGA)
    island: IslandConfig = field(default_factory=IslandConfig)

    # Renderizado
    rendering: RenderingConfig = field(default_factory=RenderingConfig)

    def to_evolution_config(self) -> EvolutionConfig:
        """Convierte a EvolutionConfig para el motor."""
        phased_early_method = self.crossover.early_method or self.crossover.method
        return EvolutionConfig(
            population_size=self.population_size,
            num_triangles=self.num_triangles,
            shape_type=self.shape_type,
            max_generations=self.max_generations,
            fitness_threshold=self.fitness_threshold,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            elite_count=self.survival.elite_count,
            stagnation_threshold=self.stagnation_threshold,
            max_patience=self.max_patience,
            transition_methods=self.transition_methods,
            seed_ratio=self.seed_ratio,
            phased_crossover_enabled=self.crossover.phased_enabled,
            phased_crossover_early_method=phased_early_method,
            phased_crossover_late_method=self.crossover.late_method,
            phased_crossover_switch_ratio=self.crossover.switch_ratio,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        """
        Crea configuración desde diccionario.

        Args:
            data: Diccionario con la configuración.

        Returns:
            Instancia de Config.
        """
        # Extraer secciones anidadas
        fitness_data = data.pop("fitness", {})
        selection_data = data.pop("selection", {})
        crossover_data = data.pop("crossover", {})
        crossover_phased_data = (
            crossover_data.pop("phased", {}) if isinstance(crossover_data, dict) else {}
        )
        mutation_data = data.pop("mutation", {})
        survival_data = data.pop("survival", {})
        output_data = data.pop("output", {})
        rendering_data = data.pop("rendering", {})
        island_data = data.pop("island", {})

        # Extraer de secciones legacy si existen
        if "fitness_threshold" not in data and "error_threshold" in data:
            # Compatibilidad hacia atrás
            data["fitness_threshold"] = _legacy_error_threshold_to_fitness(
                data.get("error_threshold")
            )

        if "image" in data:
            image_data = data.pop("image")
            if "target_path" in image_data:
                data["target_path"] = image_data["target_path"]

        if "genotype" in data:
            genotype_data = data.pop("genotype")
            data.setdefault("num_triangles", genotype_data.get("num_triangles", 50))
            data.setdefault(
                "shape_type",
                genotype_data.get("shape_type", genotype_data.get("shape", "triangle")),
            )
            data.setdefault("alpha_min", genotype_data.get("alpha_min", 0.1))
            data.setdefault("alpha_max", genotype_data.get("alpha_max", 0.8))

        if "genetic" in data:
            genetic_data = data.pop("genetic")
            data.setdefault("population_size", genetic_data.get("population_size", 100))
            data.setdefault(
                "max_generations", genetic_data.get("max_generations", 5000)
            )
            if "fitness_threshold" in genetic_data:
                fitness_threshold = genetic_data.get("fitness_threshold")
            else:
                fitness_threshold = _legacy_error_threshold_to_fitness(
                    genetic_data.get("error_threshold")
                )
            data.setdefault("fitness_threshold", fitness_threshold)
            data.setdefault(
                "stagnation_threshold", genetic_data.get("stagnation_threshold", 0.0005)
            )
            data.setdefault("max_patience", genetic_data.get("max_patience", 20))
            data.setdefault("seed_ratio", genetic_data.get("seed_ratio", 0.0))

            t_methods = genetic_data.get("transition_methods")
            if isinstance(t_methods, str):
                t_methods = [t_methods]

            data.setdefault("transition_methods", t_methods)

        return cls(
            target_path=data.get("target_path"),
            num_triangles=data.get("num_triangles", 50),
            shape_type=data.get("shape_type", data.get("shape", "triangle")),
            alpha_min=data.get("alpha_min", 0.1),
            alpha_max=data.get("alpha_max", 0.8),
            population_size=data.get("population_size", 100),
            max_generations=data.get("max_generations", 5000),
            fitness_threshold=data.get("fitness_threshold"),
            stagnation_threshold=data.get("stagnation_threshold", 0.0005),
            max_patience=data.get("max_patience", 20),
            transition_methods=data.get("transition_methods", None),
            seed_ratio=data.get("seed_ratio", 0.0),
            fitness=FitnessConfig(**fitness_data) if fitness_data else FitnessConfig(),
            selection=SelectionConfig(**selection_data)
            if selection_data
            else SelectionConfig(),
            crossover=CrossoverConfig(
                **crossover_data,
                phased_enabled=crossover_phased_data.get("enabled", False),
                early_method=crossover_phased_data.get("early_method"),
                late_method=crossover_phased_data.get("late_method", "uniform"),
                switch_ratio=crossover_phased_data.get("switch_ratio", 0.6),
            )
            if crossover_data
            else CrossoverConfig(
                phased_enabled=crossover_phased_data.get("enabled", False),
                early_method=crossover_phased_data.get("early_method"),
                late_method=crossover_phased_data.get("late_method", "uniform"),
                switch_ratio=crossover_phased_data.get("switch_ratio", 0.6),
            ),
            mutation=MutationConfig(**mutation_data)
            if mutation_data
            else MutationConfig(),
            survival=SurvivalConfig(**survival_data)
            if survival_data
            else SurvivalConfig(),
            output=OutputConfig(**output_data) if output_data else OutputConfig(),
            island=IslandConfig(**island_data) if island_data else IslandConfig(),
            rendering=RenderingConfig(**rendering_data)
            if rendering_data
            else RenderingConfig(),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """
        Carga configuración desde archivo YAML.

        Args:
            path: Ruta al archivo YAML.

        Returns:
            Instancia de Config.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    def merge_cli_args(self, **kwargs) -> Config:
        """
        Combina con argumentos de línea de comandos.

        Los argumentos CLI tienen prioridad sobre el archivo.

        Args:
            **kwargs: Argumentos de CLI.

        Returns:
            Nueva instancia con valores combinados.
        """
        # Crear copia de la configuración actual
        import copy

        new_config = copy.deepcopy(self)

        # Aplicar overrides
        if kwargs.get("image"):
            new_config.target_path = kwargs["image"]

        if kwargs.get("triangles"):
            new_config.num_triangles = kwargs["triangles"]

        if kwargs.get("shape"):
            new_config.shape_type = kwargs["shape"]

        if kwargs.get("population"):
            new_config.population_size = kwargs["population"]

        if kwargs.get("generations"):
            new_config.max_generations = kwargs["generations"]

        if kwargs.get("mutation_rate"):
            new_config.mutation.probability = kwargs["mutation_rate"]

        if kwargs.get("output"):
            new_config.output.directory = kwargs["output"]

        if kwargs.get("save_interval"):
            new_config.output.save_interval = kwargs["save_interval"]

        if kwargs.get("selection"):
            new_config.selection.method = kwargs["selection"]

        if kwargs.get("crossover"):
            new_config.crossover.method = kwargs["crossover"]

        if kwargs.get("mutation_method"):
            new_config.mutation.method = kwargs["mutation_method"]

        if kwargs.get("guided_ratio") is not None:
            new_config.mutation.guided_ratio = kwargs["guided_ratio"]

        if kwargs.get("field_probability") is not None:
            new_config.mutation.field_probability = kwargs["field_probability"]

        if kwargs.get("survival_method"):
            new_config.survival.method = kwargs["survival_method"]

        if kwargs.get("offspring_ratio"):
            new_config.survival.offspring_ratio = kwargs["offspring_ratio"]

        if kwargs.get("elite_count") is not None:
            new_config.survival.elite_count = kwargs["elite_count"]

        if kwargs.get("fitness_method"):
            new_config.fitness.method = kwargs["fitness_method"]

        if kwargs.get("fitness_scale") is not None:
            new_config.fitness.exponential_scale = kwargs["fitness_scale"]

        if kwargs.get("renderer"):
            new_config.rendering.backend = kwargs["renderer"]

        if kwargs.get("islands") is not None:
            new_config.island.num_islands = kwargs["islands"]
            new_config.island.enabled = kwargs["islands"] > 1

        if kwargs.get("migration_size") is not None:
            new_config.island.migration_size = kwargs["migration_size"]

        if kwargs.get("migration_interval") is not None:
            new_config.island.migration_interval = kwargs["migration_interval"]

        return new_config

    def validate(self) -> list[str]:
        """
        Valida la configuración.

        Returns:
            Lista de errores encontrados (vacía si válida).
        """
        errors = []

        if not self.target_path:
            errors.append("Se requiere una imagen objetivo (--image)")
        elif not Path(self.target_path).exists():
            errors.append(f"Imagen no encontrada: {self.target_path}")

        if self.num_triangles < 1:
            errors.append("num_triangles debe ser al menos 1")

        if self.shape_type not in {"triangle", "ellipse"}:
            errors.append("shape_type debe ser 'triangle' o 'ellipse'")

        if self.population_size < 2:
            errors.append("population_size debe ser al menos 2")

        if self.max_generations < 1:
            errors.append("max_generations debe ser al menos 1")

        if not 0 <= self.alpha_min <= 1:
            errors.append("alpha_min debe estar en [0, 1]")

        if not 0 <= self.alpha_max <= 1:
            errors.append("alpha_max debe estar en [0, 1]")

        if self.alpha_min > self.alpha_max:
            errors.append("alpha_min no puede ser mayor que alpha_max")

        if self.fitness_threshold is not None:
            if not 0 < self.fitness_threshold <= 1:
                errors.append("fitness_threshold debe estar en (0, 1]")

        from src.fitness.mse import FITNESS_METHODS
        from src.genetic.crossover import create_crossover_method

        if self.fitness.method not in FITNESS_METHODS:
            errors.append(
                f"fitness.method debe ser uno de: {', '.join(sorted(FITNESS_METHODS))}"
            )

        try:
            create_crossover_method(self.crossover.method, self.crossover.probability)
        except ValueError as exc:
            errors.append(str(exc))

        if self.crossover.phased_enabled:
            early_method = self.crossover.early_method or self.crossover.method
            try:
                create_crossover_method(early_method, self.crossover.probability)
            except ValueError as exc:
                errors.append(f"crossover.early_method inválido: {exc}")

            try:
                create_crossover_method(
                    self.crossover.late_method, self.crossover.probability
                )
            except ValueError as exc:
                errors.append(f"crossover.late_method inválido: {exc}")

            if not 0 <= self.crossover.switch_ratio <= 1:
                errors.append("crossover.switch_ratio debe estar en [0, 1]")

        # Validación de islas
        if self.island.enabled:
            if self.island.num_islands < 2:
                errors.append("island.num_islands debe ser >= 2 cuando islas están habilitadas")
            if self.island.migration_size < 1:
                errors.append("island.migration_size debe ser >= 1")
            if self.island.migration_interval < 1:
                errors.append("island.migration_interval debe ser >= 1")
            if self.island.topology not in {"ring"}:
                errors.append("island.topology debe ser 'ring'")
            if self.island.migration_size * 2 >= self.population_size:
                errors.append(
                    "island.migration_size * 2 debe ser < population_size"
                )

        return errors


def load_config(config_path: Optional[str] = None, **cli_args) -> Config:
    """
    Carga configuración combinando archivo YAML y argumentos CLI.

    Args:
        config_path: Ruta al archivo de configuración (opcional).
        **cli_args: Argumentos de línea de comandos.

    Returns:
        Configuración combinada.
    """
    # Cargar desde archivo si existe
    if config_path and Path(config_path).exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    # Aplicar overrides de CLI
    if cli_args:
        config = config.merge_cli_args(**cli_args)

    return config
