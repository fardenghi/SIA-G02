"""Carga y validación del JSON de configuración del experimento.

Secciones: `data`, `architecture`, `training`, `denoising`, `output`. Ante una clave
faltante o un valor fuera de dominio se levanta `ConfigError` antes de entrenar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """Error de configuración (clave faltante o valor fuera de dominio)."""


VALID_ACTIVATIONS = {"tanh", "relu", "sigmoid"}
VALID_OUTPUT_ACTIVATIONS = {"sigmoid", "tanh", "linear"}
VALID_INITS = {"xavier_uniform", "xavier_normal", "he_uniform", "he_normal",
               "normal", "uniform"}
VALID_OPTIMIZERS = {"adam", "lbfgs"}
VALID_LOSSES = {"bce", "mse"}
VALID_NOISE = {"salt_pepper", "bit_flip"}


@dataclass
class DataConfig:
    font_path: str = "font/font.h"
    subset: list[int] | None = None


@dataclass
class ArchitectureConfig:
    encoder_layers: list[int] = field(default_factory=lambda: [35, 25, 15, 8, 2])
    activation: str = "tanh"
    output_activation: str = "sigmoid"
    init: str = "xavier_normal"


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    loss: str = "bce"
    epochs: int = 20000
    lr: float = 1e-3
    restarts: int = 20
    seed: int = 42
    log_every: int = 100
    stop_at: int | None = 0


@dataclass
class DenoisingConfig:
    enabled: bool = False
    noise_type: str = "salt_pepper"
    level: float = 0.1
    sweep_levels: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])


@dataclass
class OutputConfig:
    metrics_csv: str = "out/metrics.csv"
    plots_dir: str = "out/plots"


@dataclass
class Config:
    data: DataConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    denoising: DenoisingConfig
    output: OutputConfig
    name: str = "experiment"


def _require_in(value, domain, key):
    if value not in domain:
        raise ConfigError(
            f"Valor inválido para '{key}': {value!r}. Permitidos: {sorted(domain)}"
        )


def _unknown_keys(section: dict, allowed: set[str], section_name: str):
    extra = set(section) - allowed
    if extra:
        raise ConfigError(
            f"Claves desconocidas en '{section_name}': {sorted(extra)}"
        )


def load_config(path: str | Path) -> Config:
    """Carga, valida y construye la `Config` desde un archivo JSON."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No existe el archivo de config: {path}")
    raw = json.loads(path.read_text())

    arch_raw = raw.get("architecture", {})
    _unknown_keys(arch_raw, {"encoder_layers", "activation", "output_activation",
                             "init"}, "architecture")
    if "encoder_layers" not in arch_raw:
        raise ConfigError("Falta 'architecture.encoder_layers'")
    encoder_layers = arch_raw["encoder_layers"]
    if (not isinstance(encoder_layers, list) or len(encoder_layers) < 2
            or not all(isinstance(n, int) and n > 0 for n in encoder_layers)):
        raise ConfigError("'architecture.encoder_layers' debe ser lista de enteros > 0")
    if encoder_layers[0] != 35:
        raise ConfigError("'architecture.encoder_layers[0]' debe ser 35 (dim de entrada)")

    architecture = ArchitectureConfig(
        encoder_layers=encoder_layers,
        activation=arch_raw.get("activation", "tanh"),
        output_activation=arch_raw.get("output_activation", "sigmoid"),
        init=arch_raw.get("init", "xavier_normal"),
    )
    _require_in(architecture.activation, VALID_ACTIVATIONS, "architecture.activation")
    _require_in(architecture.output_activation, VALID_OUTPUT_ACTIVATIONS,
                "architecture.output_activation")
    _require_in(architecture.init, VALID_INITS, "architecture.init")

    train_raw = raw.get("training", {})
    _unknown_keys(train_raw, {"optimizer", "loss", "epochs", "lr", "restarts",
                              "seed", "log_every", "stop_at"}, "training")
    stop_at_raw = train_raw.get("stop_at", 0)
    stop_at = None if stop_at_raw is None else int(stop_at_raw)
    if stop_at is not None and stop_at < 0:
        raise ConfigError("'training.stop_at' debe ser >= 0 o null")
    training = TrainingConfig(
        optimizer=train_raw.get("optimizer", "adam"),
        loss=train_raw.get("loss", "bce"),
        epochs=int(train_raw.get("epochs", 20000)),
        lr=float(train_raw.get("lr", 1e-3)),
        restarts=int(train_raw.get("restarts", 20)),
        seed=int(train_raw.get("seed", 42)),
        log_every=int(train_raw.get("log_every", 100)),
        stop_at=stop_at,
    )
    _require_in(training.optimizer, VALID_OPTIMIZERS, "training.optimizer")
    _require_in(training.loss, VALID_LOSSES, "training.loss")
    if training.epochs <= 0:
        raise ConfigError("'training.epochs' debe ser > 0")
    if training.restarts <= 0:
        raise ConfigError("'training.restarts' debe ser > 0")

    data_raw = raw.get("data", {})
    _unknown_keys(data_raw, {"font_path", "subset"}, "data")
    data = DataConfig(
        font_path=data_raw.get("font_path", "font/font.h"),
        subset=data_raw.get("subset", None),
    )

    den_raw = raw.get("denoising", {})
    _unknown_keys(den_raw, {"enabled", "noise_type", "level", "sweep_levels"},
                  "denoising")
    denoising = DenoisingConfig(
        enabled=bool(den_raw.get("enabled", False)),
        noise_type=den_raw.get("noise_type", "salt_pepper"),
        level=float(den_raw.get("level", 0.1)),
        sweep_levels=den_raw.get("sweep_levels", [0.05, 0.1, 0.2, 0.3]),
    )
    _require_in(denoising.noise_type, VALID_NOISE, "denoising.noise_type")
    if not (0.0 <= denoising.level <= 1.0):
        raise ConfigError("'denoising.level' debe estar en [0, 1]")

    out_raw = raw.get("output", {})
    _unknown_keys(out_raw, {"metrics_csv", "plots_dir"}, "output")
    output = OutputConfig(
        metrics_csv=out_raw.get("metrics_csv", "out/metrics.csv"),
        plots_dir=out_raw.get("plots_dir", "out/plots"),
    )

    return Config(
        data=data,
        architecture=architecture,
        training=training,
        denoising=denoising,
        output=output,
        name=raw.get("name", path.stem),
    )
