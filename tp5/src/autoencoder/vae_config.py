"""Carga y validación del JSON de configuración del VAE (Ej2).

Independiente del `config.py` del Ej1, pero reutiliza sus constantes de dominio
(`VALID_ACTIVATIONS`, etc.) y la excepción `ConfigError` (import de solo lectura, sin
modificarlo). Secciones: `data`, `architecture`, `training`, `output`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    VALID_ACTIVATIONS,
    VALID_INITS,
    VALID_LOSSES,
    VALID_OUTPUT_ACTIVATIONS,
    ConfigError,
)


def _require_in(value, domain, key):
    if value not in domain:
        raise ConfigError(
            f"Valor inválido para '{key}': {value!r}. Permitidos: {sorted(domain)}")


def _unknown_keys(section: dict, allowed: set[str], section_name: str):
    extra = set(section) - allowed
    if extra:
        raise ConfigError(f"Claves desconocidas en '{section_name}': {sorted(extra)}")


@dataclass
class AugmentConfig:
    enabled: bool = False
    n_aug: int = 4
    max_rot: float = 15.0
    max_shift: float = 2.0
    max_zoom: float = 0.12
    seed: int = 0


@dataclass
class VAEDataConfig:
    size: int = 28
    subset: list[int] | None = None
    font_path: str = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class VAEArchConfig:
    kind: str = "mlp"  # "mlp" | "conv"
    encoder_layers: list[int] = field(default_factory=lambda: [784, 256, 64])
    latent_dim: int = 2
    activation: str = "relu"
    output_activation: str = "sigmoid"
    init: str = "xavier_normal"
    decoder_layers: list[int] | None = None
    # Específicos de kind="conv" (ignorados para "mlp"):
    conv_channels: list[int] = field(default_factory=lambda: [16, 32])
    dense_hidden: int = 64


@dataclass
class VAETrainConfig:
    loss: str = "bce"
    epochs: int = 3000
    lr: float = 1e-3
    beta: float = 1.0
    beta_warmup: int = 0
    seed: int = 0
    log_every: int = 100


@dataclass
class VAEOutputConfig:
    metrics_csv: str = "out/vae/metrics.csv"
    plots_dir: str = "out/plots"


@dataclass
class VAEConfig:
    data: VAEDataConfig
    architecture: VAEArchConfig
    training: VAETrainConfig
    output: VAEOutputConfig
    name: str = "vae"


def load_vae_config(path: str | Path) -> VAEConfig:
    """Carga, valida y construye la `VAEConfig` desde un archivo JSON."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No existe el archivo de config: {path}")
    raw = json.loads(path.read_text())

    # -- data --
    data_raw = raw.get("data", {})
    _unknown_keys(data_raw, {"size", "subset", "font_path", "augment"}, "data")
    size = int(data_raw.get("size", 28))
    if size <= 0:
        raise ConfigError("'data.size' debe ser > 0")
    aug_raw = data_raw.get("augment", {})
    _unknown_keys(aug_raw, {"enabled", "n_aug", "max_rot", "max_shift", "max_zoom",
                            "seed"}, "data.augment")
    augment = AugmentConfig(
        enabled=bool(aug_raw.get("enabled", False)),
        n_aug=int(aug_raw.get("n_aug", 4)),
        max_rot=float(aug_raw.get("max_rot", 15.0)),
        max_shift=float(aug_raw.get("max_shift", 2.0)),
        max_zoom=float(aug_raw.get("max_zoom", 0.12)),
        seed=int(aug_raw.get("seed", 0)),
    )
    if augment.n_aug < 0:
        raise ConfigError("'data.augment.n_aug' debe ser >= 0")
    data = VAEDataConfig(
        size=size,
        subset=data_raw.get("subset", None),
        font_path=data_raw.get("font_path",
                               "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
        augment=augment,
    )

    # -- architecture --
    arch_raw = raw.get("architecture", {})
    _unknown_keys(arch_raw, {"kind", "encoder_layers", "latent_dim", "activation",
                             "output_activation", "init", "decoder_layers",
                             "conv_channels", "dense_hidden"}, "architecture")
    kind = arch_raw.get("kind", "mlp")
    if kind not in ("mlp", "conv"):
        raise ConfigError(
            f"'architecture.kind' inválido: {kind!r}. Permitidos: ['conv', 'mlp']")

    latent_dim = int(arch_raw.get("latent_dim", 2))
    if latent_dim < 1:
        raise ConfigError("'architecture.latent_dim' debe ser >= 1")

    # Defaults; cada rama valida lo que le corresponde.
    encoder_layers = arch_raw.get("encoder_layers", [size * size, 256, 64])
    decoder_layers = None
    conv_channels = arch_raw.get("conv_channels", [16, 32])
    dense_hidden = int(arch_raw.get("dense_hidden", 64))

    if kind == "mlp":
        if "encoder_layers" not in arch_raw:
            raise ConfigError("Falta 'architecture.encoder_layers' (requerido para kind=mlp)")
        if (not isinstance(encoder_layers, list) or len(encoder_layers) < 1
                or not all(isinstance(n, int) and n > 0 for n in encoder_layers)):
            raise ConfigError("'architecture.encoder_layers' debe ser lista de enteros > 0")
        if encoder_layers[0] != size * size:
            raise ConfigError(
                f"'architecture.encoder_layers[0]' debe ser {size * size} "
                f"(= data.size² = {size}²), se recibió {encoder_layers[0]}")
        decoder_layers = arch_raw.get("decoder_layers", None)
        if decoder_layers is not None:
            if (not isinstance(decoder_layers, list) or len(decoder_layers) < 2
                    or not all(isinstance(n, int) and n > 0 for n in decoder_layers)):
                raise ConfigError(
                    "'architecture.decoder_layers' debe ser lista de enteros > 0 o null")
            if decoder_layers[0] != latent_dim:
                raise ConfigError(
                    "'architecture.decoder_layers[0]' debe igualar latent_dim "
                    f"({latent_dim})")
            if decoder_layers[-1] != size * size:
                raise ConfigError(
                    f"'architecture.decoder_layers[-1]' debe ser {size * size} (dim de salida)")
    else:  # kind == "conv"
        if "decoder_layers" in arch_raw:
            raise ConfigError("'architecture.decoder_layers' no aplica a kind=conv")
        if (not isinstance(conv_channels, list) or len(conv_channels) < 1
                or not all(isinstance(c, int) and c > 0 for c in conv_channels)):
            raise ConfigError(
                "'architecture.conv_channels' debe ser lista no vacía de enteros > 0")
        if dense_hidden <= 0:
            raise ConfigError("'architecture.dense_hidden' debe ser > 0")
        # Compatibilidad espacial: size/2ⁿ·2ⁿ debe recuperar size (n = nº de convs stride-2).
        feat = size
        for _ in conv_channels:
            feat = (feat + 2 * 1 - 3) // 2 + 1  # conv_out_size con k3, s2, p1
        if feat * (2 ** len(conv_channels)) != size:
            raise ConfigError(
                f"'data.size'={size} incompatible con {len(conv_channels)} capas conv: "
                f"el decoder produce {feat * (2 ** len(conv_channels))}≠{size}")

    architecture = VAEArchConfig(
        kind=kind,
        encoder_layers=encoder_layers,
        latent_dim=latent_dim,
        activation=arch_raw.get("activation", "relu"),
        output_activation=arch_raw.get("output_activation", "sigmoid"),
        init=arch_raw.get("init", "xavier_normal"),
        decoder_layers=decoder_layers,
        conv_channels=conv_channels,
        dense_hidden=dense_hidden,
    )
    _require_in(architecture.activation, VALID_ACTIVATIONS, "architecture.activation")
    _require_in(architecture.output_activation, VALID_OUTPUT_ACTIVATIONS,
                "architecture.output_activation")
    _require_in(architecture.init, VALID_INITS, "architecture.init")

    # -- training --
    train_raw = raw.get("training", {})
    _unknown_keys(train_raw, {"loss", "epochs", "lr", "beta", "beta_warmup", "seed",
                              "log_every"}, "training")
    training = VAETrainConfig(
        loss=train_raw.get("loss", "bce"),
        epochs=int(train_raw.get("epochs", 3000)),
        lr=float(train_raw.get("lr", 1e-3)),
        beta=float(train_raw.get("beta", 1.0)),
        beta_warmup=int(train_raw.get("beta_warmup", 0)),
        seed=int(train_raw.get("seed", 0)),
        log_every=int(train_raw.get("log_every", 100)),
    )
    _require_in(training.loss, VALID_LOSSES, "training.loss")
    if training.epochs <= 0:
        raise ConfigError("'training.epochs' debe ser > 0")
    if training.lr <= 0:
        raise ConfigError("'training.lr' debe ser > 0")
    if training.beta < 0:
        raise ConfigError("'training.beta' debe ser >= 0")
    if training.beta_warmup < 0:
        raise ConfigError("'training.beta_warmup' debe ser >= 0")

    # -- output --
    out_raw = raw.get("output", {})
    _unknown_keys(out_raw, {"metrics_csv", "plots_dir"}, "output")
    output = VAEOutputConfig(
        metrics_csv=out_raw.get("metrics_csv", "out/vae/metrics.csv"),
        plots_dir=out_raw.get("plots_dir", "out/plots"),
    )

    return VAEConfig(
        data=data,
        architecture=architecture,
        training=training,
        output=output,
        name=raw.get("name", path.stem),
    )
