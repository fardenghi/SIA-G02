"""CLI del VAE (Ej2): entrena el autoencoder variacional a partir de un `config.json`.

    uv run autoencoder-vae --config configs/vae/base.json

Carga el dataset de emojis (Ej2a), entrena el VAE con β-warmup (Ej2b) y exporta métricas.
La generación de muestras y las visualizaciones (Ej2c) se agregan en la fase de viz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .emoji_data import augment_dataset, load_emojis
from .vae import VAE
from .vae_config import load_vae_config
from .vae_train import VAEMetricsTracker, train_vae


def run(config_path: str) -> dict:
    cfg = load_vae_config(config_path)
    plots_dir = Path(cfg.output.plots_dir) / cfg.name

    print(f"== VAE: {cfg.name} ==")
    print(f"  arquitectura: encoder {cfg.architecture.encoder_layers} "
          f"latent={cfg.architecture.latent_dim} act={cfg.architecture.activation} "
          f"salida={cfg.architecture.output_activation} init={cfg.architecture.init}")
    print(f"  training: loss={cfg.training.loss} epochs={cfg.training.epochs} "
          f"lr={cfg.training.lr} beta={cfg.training.beta} "
          f"warmup={cfg.training.beta_warmup}")

    # Datos (Ej2a)
    X, labels = load_emojis(size=cfg.data.size, subset=cfg.data.subset,
                            font_path=cfg.data.font_path)
    if cfg.data.augment.enabled:
        import numpy as np
        X, labels = augment_dataset(
            X, labels, size=cfg.data.size, n_aug=cfg.data.augment.n_aug,
            rng=np.random.default_rng(cfg.data.augment.seed),
            max_rot=cfg.data.augment.max_rot, max_shift=cfg.data.augment.max_shift,
            max_zoom=cfg.data.augment.max_zoom)
    print(f"  dataset: {X.shape[0]} muestras de {X.shape[1]} px "
          f"({cfg.data.size}×{cfg.data.size})")

    # Modelo (Ej2b)
    vae = VAE(
        encoder_layers=cfg.architecture.encoder_layers,
        latent_dim=cfg.architecture.latent_dim,
        activation=cfg.architecture.activation,
        output_activation=cfg.architecture.output_activation,
        init=cfg.architecture.init,
        loss=cfg.training.loss,
        seed=cfg.training.seed,
        decoder_layers=cfg.architecture.decoder_layers,
    )

    # Entrenamiento
    tracker = VAEMetricsTracker(run_label=cfg.name)
    final = train_vae(
        vae, X, epochs=cfg.training.epochs, lr=cfg.training.lr,
        beta=cfg.training.beta, beta_warmup=cfg.training.beta_warmup,
        seed=cfg.training.seed, tracker=tracker, log_every=cfg.training.log_every,
    )
    tracker.to_csv(cfg.output.metrics_csv)
    print(f"  -> ELBO={final['elbo']:.4f} recon={final['recon']:.4f} "
          f"kl={final['kl']:.4f} recon_det={final['recon_det']:.4f}")
    print(f"  -> métricas: {cfg.output.metrics_csv}")
    print(f"  -> plots: {plots_dir}")

    return {"name": cfg.name, **final, "_vae": vae, "_X": X, "_labels": labels,
            "_plots_dir": plots_dir, "_cfg": cfg}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VAE TP5 (ejercicio 2)")
    parser.add_argument("--config", required=True, help="Ruta al config.json")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
