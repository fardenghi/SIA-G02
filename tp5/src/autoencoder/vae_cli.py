"""CLI del VAE (Ej2): entrena el autoencoder variacional a partir de un `config.json`.

    uv run autoencoder-vae --config configs/vae/base.json

Carga el dataset de emojis (Ej2a), entrena el VAE con β-warmup (Ej2b) y exporta métricas.
La generación de muestras y las visualizaciones (Ej2c) se agregan en la fase de viz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import vae_metrics_viz, vae_viz
from .conv_vae import ConvVAE
from .emoji_data import augment_dataset, load_emojis
from .vae import VAE
from .vae_config import load_vae_config
from .vae_train import VAEMetricsTracker, train_vae


def build_model(cfg):
    """Construye el modelo según `architecture.kind` (mlp → VAE, conv → ConvVAE).

    Ambos exponen la misma interfaz (encode/decode/forward/elbo/backward/sample_prior/…),
    así que el resto del pipeline (entrenamiento, viz, diagnósticos) es idéntico.
    """
    arch = cfg.architecture
    if arch.kind == "conv":
        return ConvVAE(
            size=cfg.data.size, latent_dim=arch.latent_dim,
            conv_channels=arch.conv_channels, dense_hidden=arch.dense_hidden,
            activation=arch.activation, output_activation=arch.output_activation,
            init=arch.init, loss=cfg.training.loss, seed=cfg.training.seed,
        )
    return VAE(
        encoder_layers=arch.encoder_layers, latent_dim=arch.latent_dim,
        activation=arch.activation, output_activation=arch.output_activation,
        init=arch.init, loss=cfg.training.loss, seed=cfg.training.seed,
        decoder_layers=arch.decoder_layers,
    )


def run(config_path: str) -> dict:
    cfg = load_vae_config(config_path)
    plots_dir = Path(cfg.output.plots_dir) / cfg.name

    arch = cfg.architecture
    print(f"== VAE: {cfg.name} ==")
    if arch.kind == "conv":
        print(f"  arquitectura: conv canales={arch.conv_channels} "
              f"dense={arch.dense_hidden} latent={arch.latent_dim} "
              f"act={arch.activation} salida={arch.output_activation} init={arch.init}")
    else:
        print(f"  arquitectura: mlp encoder {arch.encoder_layers} "
              f"latent={arch.latent_dim} act={arch.activation} "
              f"salida={arch.output_activation} init={arch.init}")
    print(f"  training: loss={cfg.training.loss} epochs={cfg.training.epochs} "
          f"lr={cfg.training.lr} beta={cfg.training.beta} "
          f"warmup={cfg.training.beta_warmup}")

    # Datos (Ej2a)
    X, labels = load_emojis(size=cfg.data.size, subset=cfg.data.subset,
                            font_path=cfg.data.font_path)
    if cfg.data.augment.enabled:
        X, labels = augment_dataset(
            X, labels, size=cfg.data.size, n_aug=cfg.data.augment.n_aug,
            rng=np.random.default_rng(cfg.data.augment.seed),
            max_rot=cfg.data.augment.max_rot, max_shift=cfg.data.augment.max_shift,
            max_zoom=cfg.data.augment.max_zoom)
    print(f"  dataset: {X.shape[0]} muestras de {X.shape[1]} px "
          f"({cfg.data.size}×{cfg.data.size})")

    # Modelo (Ej2b) — mlp o conv según architecture.kind
    vae = build_model(cfg)

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

    # Visualizaciones y generación (Ej2c)
    size = cfg.data.size
    mu, _ = vae.encode(X)
    # Reconstrucción determinista (z = μ).
    x_hat = vae.decode(mu)
    vae_viz.plot_reconstruction_gray(X, x_hat, labels, size=size, n=min(16, X.shape[0]),
                                     path=plots_dir / "reconstruction.png")
    if vae.latent_dim > 2:
        # Latente >2D: proyectar las medias a sus 2 componentes principales para el scatter.
        proj, var_ratio = vae_metrics_viz.project_pca(mu, k=2)
        pct = 100.0 * float(var_ratio.sum())
        vae_viz.plot_latent_means(
            proj, labels, path=plots_dir / "latent_means.png",
            title=f"Espacio latente del VAE (proyección PCA, var explicada {pct:.0f}%)")
    else:
        vae_viz.plot_latent_means(mu, labels, path=plots_dir / "latent_means.png")
    if vae.latent_dim == 2:
        vae_viz.plot_latent_manifold(vae, size, path=plots_dir / "manifold.png")
    # Muestras nuevas desde el prior (Ej2c).
    rng = np.random.default_rng(cfg.training.seed)
    vae_viz.plot_samples(vae, size, n=16, rng=rng, path=plots_dir / "samples.png")
    # Interpolación entre dos emojis.
    if X.shape[0] >= 2:
        vae_viz.plot_interpolation(vae, X[0], X[X.shape[0] // 2], size,
                                   path=plots_dir / "interpolation.png")

    # Diagnósticos del encoder (métricas)
    vae_metrics_viz.plot_training_curves(tracker.df,
                                         path=plots_dir / "training_curves.png")
    vae_metrics_viz.plot_kl_per_dim(vae, X, path=plots_dir / "kl_per_dim.png")
    vae_metrics_viz.plot_posterior_stats(vae, X, path=plots_dir / "posterior_stats.png")
    if vae.latent_dim == 2:
        vae_metrics_viz.plot_aggregate_posterior(
            vae, X, path=plots_dir / "aggregate_posterior.png")
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
