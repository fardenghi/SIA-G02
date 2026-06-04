"""CLI: ejecuta el experimento end-to-end a partir de un `config.json`.

    uv run autoencoder --config configs/base_adam.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import viz
from .config import load_config
from .data import add_noise, labels_for_subset, load_font, select_subset
from .losses import validate_coherence
from .train import (
    evaluate_denoising,
    max_pixel_error,
    train_multi_restart,
)


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    plots_dir = Path(cfg.output.plots_dir) / cfg.name

    print(f"== Experimento: {cfg.name} ==")
    print(f"  arquitectura: {cfg.architecture.encoder_layers} "
          f"act={cfg.architecture.activation}/{cfg.architecture.output_activation} "
          f"init={cfg.architecture.init}")
    print(f"  training: {cfg.training.optimizer} loss={cfg.training.loss} "
          f"epochs={cfg.training.epochs} restarts={cfg.training.restarts}")

    # Datos
    X_full = load_font(cfg.data.font_path)
    X = select_subset(X_full, cfg.data.subset)
    labels = labels_for_subset(cfg.data.subset)

    # Coherencia (warnings, no errores)
    validate_coherence(cfg.training.loss, cfg.architecture.output_activation,
                       cfg.architecture.activation, cfg.architecture.init)

    # Entrenamiento multi-restart
    denoising = {
        "enabled": cfg.denoising.enabled,
        "noise_type": cfg.denoising.noise_type,
        "level": cfg.denoising.level,
    }
    best_net, best_tracker, summary = train_multi_restart(
        X,
        encoder_layers=cfg.architecture.encoder_layers,
        activation=cfg.architecture.activation,
        output_activation=cfg.architecture.output_activation,
        init=cfg.architecture.init,
        loss=cfg.training.loss,
        optimizer=cfg.training.optimizer,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        restarts=cfg.training.restarts,
        seed=cfg.training.seed,
        denoising=denoising,
        log_every=cfg.training.log_every,
        stop_at=cfg.training.stop_at,
    )

    # Métricas a CSV
    best_tracker.to_csv(cfg.output.metrics_csv)
    summary_path = Path(cfg.output.metrics_csv).with_name(
        Path(cfg.output.metrics_csv).stem + "_restarts.csv")
    summary.to_csv(summary_path, index=False)

    recon = best_net.forward(X)
    best_mpe = max_pixel_error(X, recon)
    print(f"  -> mejor max_pixel_error = {best_mpe} "
          f"({int((np.asarray((recon >= 0.5).astype(int) == (X >= 0.5).astype(int)).all(axis=1)).sum())}/{X.shape[0]} patrones exactos)")
    print(f"  -> métricas: {cfg.output.metrics_csv}")

    # Visualizaciones (1a3 latente, reconstrucción, 1a4 letra nueva)
    Z = best_net.encode(X)
    # El scatter 2D (1a3) solo aplica con latente de 2 dimensiones.
    if best_net.latent_dim == 2:
        viz.plot_latent_scatter(Z, labels, path=plots_dir / "latent_scatter.png")
    viz.plot_reconstruction(X, recon, labels, path=plots_dir / "reconstruction.png")

    # Letra nueva: interpolación entre dos códigos latentes -> decode -> threshold
    if X.shape[0] >= 2:
        i, j = 0, X.shape[0] // 2
        midpoint = 0.5 * (Z[i] + Z[j])
        new_glyph = best_net.decode(midpoint.reshape(1, -1))
        viz.plot_glyph_grid(
            np.vstack([X[i], (new_glyph >= 0.5).astype(float), X[j]]),
            labels=[f"{labels[i]}", "nueva", f"{labels[j]}"],
            ncols=3,
            path=plots_dir / "new_letter.png",
        )

    result = {"name": cfg.name, "max_pixel_error": best_mpe}

    # Denoising: barrido de niveles + visualización ruidoso->recon->limpio
    if cfg.denoising.enabled:
        sweep = evaluate_denoising(
            best_net, X, cfg.denoising.noise_type,
            levels=cfg.denoising.sweep_levels, seed=cfg.training.seed,
        )
        sweep_path = Path(cfg.output.metrics_csv).with_name("denoising_sweep.csv")
        sweep.to_csv(sweep_path, index=False)
        print(f"  -> denoising sweep:\n{sweep.to_string(index=False)}")

        rng = np.random.default_rng(cfg.training.seed)
        noisy = add_noise(X, cfg.denoising.noise_type, cfg.denoising.level, rng)
        recon_noisy = best_net.forward(noisy)
        viz.plot_denoise_triplet(noisy, recon_noisy, X, labels,
                                 path=plots_dir / "denoising.png")
        result["denoising_sweep"] = sweep.to_dict(orient="records")

    print(f"  -> plots: {plots_dir}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autoencoder TP5 (ejercicio 1)")
    parser.add_argument("--config", required=True, help="Ruta al config.json")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
