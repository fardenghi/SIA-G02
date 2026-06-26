"""Experimento exploratorio: ¿cambia el veredicto MLP vs CNN al subir la resolución?

Entrena MLP-VAE y ConvVAE en grises a una resolución mayor (default 40×40) y compara
recon_det por píxel contra el baseline de 28×28 (MLP 0.395 vs CNN 0.405 a latente 8). Aísla la
variable resolución (grises, sin augment, mismo latente). El costo de la CNN escala ~con el nº
de píxeles: se acota con `--size`/`--epochs` (ver benchmark en el README).

    uv run python scripts/resolution_experiment.py --size 40 --epochs 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.conv_vae import ConvVAE  # noqa: E402
from autoencoder.emoji_data import load_emojis  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402
from autoencoder.vae_train import train_vae  # noqa: E402


def grid(rows, titles, size, path, suptitle):
    n = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.3))
    axes = np.atleast_2d(axes)
    for r, (block, name) in enumerate(zip(rows, titles)):
        for j in range(n):
            axes[r, j].imshow(block[j].reshape(size, size), cmap="Greys", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="MLP vs CNN subiendo la resolución")
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--latent", type=int, default=8)
    args = p.parse_args(argv)
    S, LAT, D = args.size, args.latent, args.size * args.size

    X, _ = load_emojis(size=S)
    print(f"resolución {S}×{S}, latente {LAT}, {args.epochs} épocas, sin augment\n")

    mlp = VAE([D, 256, 64], latent_dim=LAT, activation="relu", output_activation="sigmoid",
              init="he_normal", loss="bce", seed=0)
    fm = train_vae(mlp, X, epochs=args.epochs, lr=1e-3, beta=1.0, beta_warmup=1000, seed=0)
    cnn = ConvVAE(size=S, latent_dim=LAT, conv_channels=[16, 32], dense_hidden=64,
                  activation="relu", output_activation="sigmoid", init="he_normal",
                  loss="bce", seed=0)
    fc = train_vae(cnn, X, epochs=args.epochs, lr=1e-3, beta=1.0, beta_warmup=1000, seed=0)

    print(f"{'modelo':>6} {'recon_det':>10} {'/px':>8} {'params':>10}")
    print(f"{'MLP':>6} {fm['recon_det']:>10.1f} {fm['recon_det']/D:>8.4f} {mlp.n_params:>10}")
    print(f"{'CNN':>6} {fc['recon_det']:>10.1f} {fc['recon_det']/D:>8.4f} {cnn.n_params:>10}")
    print(f"\ngap/px CNN-MLP = {(fc['recon_det']-fm['recon_det'])/D:+.4f}  "
          f"(baseline 28×28: +0.0100)")

    out = Path("out/resolution"); out.mkdir(parents=True, exist_ok=True)
    idx = np.arange(10)
    grid([X[idx], mlp.decode(mlp.encode(X)[0])[idx], cnn.decode(cnn.encode(X)[0])[idx]],
         ["in", "MLP", "CNN"], S, out / f"recon_{S}.png",
         f"Reconstrucción {S}×{S} (latente {LAT}) — in / MLP / CNN")
    z = np.random.default_rng(7).standard_normal((10, LAT))
    grid([mlp.decode(z), cnn.decode(z)], ["MLP", "CNN"], S, out / f"samples_{S}.png",
         f"Muestras del prior {S}×{S} (latente {LAT}) — mismos z")
    print(f"-> {out}/recon_{S}.png + samples_{S}.png")


if __name__ == "__main__":
    main()
