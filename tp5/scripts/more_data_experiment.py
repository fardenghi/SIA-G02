"""Experimento exploratorio: ¿gana la CNN al salir del régimen de memorización (más datos)?

Pasa de 32 a ~1300 emojis distintos (`load_many_emojis`) y entrena MLP-VAE vs ConvVAE con
**mini-batch** (que desacopla el costo de N). Con tantas plantillas el MLP ya no puede
memorizarlas en un latente chico → tiene que generalizar, justo donde el prior convolucional
debería pagar. Costo acotado por (pasos × batch), no por N.

    uv run python scripts/more_data_experiment.py --epochs 120 --batch 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.conv_vae import ConvVAE  # noqa: E402
from autoencoder.emoji_data import load_many_emojis  # noqa: E402
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
    p = argparse.ArgumentParser(description="MLP vs CNN con más datos (mini-batch)")
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-n", type=int, default=None, help="cap de glifos (default: todos)")
    args = p.parse_args(argv)
    S, LAT, D = args.size, args.latent, args.size * args.size

    X, _ = load_many_emojis(size=S, max_n=args.max_n)
    print(f"dataset: {X.shape[0]} emojis distintos {S}×{S}, latente {LAT}, "
          f"{args.epochs} épocas, batch {args.batch}\n")

    mlp = VAE([D, 256, 64], latent_dim=LAT, activation="relu", output_activation="sigmoid",
              init="he_normal", loss="bce", seed=0)
    fm = train_vae(mlp, X, epochs=args.epochs, lr=1e-3, beta=1.0, beta_warmup=args.epochs // 4,
                   seed=0, batch_size=args.batch)
    cnn = ConvVAE(size=S, latent_dim=LAT, conv_channels=[16, 32], dense_hidden=64,
                  activation="relu", output_activation="sigmoid", init="he_normal",
                  loss="bce", seed=0)
    fc = train_vae(cnn, X, epochs=args.epochs, lr=1e-3, beta=1.0, beta_warmup=args.epochs // 4,
                   seed=0, batch_size=args.batch)

    print(f"{'modelo':>6} {'recon_det':>10} {'/px':>8} {'params':>10}")
    print(f"{'MLP':>6} {fm['recon_det']:>10.1f} {fm['recon_det']/D:>8.4f} {mlp.n_params:>10}")
    print(f"{'CNN':>6} {fc['recon_det']:>10.1f} {fc['recon_det']/D:>8.4f} {cnn.n_params:>10}")
    gap = (fc['recon_det'] - fm['recon_det']) / D
    print(f"\ngap/px CNN-MLP = {gap:+.4f}   (32 emojis: +0.0100; <0 ⇒ la CNN gana)")

    out = Path("out/more_data"); out.mkdir(parents=True, exist_ok=True)
    idx = np.random.default_rng(0).choice(X.shape[0], 12, replace=False)
    grid([X[idx], mlp.decode(mlp.encode(X[idx])[0]), cnn.decode(cnn.encode(X[idx])[0])],
         ["in", "MLP", "CNN"], S, out / "recon.png",
         f"Reconstrucción con {X.shape[0]} emojis (latente {LAT}) — in / MLP / CNN")
    z = np.random.default_rng(7).standard_normal((12, LAT))
    grid([mlp.decode(z), cnn.decode(z)], ["MLP", "CNN"], S, out / "samples.png",
         f"Muestras del prior con {X.shape[0]} emojis (latente {LAT}) — mismos z")
    print(f"-> {out}/recon.png + samples.png")


if __name__ == "__main__":
    main()
