"""Corrida grande (overnight): MLP-VAE vs ConvVAE en caras+animales, alta capacidad.

Dataset homogéneo (~220 caras/animales) para que las figuras salgan reconocibles. Red grande,
mayor resolución, mini-batch, cosine LR y β-VAE. Guarda checkpoints y figuras de progreso cada
`--ckpt-every` épocas (para monitorear sin esperar el final) en `out/big_run/`.

    uv run python scripts/big_run.py --size 56 --latent 48 --epochs 2000

Sizing del costo: ver el benchmark impreso al arrancar (s/época × épocas).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.conv_vae import ConvVAE  # noqa: E402
from autoencoder.emoji_data import FACE_ANIMAL_RANGES, load_many_emojis  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402
from autoencoder.vae_train import train_vae  # noqa: E402

OUT = Path("out/big_run")


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
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_callback(name, X, size, latent):
    """Callback: guarda checkpoint de pesos + figura (in/recon/samples) y loguea progreso."""
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], 10, replace=False)
    z = np.random.default_rng(7).standard_normal((10, latent))

    def cb(epoch, vae, met):
        np.save(OUT / f"{name}_ckpt.npy", vae.get_params())
        recon = vae.decode(vae.encode(X[idx])[0])
        grid([X[idx], recon, vae.decode(z)], ["in", "recon", "sample"], size,
             OUT / f"{name}_progress.png",
             f"{name} — época {epoch} (recon det. + muestras del prior)")
        print(f"  [{name}] época {epoch:>5}: recon={met['recon']:.1f} "
              f"kl={met['kl']:.2f} lr={met['lr']:.2e}", flush=True)

    return cb


def train(name, vae, X, args):
    print(f"\n== {name}: {vae.n_params} params ==", flush=True)
    t = time.time()
    f = train_vae(vae, X, epochs=args.epochs, lr=args.lr, beta=args.beta,
                  beta_warmup=args.epochs // 5, seed=0, batch_size=args.batch,
                  lr_schedule="cosine", callback=make_callback(name, X, args.size, args.latent),
                  callback_every=args.ckpt_every)
    D = args.size * args.size
    print(f"== {name} listo en {(time.time()-t)/60:.1f} min: "
          f"recon_det={f['recon_det']:.1f} ({f['recon_det']/D:.4f}/px) kl={f['kl']:.2f} ==",
          flush=True)
    return f


def main(argv=None):
    p = argparse.ArgumentParser(description="Corrida grande MLP vs CNN (caras+animales)")
    p.add_argument("--size", type=int, default=56)
    p.add_argument("--latent", type=int, default=48)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--dense-hidden", type=int, default=256)
    p.add_argument("--ckpt-every", type=int, default=100)
    args = p.parse_args(argv)
    S, D = args.size, args.size * args.size
    OUT.mkdir(parents=True, exist_ok=True)

    X, _ = load_many_emojis(size=S, ranges=FACE_ANIMAL_RANGES)
    print(f"dataset: {X.shape[0]} caras/animales {S}×{S} | latente {args.latent} "
          f"canales {args.channels} dense {args.dense_hidden} β {args.beta} "
          f"| {args.epochs} épocas batch {args.batch} (cosine LR)", flush=True)

    cnn = ConvVAE(size=S, latent_dim=args.latent, conv_channels=args.channels,
                  dense_hidden=args.dense_hidden, activation="relu",
                  output_activation="sigmoid", init="he_normal", loss="bce", seed=0)
    mlp = VAE([D, 512, 256], latent_dim=args.latent, activation="relu",
              output_activation="sigmoid", init="he_normal", loss="bce", seed=0)

    # Benchmark de costo (3 épocas de la CNN) para estimar el total antes de comprometerse.
    t0 = time.time()
    train_vae(ConvVAE(size=S, latent_dim=args.latent, conv_channels=args.channels,
                      dense_hidden=args.dense_hidden, loss="bce", seed=1),
              X, epochs=3, lr=args.lr, beta=args.beta, seed=1, batch_size=args.batch)
    spe = (time.time() - t0) / 3
    print(f"benchmark CNN: {spe:.2f} s/época -> {args.epochs} épocas ≈ "
          f"{spe*args.epochs/3600:.1f} h\n", flush=True)

    fc = train("cnn", cnn, X, args)
    fm = train("mlp", mlp, X, args)
    print(f"\nRESUMEN  CNN recon/px={fc['recon_det']/D:.4f}  MLP recon/px={fm['recon_det']/D:.4f}"
          f"  gap(CNN-MLP)={ (fc['recon_det']-fm['recon_det'])/D:+.4f}", flush=True)


if __name__ == "__main__":
    main()
