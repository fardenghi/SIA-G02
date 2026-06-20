"""MLP-VAE en caras/animales a latente intermedio (8/16) + generación con GMM (Ej2c).

Busca el **balance**: latente intermedio (no 48) para que la reconstrucción conserve detalle sin
vaciar tanto el prior, y muestreo del **posterior agregado** (GMM, ex-post density estimation) en
vez de `N(0,I)` para que la generación salga coherente. Mismo dataset y setup que `big_run.py`
(MLP `[D,512,256]`, β=0.5, batch 64, cosine LR, 2000 épocas); la única variable es el latente.

Para cada latente entrena y produce una figura comparando, sobre el MISMO decoder:
**in / recon / sample N(0,I) / sample GMM**. Así se ve que el GMM mejora la generación sin tocar
la reconstrucción. La integración por config (revertible a `N(0,I)`) está en `generation.prior`.

    uv run python scripts/gmm_latent_experiment.py --latents 8 16 --epochs 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.aggregate_prior import AggregatePrior, posterior_prior_mismatch  # noqa: E402
from autoencoder.emoji_data import FACE_ANIMAL_RANGES, load_many_emojis  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402
from autoencoder.vae_train import train_vae  # noqa: E402

OUT = Path("out/gmm_latent")


def grid(rows, titles, size, path, suptitle):
    n = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.3))
    axes = np.atleast_2d(axes)
    for r, (block, name) in enumerate(zip(rows, titles)):
        for j in range(n):
            axes[r, j].imshow(block[j].reshape(size, size), cmap="Greys", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_latent(latent, X, size, epochs, beta, batch, lr, gmm_k):
    D = size * size
    vae = VAE([D, 512, 256], latent_dim=latent, activation="relu",
              output_activation="sigmoid", init="he_normal", loss="bce", seed=0)
    print(f"\n== latente {latent}: {vae.n_params} params ==", flush=True)
    t = time.time()
    f = train_vae(vae, X, epochs=epochs, lr=lr, beta=beta, beta_warmup=epochs // 5,
                  seed=0, batch_size=batch, lr_schedule="cosine")
    np.save(OUT / f"mlp_l{latent}_ckpt.npy", vae.get_params())

    mu, _ = vae.encode(X)
    m = posterior_prior_mismatch(mu)
    gmm = AggregatePrior.fit(mu, kind="gmm", k=gmm_k, seed=0)

    rng = np.random.default_rng(7)
    idx = np.random.default_rng(0).choice(X.shape[0], 10, replace=False)
    recon = vae.decode(mu[idx])
    z_prior = rng.standard_normal((10, latent))
    grid([X[idx], recon, vae.decode(z_prior), vae.decode(gmm.sample(10, rng))],
         ["in", "recon", "sample\nN(0,I)", f"sample\nGMM k={gmm_k}"], size,
         OUT / f"mlp_l{latent}.png",
         f"MLP latente {latent} — recon nítida + generación N(0,I) vs posterior agregado (GMM)")

    line = (f"latent={latent:>2} | recon_det/px={f['recon_det']/D:.4f} kl={f['kl']:.2f} "
            f"params={vae.n_params} | mismatch ‖mean‖={m['mean_norm']:.2f} "
            f"std={m['std_mean']:.2f}[{m['std_min']:.2f},{m['std_max']:.2f}] "
            f"| {(time.time()-t)/60:.1f} min")
    print("  " + line, flush=True)
    return line


def main(argv=None):
    p = argparse.ArgumentParser(description="MLP-VAE latente intermedio + GMM (caras/animales)")
    p.add_argument("--latents", type=int, nargs="+", default=[8, 16])
    p.add_argument("--size", type=int, default=56)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gmm-k", type=int, default=8)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    X, _ = load_many_emojis(size=args.size, ranges=FACE_ANIMAL_RANGES)
    print(f"dataset: {X.shape[0]} caras/animales {args.size}×{args.size} | "
          f"MLP β={args.beta} batch={args.batch} cosine | {args.epochs} épocas | "
          f"latentes {args.latents} | GMM k={args.gmm_k}", flush=True)

    lines = [run_latent(L, X, args.size, args.epochs, args.beta, args.batch, args.lr,
                        args.gmm_k) for L in args.latents]
    summary = ("MLP-VAE caras/animales — latente intermedio + GMM\n"
               f"referencia big_run latente 48: recon_det/px=0.3487 ‖mean‖=1.57\n\n"
               + "\n".join(lines) + "\n")
    (OUT / "summary.txt").write_text(summary)
    print("\n" + "=" * 60 + "\n" + summary + "=" * 60, flush=True)


if __name__ == "__main__":
    main()
