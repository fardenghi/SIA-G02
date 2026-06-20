"""Barrido de β con generación por GMM (MLP-VAE caras/animales, latente 16 fijo).

Motivado por el experimento de latente: la reconstrucción quedó **plana** (~0.348/px) variando el
latente → el latente no es la perilla de la nitidez; **β sí** (β bajo = recon más nítida). Y como
el muestreo del **posterior agregado (GMM)** desacopla la calidad del sample del matcheo con el
prior, podemos **bajar β sin romper la generación**. Este barrido lo verifica: para cada β mide
`recon_det/px` (nitidez) y compara generación `N(0,I)` vs GMM sobre el mismo decoder.

Mismo dataset/setup que big_run (MLP `[D,512,256]`, batch 64, cosine LR, 2000 épocas).

    uv run python scripts/beta_gmm_experiment.py --betas 0.25 0.5 1.0 --latent 16
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

OUT = Path("out/beta_gmm")


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


def run_beta(beta, X, size, latent, epochs, batch, lr, gmm_k):
    D = size * size
    vae = VAE([D, 512, 256], latent_dim=latent, activation="relu",
              output_activation="sigmoid", init="he_normal", loss="bce", seed=0)
    tag = f"b{beta:g}".replace(".", "")
    print(f"\n== β={beta}: {vae.n_params} params ==", flush=True)
    t = time.time()
    f = train_vae(vae, X, epochs=epochs, lr=lr, beta=beta, beta_warmup=epochs // 5,
                  seed=0, batch_size=batch, lr_schedule="cosine")
    np.save(OUT / f"mlp_{tag}_ckpt.npy", vae.get_params())

    mu, _ = vae.encode(X)
    m = posterior_prior_mismatch(mu)
    gmm = AggregatePrior.fit(mu, kind="gmm", k=gmm_k, seed=0)

    rng = np.random.default_rng(7)
    idx = np.random.default_rng(0).choice(X.shape[0], 10, replace=False)
    z_prior = rng.standard_normal((10, latent))
    grid([X[idx], vae.decode(mu[idx]), vae.decode(z_prior), vae.decode(gmm.sample(10, rng))],
         ["in", "recon", "sample\nN(0,I)", f"sample\nGMM k={gmm_k}"], size,
         OUT / f"mlp_{tag}.png",
         f"MLP latente {latent}, β={beta} — recon nítida + generación N(0,I) vs GMM")

    line = (f"β={beta:<5} | recon_det/px={f['recon_det']/D:.4f} kl={f['kl']:.2f} "
            f"| mismatch ‖mean‖={m['mean_norm']:.2f} "
            f"std={m['std_mean']:.2f}[{m['std_min']:.2f},{m['std_max']:.2f}] "
            f"| {(time.time()-t)/60:.1f} min")
    print("  " + line, flush=True)
    return line


def main(argv=None):
    p = argparse.ArgumentParser(description="Barrido de β con GMM (MLP-VAE caras/animales)")
    p.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--size", type=int, default=56)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gmm-k", type=int, default=8)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    X, _ = load_many_emojis(size=args.size, ranges=FACE_ANIMAL_RANGES)
    print(f"dataset: {X.shape[0]} caras/animales {args.size}×{args.size} | MLP latente "
          f"{args.latent} batch={args.batch} cosine | {args.epochs} épocas | "
          f"βs {args.betas} | GMM k={args.gmm_k}", flush=True)

    lines = [run_beta(b, X, args.size, args.latent, args.epochs, args.batch, args.lr,
                      args.gmm_k) for b in args.betas]
    summary = (f"MLP-VAE caras/animales — barrido de β (latente {args.latent}, GMM)\n"
               "objetivo: β bajo baja recon_det/px (más nítido); el GMM sostiene el sample\n\n"
               + "\n".join(lines) + "\n")
    (OUT / "summary.txt").write_text(summary)
    print("\n" + "=" * 60 + "\n" + summary + "=" * 60, flush=True)


if __name__ == "__main__":
    main()
