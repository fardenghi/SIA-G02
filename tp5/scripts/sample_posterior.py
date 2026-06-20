"""Mejorar la GENERACIÓN del VAE muestreando del POSTERIOR AGREGADO, no del prior N(0,I).

Idea (Ej2c, sin reentrenar). El sample estándar es `z ~ N(0,I) -> decode`, que solo funciona si
el **posterior agregado** `q(z) = ⅟ₙ Σ q(z|xᵢ)` se parece al prior. En latente alto (la corrida
grande usa 48) eso **falla**: una muestra típica de `N(0,I)` vive a radio ≈√48≈7, mientras los
glifos codificados ocupan una cáscara finísima → la mayoría de los `z` caen en zonas que el
decoder nunca vio (manchones). El arreglo clásico (*ex-post density estimation*) es ajustar una
densidad a los `μ` del dataset y muestrear de ahí.

Corre sobre el checkpoint del big_run (`out/big_run/{cnn,mlp}_ckpt.npy`) y compara cuatro
estrategias de muestreo: prior `N(0,I)`, prior con **temperatura** `N(0,T·I)`, **Gaussiana
full-cov** del posterior agregado y **GMM diagonal** (EM, capta multimodalidad caras-vs-animales).

    uv run python scripts/sample_posterior.py            # cnn y mlp
    uv run python scripts/sample_posterior.py --model mlp --gmm-k 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.aggregate_prior import AggregatePrior, posterior_prior_mismatch  # noqa: E402
from autoencoder.conv_vae import ConvVAE  # noqa: E402
from autoencoder.emoji_data import FACE_ANIMAL_RANGES, load_many_emojis  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402

OUT = Path("out/big_run")


# --- figura --------------------------------------------------------------------- #
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


def build_model(kind, size, latent, channels, dense):
    if kind == "cnn":
        return ConvVAE(size=size, latent_dim=latent, conv_channels=channels,
                       dense_hidden=dense, activation="relu", output_activation="sigmoid",
                       init="he_normal", loss="bce", seed=0)
    return VAE([size * size, 512, 256], latent_dim=latent, activation="relu",
               output_activation="sigmoid", init="he_normal", loss="bce", seed=0)


def run_model(kind, X, size, latent, channels, dense, n, temp, gmm_k):
    ckpt = OUT / f"{kind}_ckpt.npy"
    if not ckpt.exists():
        print(f"[{kind}] sin checkpoint en {ckpt}; salteo.")
        return
    vae = build_model(kind, size, latent, channels, dense)
    vae.set_params(np.load(ckpt))

    mu, _ = vae.encode(X)                                   # posterior agregado = {μ_i}
    # Diagnóstico del mismatch con el prior N(0,I): si q(z)≈prior, mean≈0 y std≈1 por dim.
    m = posterior_prior_mismatch(mu)
    print(f"\n[{kind}] mismatch posterior-prior (objetivo: mean≈0, std≈1):")
    print(f"  ‖mean(μ)‖={m['mean_norm']:.2f}  std(μ) por dim: medio={m['std_mean']:.2f} "
          f"[min {m['std_min']:.2f}, max {m['std_max']:.2f}]")
    print(f"  → un prior N(0,I) esperaría ‖mean‖≈0 y std≈1; cuanto más lejos, peores los samples.")

    rng = np.random.default_rng(7)
    gauss = AggregatePrior.fit(mu, kind="gaussian")
    gmm = AggregatePrior.fit(mu, kind="gmm", k=gmm_k)

    z_prior = rng.standard_normal((n, latent))
    z_temp = temp * rng.standard_normal((n, latent))
    z_gauss = gauss.sample(n, rng)
    z_gmm = gmm.sample(n, rng)

    grid([vae.decode(z_prior), vae.decode(z_temp), vae.decode(z_gauss), vae.decode(z_gmm)],
         [f"prior\nN(0,I)", f"temp\nT={temp}", "post.agg.\nGauss", f"post.agg.\nGMM k={gmm_k}"],
         size, OUT / f"{kind}_sample_fix.png",
         f"{kind} — generación: prior vs posterior agregado (sin reentrenar)")
    print(f"  figura -> {OUT / f'{kind}_sample_fix.png'}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Muestrear del posterior agregado (mejora samples)")
    p.add_argument("--model", choices=["cnn", "mlp", "both"], default="both")
    p.add_argument("--size", type=int, default=56)
    p.add_argument("--latent", type=int, default=48)
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--dense-hidden", type=int, default=256)
    p.add_argument("--n", type=int, default=10, help="muestras por fila")
    p.add_argument("--temp", type=float, default=0.6, help="temperatura del prior N(0,T·I)")
    p.add_argument("--gmm-k", type=int, default=8, help="componentes del GMM")
    args = p.parse_args(argv)

    X, _ = load_many_emojis(size=args.size, ranges=FACE_ANIMAL_RANGES)
    print(f"dataset: {X.shape[0]} caras/animales {args.size}×{args.size} | latente {args.latent}")

    kinds = ["cnn", "mlp"] if args.model == "both" else [args.model]
    for kind in kinds:
        run_model(kind, X, args.size, args.latent, args.channels, args.dense_hidden,
                  args.n, args.temp, args.gmm_k)


if __name__ == "__main__":
    main()
