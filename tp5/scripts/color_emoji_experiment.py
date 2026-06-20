"""Experimento exploratorio: ¿cambia el veredicto MLP vs CNN con emojis a COLOR (3 canales)?

Entrena MLP-VAE y ConvVAE sobre los 32 emojis en RGB (sin augment, mismo régimen que el barrido
de grises) y compara recon_det + genera figuras a color (reconstrucción y muestras del prior).
El color triplica las entradas/salidas densas del MLP mientras la CNN casi no crece (pesos
compartidos): es el régimen donde la eficiencia de parámetros de la conv empieza a pesar.

    uv run python scripts/color_emoji_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.conv_vae import ConvVAE  # noqa: E402
from autoencoder.emoji_data import load_emojis  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402
from autoencoder.vae_train import train_vae  # noqa: E402

S, C, LAT, EP = 28, 3, 8, 2000
D = C * S * S


def to_img(vec):
    """Vector canal-mayor (3·S·S,) -> (S, S, 3) en [0,1] para imshow RGB."""
    return np.clip(vec.reshape(C, S, S).transpose(1, 2, 0), 0, 1)


def grid(rows, titles, path, suptitle):
    n = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.3))
    axes = np.atleast_2d(axes)
    for r, (block, name) in enumerate(zip(rows, titles)):
        for j in range(n):
            axes[r, j].imshow(to_img(block[j]), interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    X, labels = load_emojis(size=S, color=True)
    print(f"dataset color: {X.shape}  (latente {LAT}, {EP} épocas, sin augment)\n")

    mlp = VAE([D, 256, 64], latent_dim=LAT, activation="relu", output_activation="sigmoid",
              init="he_normal", loss="bce", seed=0)
    fm = train_vae(mlp, X, epochs=EP, lr=1e-3, beta=1.0, beta_warmup=1000, seed=0)
    cnn = ConvVAE(size=S, latent_dim=LAT, conv_channels=[16, 32], dense_hidden=64,
                  in_channels=C, activation="relu", output_activation="sigmoid",
                  init="he_normal", loss="bce", seed=0)
    fc = train_vae(cnn, X, epochs=EP, lr=1e-3, beta=1.0, beta_warmup=1000, seed=0)

    print(f"{'modelo':>6} {'recon_det':>10} {'/canal-px':>10} {'params':>10}")
    print(f"{'MLP':>6} {fm['recon_det']:>10.1f} {fm['recon_det']/D:>10.4f} {mlp.n_params:>10}")
    print(f"{'CNN':>6} {fc['recon_det']:>10.1f} {fc['recon_det']/D:>10.4f} {cnn.n_params:>10}")
    print(f"\ngap CNN-MLP = {fc['recon_det']-fm['recon_det']:+.1f} nats "
          f"({(fc['recon_det']-fm['recon_det'])/D:+.4f}/canal-px)")

    out = Path("out/color_emoji"); out.mkdir(parents=True, exist_ok=True)
    # Reconstrucción determinista (z=μ) de los primeros 10 emojis.
    idx = np.arange(10)
    xm = mlp.decode(mlp.encode(X)[0])
    xc = cnn.decode(cnn.encode(X)[0])
    grid([X[idx], xm[idx], xc[idx]], ["in", "MLP", "CNN"],
         out / "color_reconstruction.png",
         f"Reconstrucción a color (latente {LAT}) — in / MLP / CNN")
    # Muestras del prior (mismos z para ambos).
    z = np.random.default_rng(7).standard_normal((10, LAT))
    grid([mlp.decode(z), cnn.decode(z)], ["MLP", "CNN"],
         out / "color_samples.png",
         f"Muestras del prior a color (latente {LAT}) — mismos z")
    print(f"\n-> {out}/color_reconstruction.png + color_samples.png")


if __name__ == "__main__":
    main()
