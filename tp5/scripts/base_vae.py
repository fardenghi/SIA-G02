"""VAE base (vuelta a las bases): MLP, β=1, latente 8, muestreo gaussiano N(0,I).

Sin GMM, sin two-stage, sin trucos. El autoencoder generativo más simple del Ej2:
encoder/decoder MLP `[784, 256, 64] -> 8 -> [64, 256, 784]`, BCE, β=1 (ELBO estándar),
y generación muestreando del prior `z ~ N(0, I)`. Produce UNA figura con tres filas:
entrada / reconstrucción / muestras nuevas.

    uv run python scripts/base_vae.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.emoji_data import augment_dataset, load_emojis  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402
from autoencoder.vae_train import train_vae  # noqa: E402

SIZE = 56
LATENT = 2
EPOCHS = 2000
SEED = 0


def main():
    D = SIZE * SIZE
    out = Path("out/base_vae"); out.mkdir(parents=True, exist_ok=True)

    # ---- datos: 32 emojis + augment (set base del TP) ----------------------------------- #
    X0, labels0 = load_emojis(size=SIZE)
    X, _ = augment_dataset(X0, labels0, size=SIZE, n_aug=8,
                           rng=np.random.default_rng(SEED))
    print(f"dataset: {X0.shape[0]} emojis -> {X.shape[0]} con augment, {SIZE}x{SIZE}")

    # ---- VAE base: MLP, latente 8, BCE, β=1, warmup 20% --------------------------------- #
    vae = VAE([D, 256, 64], latent_dim=LATENT, activation="relu",
              output_activation="sigmoid", init="he_normal", loss="bce", seed=SEED)
    final = train_vae(vae, X, epochs=EPOCHS, lr=1e-3, beta=1.0,
                      beta_warmup=EPOCHS // 5, seed=SEED)
    print(f"recon_det={final['recon_det']:.1f} (/px {final['recon_det']/D:.4f})  "
          f"kl={final['kl']:.3f}")

    # ---- figura: in / recon / samples --------------------------------------------------- #
    n = 12
    rng = np.random.default_rng(SEED)
    ins = X0[:n]                                   # entradas (32 originales, sin augment)
    recon = vae.decode(vae.encode(ins)[0])         # reconstrucción determinista (z = μ)
    samples = vae.decode(vae.sample_prior(n, rng))  # muestras: z ~ N(0, I) -> decode

    rows = [(ins, "in"), (recon, "recon"), (samples, "samples")]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.1, 3 * 1.25))
    for r, (block, name) in enumerate(rows):
        for j in range(n):
            axes[r, j].imshow(block[j].reshape(SIZE, SIZE), cmap="Greys", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
    fig.suptitle(f"VAE base — MLP, latente {LATENT}, β=1, muestreo N(0,I)")
    fig.tight_layout()
    fig.savefig(out / "in_recon_samples.png", dpi=120)
    plt.close(fig)
    print(f"-> {out}/in_recon_samples.png")


if __name__ == "__main__":
    main()
