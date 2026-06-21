"""Análisis de épocas óptimas del VAE base vía train/val loss (early stopping).

¿Cuántas épocas conviene entrenar? La respuesta principista: hasta que la pérdida de
**validación** deja de mejorar. Entrenar de más sobreajusta (la train sigue bajando pero la
val sube). Acá:

  1. Split train/val SIN fuga: la val son poses nuevas (augment con otra semilla) de las
     mismas 32 plantillas, sin los originales. Partir el pool aumentado al azar metería copias
     rotadas de la misma plantilla en ambos lados → val demasiado fácil. Esto evita eso.
  2. Curva train vs val (recon por píxel, determinista z=μ) cada `EVAL_EVERY` épocas.
  3. in vs recon con el error de reconstrucción por imagen (MSE/px) y su media.
  4. Época óptima = argmin de la val; además se reporta el inicio del plateau (dentro del 1%).

Latente 8 (con capacidad de memorizar → muestra el sobreajuste), 28px para iterar rápido.

    uv run python scripts/epochs_analysis.py
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

SIZE = 28
LATENT = 8
EPOCHS = 3000
EVAL_EVERY = 25
SEED = 0


def recon_px(vae: VAE, data: np.ndarray) -> float:
    """Recon determinista (z=μ) por píxel: BCE/px, comparable a `recon_det/px`."""
    mu, _ = vae.encode(data)
    return float(vae._loss_value(vae.decode(mu), data))


def main():
    D = SIZE * SIZE
    out = Path("out/epochs_analysis"); out.mkdir(parents=True, exist_ok=True)

    # ---- split train/val sin fuga ------------------------------------------------------- #
    X0, labels0 = load_emojis(size=SIZE)                       # 32 originales (plantillas)
    Xtr, _ = augment_dataset(X0, labels0, size=SIZE, n_aug=8,  # train: originales + 8 poses
                             rng=np.random.default_rng(SEED))
    # val: poses NUEVAS (otra semilla), sin los originales (primer bloque del augment).
    val_aug, _ = augment_dataset(X0, labels0, size=SIZE, n_aug=4,
                                 rng=np.random.default_rng(999))
    Xval = val_aug[X0.shape[0]:]                               # descarta el bloque original
    print(f"train: {Xtr.shape[0]}  val: {Xval.shape[0]}  ({SIZE}x{SIZE}, latente {LATENT})")

    # ---- entrenar registrando train/val cada EVAL_EVERY épocas -------------------------- #
    vae = VAE([D, 256, 64], latent_dim=LATENT, activation="relu",
              output_activation="sigmoid", init="he_normal", loss="bce", seed=SEED)
    hist = {"epoch": [], "train": [], "val": []}

    def cb(epoch, model, metrics):
        hist["epoch"].append(epoch)
        hist["train"].append(recon_px(model, Xtr))
        hist["val"].append(recon_px(model, Xval))

    train_vae(vae, Xtr, epochs=EPOCHS, lr=1e-3, beta=1.0, beta_warmup=EPOCHS // 5,
              seed=SEED, callback=cb, callback_every=EVAL_EVERY)

    ep = np.array(hist["epoch"]); tr = np.array(hist["train"]); va = np.array(hist["val"])

    # ---- época óptima: argmin de val + inicio del plateau (dentro del 1%) ---------------- #
    best_i = int(va.argmin())
    best_epoch, best_val = int(ep[best_i]), float(va[best_i])
    plateau_epoch = int(ep[va <= best_val * 1.01][0])
    print(f"\n== ANALISIS DE EPOCAS ==")
    print(f"  val minima:       {best_val:.4f} /px  @ epoca {best_epoch}")
    print(f"  inicio plateau:   epoca {plateau_epoch}  (val a <=1% del minimo)")
    print(f"  train @ optimo:   {float(tr[best_i]):.4f} /px")
    print(f"  gap train-val:    {best_val - float(tr[best_i]):+.4f} /px")

    # ---- plot 1: curvas train vs val ---------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, tr, label="train", color="tab:blue")
    ax.plot(ep, va, label="val", color="tab:orange")
    ax.axvline(best_epoch, color="tab:green", ls="--", lw=1.2,
               label=f"óptimo (val min) = {best_epoch} ep")
    ax.axvline(plateau_epoch, color="gray", ls=":", lw=1.0,
               label=f"plateau ≈ {plateau_epoch} ep")
    ax.scatter([best_epoch], [best_val], color="tab:green", zorder=5)
    ax.set_xlabel("época"); ax.set_ylabel("recon BCE / px (determinista, z=μ)")
    ax.set_title(f"Train vs Val — VAE base latente {LATENT} ({SIZE}×{SIZE})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "loss_curves.png", dpi=120); plt.close(fig)

    # ---- plot 2: in vs recon + error por imagen ----------------------------------------- #
    n = 12
    ins = X0[:n]
    recon = vae.decode(vae.encode(ins)[0])
    err = ((recon - ins) ** 2).mean(axis=1)               # MSE/px por imagen
    mse_val = float(((vae.decode(vae.encode(Xval)[0]) - Xval) ** 2).mean())
    fig, axes = plt.subplots(2, n, figsize=(n * 1.1, 2 * 1.45))
    for r, (block, name) in enumerate([(ins, "in"), (recon, "recon")]):
        for j in range(n):
            axes[r, j].imshow(block[j].reshape(SIZE, SIZE), cmap="Greys", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
    for j in range(n):
        axes[1, j].set_xlabel(f"{err[j]:.3f}", fontsize=8)     # MSE/px por columna
    fig.suptitle(f"in vs recon — MSE/px medio: originales {float(err.mean()):.4f}  |  "
                 f"val {mse_val:.4f}")
    fig.tight_layout(); fig.savefig(out / "in_vs_recon.png", dpi=120); plt.close(fig)

    print(f"\n-> {out}/loss_curves.png")
    print(f"-> {out}/in_vs_recon.png")


if __name__ == "__main__":
    main()
