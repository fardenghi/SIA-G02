"""Two-Stage VAE (Dai & Wipf 2019): desacopla reconstrucción de cobertura del latente.

El VAE simple tiene un trade-off duro: latente alto reconstruye nítido pero el posterior
agregado vive en una variedad finísima que `N(0,I)` no cubre (genera manchones); latente bajo
cubre mejor pero reconstruye borroso. El **two-stage** rompe el trade-off entrenando DOS VAEs:

  - **Etapa 1**: VAE de latente alto (8) sobre las imágenes. Reconstruye nítido. Sus códigos
    `μ` viven en una variedad compleja dentro de R^8.
  - **Etapa 2**: un segundo VAE de latente bajo (2) entrenado **sobre esos códigos**. Aprende
    la forma de la variedad y la mapea a un `N(0,I)` 2D denso y muestreable.

Generar = `z₂ ~ N(0,I)` → decode etapa 2 → código 8D coherente → decode etapa 1 → imagen.

El segundo VAE **reemplaza al GMM** (`aggregate_prior`): cumple el mismo rol (modelar dónde
viven los códigos) pero con capacidad arbitraria, no solo mezcla de gaussianas. Tiene sentido
conceptual recién con un dataset grande: con 32 emojis el posterior agregado son 32 manchas
que un GMM ya modela bien; con cientos de glifos distintos es una distribución compleja donde
el segundo VAE paga. Por eso usamos `load_many_emojis` (caras+animales, ~220 glifos distintos).

    uv run python scripts/two_stage_experiment.py --epochs1 400 --epochs2 3000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.aggregate_prior import AggregatePrior  # noqa: E402
from autoencoder.conv_vae import ConvVAE  # noqa: E402
from autoencoder.emoji_data import (  # noqa: E402
    FACE_ANIMAL_RANGES,
    augment_dataset,
    load_emojis,
    load_many_emojis,
)
from autoencoder.vae import VAE  # noqa: E402
from autoencoder.vae_train import train_vae  # noqa: E402

# Caras puras (homogéneas): 110 glifos distintos — base de buena recon sin memorización.
FACES_RANGES = [(0x1F600, 0x1F64A), (0x1F910, 0x1F917), (0x1F920, 0x1F92F),
                (0x1F970, 0x1F97A), (0x1F60E, 0x1F60E)]


def grid(rows, titles, size, path, suptitle):
    """Grilla de imágenes en gris: una fila por bloque, con etiqueta a la izquierda."""
    n = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.1, len(rows) * 1.25))
    axes = np.atleast_2d(axes)
    for r, (block, name) in enumerate(zip(rows, titles)):
        for j in range(n):
            axes[r, j].imshow(block[j].reshape(size, size), cmap="Greys", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Two-Stage VAE: desacopla recon de generación")
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--dataset", choices=["curated", "many", "faces"], default="many",
                   help="curated=32 emojis; many=caras+animales; faces=110 caras homogéneas")
    p.add_argument("--stage1", choices=["mlp", "conv"], default="mlp",
                   help="modelo de la etapa 1: MLP-VAE o ConvVAE (sesgo espacial)")
    p.add_argument("--latent1", type=int, default=8, help="latente etapa 1 (reconstrucción)")
    p.add_argument("--latent2", type=int, default=2, help="latente etapa 2 (generación)")
    p.add_argument("--epochs1", type=int, default=400)
    p.add_argument("--epochs2", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-aug", type=int, default=4, help="copias aumentadas por glifo (0=off)")
    p.add_argument("--gmm-k", type=int, default=12, help="componentes del GMM de comparación")
    p.add_argument("--max-n", type=int, default=None, help="cap de glifos distintos")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    S, L1, L2, D = args.size, args.latent1, args.latent2, args.size * args.size
    rng = np.random.default_rng(args.seed)
    # Carpeta etiquetada por dataset, modelo y latentes: permite comparar sin pisarse.
    out = Path(f"out/two_stage/{args.dataset}_{args.stage1}_L1-{L1}_L2-{L2}_aug{args.n_aug}")
    out.mkdir(parents=True, exist_ok=True)

    # ---- datos: 32 emojis / caras+animales / caras homogéneas --------------------------- #
    if args.dataset == "curated":
        X, labels = load_emojis(size=S)
    elif args.dataset == "faces":
        X, labels = load_many_emojis(size=S, ranges=FACES_RANGES, max_n=args.max_n)
    else:
        X, labels = load_many_emojis(size=S, ranges=FACE_ANIMAL_RANGES, max_n=args.max_n)
    n_distinct = X.shape[0]
    if args.n_aug > 0:
        X, labels = augment_dataset(X, labels, size=S, n_aug=args.n_aug,
                                    rng=np.random.default_rng(args.seed))
    print(f"dataset: {n_distinct} glifos distintos -> {X.shape[0]} con augment "
          f"(n_aug={args.n_aug}), {S}x{S}\n")

    # ================== ETAPA 1: VAE de imágenes (latente alto, recon nítida) ============ #
    if args.stage1 == "conv":
        print(f"== ETAPA 1: ConvVAE  canales=[16,32] dense=64 latente {L1}  "
              f"({args.epochs1} ep) ==")
        vae1 = ConvVAE(size=S, latent_dim=L1, conv_channels=[16, 32], dense_hidden=64,
                       activation="relu", output_activation="sigmoid", init="he_normal",
                       loss="bce", seed=args.seed)
    else:
        print(f"== ETAPA 1: VAE imagenes  [{D}, 256, 64] latente {L1}  ({args.epochs1} ep) ==")
        vae1 = VAE([D, 256, 64], latent_dim=L1, activation="relu", output_activation="sigmoid",
                   init="he_normal", loss="bce", seed=args.seed)
    f1 = train_vae(vae1, X, epochs=args.epochs1, lr=1e-3, beta=1.0,
                   beta_warmup=args.epochs1 // 5, seed=args.seed, batch_size=args.batch,
                   lr_schedule="cosine")
    print(f"   recon_det={f1['recon_det']:.1f}  (/px {f1['recon_det']/D:.4f})  "
          f"kl={f1['kl']:.3f}\n")

    # ---- puente: códigos del posterior agregado, estandarizados ------------------------- #
    mu1, _ = vae1.encode(X)                       # (N, L1) — variedad compleja en R^L1
    # recon determinista (z=μ) SOLO sobre los originales (primer bloque del augment): la
    # métrica que importa para "fidelidad", sin las copias rotadas/desplazadas que la inflan.
    recon_orig = vae1.input_dim * vae1._loss_value(vae1.decode(mu1[:n_distinct]), X[:n_distinct])
    print(f"   recon_det(originales)={recon_orig:.1f}  (/px {recon_orig/D:.4f})\n")
    code_mean = mu1.mean(axis=0)
    code_std = mu1.std(axis=0) + 1e-8
    codes = (mu1 - code_mean) / code_std          # z-score: MSE parejo entre dims

    # ================== ETAPA 2: VAE de códigos (latente bajo, generación) =============== #
    print(f"== ETAPA 2: VAE codigos   [{L1}, 64, 32] latente {L2}  ({args.epochs2} ep) ==")
    vae2 = VAE([L1, 64, 32], latent_dim=L2, activation="relu", output_activation="linear",
               init="he_normal", loss="mse", seed=args.seed)
    f2 = train_vae(vae2, codes, epochs=args.epochs2, lr=1e-3, beta=1.0,
                   beta_warmup=args.epochs2 // 5, seed=args.seed, batch_size=args.batch,
                   lr_schedule="cosine")
    print(f"   recon_det={f2['recon_det']:.4f}  kl={f2['kl']:.3f}\n")

    # ================== GENERACIÓN: tres métodos, misma cantidad ========================= #
    n_show = 12

    # 1) Etapa 1 sola: z8 ~ N(0,I) -> decode (línea base mala en latente alto).
    z8 = rng.standard_normal((n_show, L1))
    gen_prior = vae1.decode(z8)

    # 2) Etapa 1 + GMM sobre los códigos (el mejor método actual: ex-post density est.).
    gmm = AggregatePrior.fit(mu1, kind="gmm", k=args.gmm_k, seed=args.seed)
    gen_gmm = vae1.decode(gmm.sample(n_show, rng))

    # 3) Two-stage: z2 ~ N(0,I) -> decode etapa 2 -> des-estandarizar -> decode etapa 1.
    z2 = rng.standard_normal((n_show, L2))
    codes_hat = vae2.decode(z2) * code_std + code_mean
    gen_two = vae1.decode(codes_hat)

    grid([gen_prior, gen_gmm, gen_two],
         ["1: N(0,I)", "2: +GMM", "3: 2-stage"], S, out / "samples_compare.png",
         f"Generación con {n_distinct} glifos: etapa1 sola / etapa1+GMM / two-stage")

    # ---- reconstrucción de la etapa 1 (sanity de nitidez) ------------------------------- #
    idx = rng.choice(X.shape[0], n_show, replace=False)
    grid([X[idx], vae1.decode(vae1.encode(X[idx])[0])], ["in", "recon"], S,
         out / "recon_stage1.png", f"Reconstrucción etapa 1 (latente {L1})")

    # ---- recorrido del manifold 2D de la etapa 2 (la demostración del espacio navegable) - #
    if L2 == 2:
        g = 12
        # Rejilla en [-2,2]^2 del latente de la etapa 2; decode 2->8->imagen.
        lin = np.linspace(-2.0, 2.0, g)
        gx, gy = np.meshgrid(lin, lin)
        zgrid = np.stack([gx.ravel(), gy.ravel()], axis=1)        # (g*g, 2)
        codes_grid = vae2.decode(zgrid) * code_std + code_mean
        imgs = vae1.decode(codes_grid).reshape(g, g, S, S)
        canvas = np.zeros((g * S, g * S))
        for i in range(g):
            for j in range(g):
                canvas[i * S:(i + 1) * S, j * S:(j + 1) * S] = imgs[i, j]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(canvas, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Manifold 2D de la etapa 2 (decodificado 2->8->imagen)")
        fig.tight_layout()
        fig.savefig(out / "manifold_two_stage.png", dpi=120)
        plt.close(fig)

    print(f"-> {out}/samples_compare.png")
    print(f"-> {out}/recon_stage1.png")
    if L2 == 2:
        print(f"-> {out}/manifold_two_stage.png")


if __name__ == "__main__":
    main()
