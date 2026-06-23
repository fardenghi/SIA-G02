"""Barrido de β del VAE: entrena un modelo por cada β y grafica recon/KL vs β.

Reproduce la ablación del README (frontera del trade-off + zona de colapso). Entrena varios
modelos, así que es un script aparte del CLI normal.

    uv run python scripts/vae_beta_sweep.py
    uv run python scripts/vae_beta_sweep.py --epochs 2500 --betas 1 8 16 100 784
    uv run python scripts/vae_beta_sweep.py --dataset fashion --batch 256 --latent 16
    uv run python scripts/vae_beta_sweep.py --dataset celeba --batch 128 --latent 16
    uv run python scripts/vae_beta_sweep.py --dataset emoji_multi --latent 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.celeba_data import load_celeba
from autoencoder.emoji_data import load_emojis
from autoencoder.emoji_multi_data import load_multi_emojis
from autoencoder.minecraft_data import load_minecraft
from autoencoder.mnist_data import load_mnist
from autoencoder.vae import VAE
from autoencoder.vae_metrics_viz import plot_beta_sweep
from autoencoder.vae_train import EarlyStopping, train_vae


def recon_px(vae, X):
    mu, _ = vae.encode(X)
    return float(vae._loss_value(vae.decode(mu), X))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Barrido de β del VAE")
    parser.add_argument("--dataset", choices=["emojis", "emoji_multi", "celeba", "mnist", "fashion", "minecraft", "minecraft-old"],
                        default="emojis", help="dataset a usar")
    parser.add_argument("--latent", type=int, default=2,
                        help="dimensión del espacio latente (default 2)")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 784.0])
    parser.add_argument("--max-n", type=int, default=None,
                        help="máximo de muestras (celeba/mnist); None = todos")
    parser.add_argument("--color", action="store_true",
                        help="usa texturas RGB en vez de grises (solo para minecraft)")
    parser.add_argument("--blocks-only", action="store_true",
                        help="filtra escaleras, vallas, plantas, etc. (solo minecraft)")
    parser.add_argument("--batch", type=int, default=None,
                        help="mini-batch (None=full-batch); recomendado para datasets grandes")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stop: evaluaciones sin mejora (0=off)")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="cada cuántas épocas evaluar val para el early stopping")
    parser.add_argument("--from-csv", default=None,
                        help="carga este CSV y regenera solo el plot (no entrena)")
    parser.add_argument("--out-dir", default="out/vae_beta_sweep")
    args = parser.parse_args(argv)

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        tag = Path(args.from_csv).stem.removeprefix("beta_sweep_")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_beta_sweep(df, path=out_dir / f"beta_sweep_{tag}.png")
        print(f"-> {out_dir}/beta_sweep_{tag}.png")
        return 0

    if args.dataset == "celeba":
        X, _ = load_celeba(n=args.max_n or 3000, size=args.size, seed=args.seed)
    elif args.dataset in ("mnist", "fashion"):
        X, _ = load_mnist(n=args.max_n, seed=args.seed, kind=args.dataset)
    elif args.dataset == "emoji_multi":
        X, _ = load_multi_emojis(size=args.size, seed=args.seed)
    elif args.dataset in ("minecraft", "minecraft-old"):
        X, _ = load_minecraft(size=args.size, color=args.color, n=args.max_n,
                              seed=args.seed, blocks_only=args.blocks_only,
                              classic=(args.dataset == "minecraft-old"))
    else:
        X, _ = load_emojis(size=args.size)

    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(round(args.val_frac * X.shape[0])))
    perm = rng.permutation(X.shape[0])
    Xval, Xtr = X[perm[:n_val]], X[perm[n_val:]]

    tag = f"{args.dataset}_L{args.latent}"
    print(f"barrido de β sobre {X.shape[0]} muestras ({args.dataset}), "
          f"train={Xtr.shape[0]} val={Xval.shape[0]}, "
          f"latente={args.latent}, max {args.epochs} épocas\n")
    print(f"{'beta':>8} {'recon_tr':>10} {'recon_val':>11} {'kl':>9} {'epocas':>8}  diagnóstico")
    D = X.shape[1]
    rows = []
    for beta in args.betas:
        vae = VAE([D, 256, 64], latent_dim=args.latent, activation="relu",
                  output_activation="sigmoid", init="he_normal", loss="bce",
                  seed=args.seed)
        patience = args.patience if args.patience > 0 else args.epochs
        es = EarlyStopping(val_fn=lambda m: recon_px(m, Xval),
                           train_fn=lambda m: recon_px(m, Xtr),
                           patience=patience)
        f = train_vae(vae, Xtr, epochs=args.epochs, lr=args.lr, beta=beta,
                      beta_warmup=args.beta_warmup, seed=args.seed,
                      batch_size=args.batch, callback=es, callback_every=args.eval_every)
        if args.patience > 0:
            es.restore(vae)
        recon_tr = recon_px(vae, Xtr)
        recon_va = recon_px(vae, Xval)
        stopped = es.stopped_epoch if es.stopped_epoch >= 0 else args.epochs
        diag = "COLAPSO" if f["kl"] < 0.05 else "sano"
        print(f"{beta:>8.2f} {recon_tr:>10.4f} {recon_va:>11.4f} {f['kl']:>9.3f} {stopped:>8}  {diag}")
        rows.append({"beta": beta, "recon": recon_tr, "recon_val": recon_va,
                     "kl": f["kl"], "epochs": stopped})

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"beta_sweep_{tag}.csv", index=False)
    plot_beta_sweep(df, path=out_dir / f"beta_sweep_{tag}.png")
    print(f"\n-> {out_dir}/beta_sweep_{tag}.csv + beta_sweep_{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
