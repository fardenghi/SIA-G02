"""Barrido de latent_dim del VAE (MLP o CNN): recon y nº de unidades activas vs latent_dim.

Responde "¿cuántas dimensiones efectivas piden los emojis?" y, con `--kind conv`, permite
comparar la CNN contra el MLP bajo el mismo régimen (mismo dataset, épocas y latentes). Entrena
varios modelos con β=1 canónico y observa el pruning automático (dims que colapsan a KL≈0). Es
un script aparte del CLI normal.

    uv run python scripts/vae_latent_sweep.py --kind mlp
    uv run python scripts/vae_latent_sweep.py --kind conv --latents 2 4 8 16 32
    uv run python scripts/vae_latent_sweep.py --kind conv --conv-channels 32 64 --tag conv_wide
    uv run python scripts/vae_latent_sweep.py --dataset celeba --batch 128 --max-n 3000
    uv run python scripts/vae_latent_sweep.py --dataset mnist --batch 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.celeba_data import load_celeba
from autoencoder.conv_vae import ConvVAE
from autoencoder.emoji_data import augment_dataset, load_emojis
from autoencoder.emoji_multi_data import load_multi_emojis
from autoencoder.minecraft_data import load_minecraft
from autoencoder.mnist_data import load_mnist
from autoencoder.vae import VAE
from autoencoder.vae_metrics_viz import active_units, plot_latent_sweep
from autoencoder.vae_train import EarlyStopping, train_vae


def recon_px(vae, X):
    mu, _ = vae.encode(X)
    return float(vae._loss_value(vae.decode(mu), X))


def build_model(kind, size, latent, channels, dense_hidden, seed, n_input=None):
    """Construye MLP-VAE o ConvVAE con la misma matemática variacional (β=1 canónico)."""
    D = n_input if n_input is not None else size * size
    if kind == "conv":
        return ConvVAE(size=size, latent_dim=latent, conv_channels=channels,
                       dense_hidden=dense_hidden, activation="relu",
                       output_activation="sigmoid", init="he_normal", loss="bce", seed=seed)
    return VAE([D, 256, 64], latent_dim=latent, activation="relu",
               output_activation="sigmoid", init="he_normal", loss="bce", seed=seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Barrido de latent_dim del VAE (mlp|conv)")
    parser.add_argument("--kind", choices=["mlp", "conv"], default="mlp")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="umbral de KL (nats) para contar una dim como activa")
    parser.add_argument("--latents", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[16, 32],
                        help="canales de las convs del encoder (solo kind=conv)")
    parser.add_argument("--dense-hidden", type=int, default=64)
    parser.add_argument("--dataset", choices=["emojis", "emoji_multi", "celeba", "mnist", "fashion", "minecraft", "minecraft-old"],
                        default="emojis", help="dataset a usar")
    parser.add_argument("--color", action="store_true",
                        help="usa texturas RGB en vez de grises (solo para minecraft)")
    parser.add_argument("--blocks-only", action="store_true",
                        help="filtra escaleras, vallas, plantas, etc. (solo minecraft)")
    parser.add_argument("--augment", action="store_true",
                        help="usa el dataset aumentado (más caro); solo para emojis")
    parser.add_argument("--max-n", type=int, default=None,
                        help="máximo de muestras (celeba/mnist); None = todos")
    parser.add_argument("--batch", type=int, default=None,
                        help="mini-batch (None=full-batch); recomendado para datasets grandes")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="fracción de datos para validación (default 0.2)")
    parser.add_argument("--patience", type=int, default=10,
                        help="early stop: evaluaciones sin mejora antes de cortar (0=off)")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="cada cuántas épocas evaluar val para el early stopping")
    parser.add_argument("--tag", default=None, help="sufijo de salida (default = dataset_kind)")
    parser.add_argument("--out-dir", default="out/vae_latent_sweep")
    args = parser.parse_args(argv)
    tag = args.tag or f"{args.dataset}_{args.kind}"

    if args.dataset == "celeba":
        X, _ = load_celeba(n=args.max_n or 3000, size=args.size, seed=args.seed)
        regime = f"celeba n={X.shape[0]}"
    elif args.dataset in ("mnist", "fashion"):
        X, _ = load_mnist(n=args.max_n, seed=args.seed, kind=args.dataset)
        regime = f"{args.dataset} n={X.shape[0]}"
    elif args.dataset == "emoji_multi":
        X, _ = load_multi_emojis(size=args.size, seed=args.seed)
        regime = f"emoji_multi n={X.shape[0]}"
    elif args.dataset in ("minecraft", "minecraft-old"):
        X, _ = load_minecraft(size=args.size, color=args.color, n=args.max_n,
                              seed=args.seed, blocks_only=args.blocks_only,
                              classic=(args.dataset == "minecraft-old"))
        era = "-old" if args.dataset == "minecraft-old" else ""
        regime = f"minecraft{era} n={X.shape[0]} ({'rgb' if args.color else 'gray'})"
    else:
        X, labels = load_emojis(size=args.size)
        if args.augment:
            X, labels = augment_dataset(X, labels, size=args.size, n_aug=8,
                                        rng=np.random.default_rng(0), max_rot=12,
                                        max_shift=2, max_zoom=0.1)
        regime = "aug" if args.augment else "base"

    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(round(args.val_frac * X.shape[0])))
    perm = rng.permutation(X.shape[0])
    Xval, Xtr = X[perm[:n_val]], X[perm[n_val:]]

    chan = f" canales={args.conv_channels}" if args.kind == "conv" else ""
    print(f"barrido {args.kind}{chan} sobre {X.shape[0]} muestras ({regime}), "
          f"train={Xtr.shape[0]} val={Xval.shape[0]}, "
          f"max {args.epochs} épocas (β={args.beta})\n")
    print(f"{'latent':>8} {'recon_tr':>10} {'recon_val':>11} {'kl':>9} {'activas':>9} {'epocas':>8}")
    rows = []
    D = X.shape[1]  # size*size o size*size*3 según --color
    for latent in args.latents:
        vae = build_model(args.kind, args.size, latent, args.conv_channels,
                          args.dense_hidden, args.seed, n_input=D)
        patience = args.patience if args.patience > 0 else args.epochs
        es = EarlyStopping(val_fn=lambda m: recon_px(m, Xval),
                           train_fn=lambda m: recon_px(m, Xtr),
                           patience=patience)
        f = train_vae(vae, Xtr, epochs=args.epochs, lr=args.lr, beta=args.beta,
                      beta_warmup=args.beta_warmup, seed=args.seed,
                      batch_size=args.batch, callback=es, callback_every=args.eval_every)
        if args.patience > 0:
            es.restore(vae)
        n_active = active_units(vae, Xtr, threshold=args.threshold)
        stopped = es.stopped_epoch if es.stopped_epoch >= 0 else args.epochs
        recon_tr = recon_px(vae, Xtr)
        recon_va = recon_px(vae, Xval)
        print(f"{latent:>8} {recon_tr:>10.2f} {recon_va:>11.2f} "
              f"{f['kl']:>9.3f} {n_active:>9} {stopped:>8}")
        rows.append({"latent_dim": latent, "recon": recon_tr, "recon_val": recon_va,
                     "kl": f["kl"], "active_units": n_active, "epochs": stopped})

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"latent_sweep_{tag}.csv", index=False)
    plot_latent_sweep(df, path=out_dir / f"latent_sweep_{tag}.png",
                      title=f"Barrido de latent_dim ({tag}): recon vs unidades activas")
    print(f"\n-> {out_dir}/latent_sweep_{tag}.csv + latent_sweep_{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
