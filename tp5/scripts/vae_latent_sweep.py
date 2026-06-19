"""Barrido de latent_dim del VAE (MLP o CNN): recon y nº de unidades activas vs latent_dim.

Responde "¿cuántas dimensiones efectivas piden los emojis?" y, con `--kind conv`, permite
comparar la CNN contra el MLP bajo el mismo régimen (mismo dataset, épocas y latentes). Entrena
varios modelos con β=1 canónico y observa el pruning automático (dims que colapsan a KL≈0). Es
un script aparte del CLI normal.

    uv run python scripts/vae_latent_sweep.py --kind mlp
    uv run python scripts/vae_latent_sweep.py --kind conv --latents 2 4 8 16 32
    uv run python scripts/vae_latent_sweep.py --kind conv --conv-channels 32 64 --tag conv_wide
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder.conv_vae import ConvVAE
from autoencoder.emoji_data import augment_dataset, load_emojis
from autoencoder.vae import VAE
from autoencoder.vae_metrics_viz import active_units, plot_latent_sweep
from autoencoder.vae_train import train_vae


def build_model(kind, size, latent, channels, dense_hidden, seed):
    """Construye MLP-VAE o ConvVAE con la misma matemática variacional (β=1 canónico)."""
    if kind == "conv":
        return ConvVAE(size=size, latent_dim=latent, conv_channels=channels,
                       dense_hidden=dense_hidden, activation="relu",
                       output_activation="sigmoid", init="he_normal", loss="bce", seed=seed)
    return VAE([size * size, 256, 64], latent_dim=latent, activation="relu",
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
    parser.add_argument("--augment", action="store_true",
                        help="usa el dataset aumentado (más caro); por defecto, 32 emojis base")
    parser.add_argument("--tag", default=None, help="sufijo de salida (default = kind)")
    parser.add_argument("--out-dir", default="out/vae_latent_sweep")
    args = parser.parse_args(argv)
    tag = args.tag or args.kind

    X, labels = load_emojis(size=args.size)
    if args.augment:
        X, labels = augment_dataset(X, labels, size=args.size, n_aug=8,
                                    rng=np.random.default_rng(0), max_rot=12,
                                    max_shift=2, max_zoom=0.1)
    regime = "aug" if args.augment else "base"
    chan = f" canales={args.conv_channels}" if args.kind == "conv" else ""
    print(f"barrido {args.kind}{chan} sobre {X.shape[0]} emojis ({regime}), "
          f"{args.epochs} épocas (β={args.beta})\n")
    print(f"{'latent':>8} {'recon':>10} {'kl':>9} {'activas':>9}")
    rows = []
    for latent in args.latents:
        vae = build_model(args.kind, args.size, latent, args.conv_channels,
                          args.dense_hidden, args.seed)
        f = train_vae(vae, X, epochs=args.epochs, lr=args.lr, beta=args.beta,
                      beta_warmup=args.beta_warmup, seed=args.seed)
        n_active = active_units(vae, X, threshold=args.threshold)
        print(f"{latent:>8} {f['recon_det']:>10.2f} {f['kl']:>9.3f} {n_active:>9}")
        rows.append({"latent_dim": latent, "recon": f["recon_det"], "kl": f["kl"],
                     "active_units": n_active})

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
