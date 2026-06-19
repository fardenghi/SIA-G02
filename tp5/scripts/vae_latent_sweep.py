"""Barrido de latent_dim del VAE: entrena un modelo por cada dim latente y grafica
reconstrucción y nº de unidades activas vs latent_dim.

Responde "¿cuántas dimensiones efectivas piden los emojis?". Entrena varios modelos con
β=1 canónico y observa el pruning automático (dims que colapsan a KL≈0). Es un script aparte
del CLI normal.

    uv run python scripts/vae_latent_sweep.py
    uv run python scripts/vae_latent_sweep.py --epochs 2500 --latents 2 4 8 16 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from autoencoder.emoji_data import load_emojis
from autoencoder.vae import VAE
from autoencoder.vae_metrics_viz import active_units, plot_latent_sweep
from autoencoder.vae_train import train_vae


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Barrido de latent_dim del VAE")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="umbral de KL (nats) para contar una dim como activa")
    parser.add_argument("--latents", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--out-dir", default="out/vae_latent_sweep")
    args = parser.parse_args(argv)

    X, _ = load_emojis(size=args.size)
    print(f"barrido de latent_dim sobre {X.shape[0]} emojis, {args.epochs} épocas (β={args.beta})\n")
    print(f"{'latent':>8} {'recon':>10} {'kl':>9} {'activas':>9}")
    rows = []
    for latent in args.latents:
        vae = VAE([args.size * args.size, 256, 64], latent_dim=latent, activation="relu",
                  output_activation="sigmoid", init="he_normal", loss="bce",
                  seed=args.seed)
        f = train_vae(vae, X, epochs=args.epochs, lr=args.lr, beta=args.beta,
                      beta_warmup=args.beta_warmup, seed=args.seed)
        n_active = active_units(vae, X, threshold=args.threshold)
        print(f"{latent:>8} {f['recon_det']:>10.2f} {f['kl']:>9.3f} {n_active:>9}")
        rows.append({"latent_dim": latent, "recon": f["recon_det"], "kl": f["kl"],
                     "active_units": n_active})

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "latent_sweep.csv", index=False)
    plot_latent_sweep(df, path=out_dir / "latent_sweep.png")
    print(f"\n-> {out_dir}/latent_sweep.csv + latent_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
