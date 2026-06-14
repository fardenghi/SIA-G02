"""Barrido de β del VAE: entrena un modelo por cada β y grafica recon/KL vs β.

Reproduce la ablación del README (frontera del trade-off + zona de colapso). Entrena varios
modelos, así que es un script aparte del CLI normal.

    uv run python scripts/vae_beta_sweep.py
    uv run python scripts/vae_beta_sweep.py --epochs 2500 --betas 1 8 16 100 784
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from autoencoder.emoji_data import load_emojis
from autoencoder.vae import VAE
from autoencoder.vae_metrics_viz import plot_beta_sweep
from autoencoder.vae_train import train_vae


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Barrido de β del VAE")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 784.0])
    parser.add_argument("--out-dir", default="out/vae_beta_sweep")
    args = parser.parse_args(argv)

    X, _ = load_emojis(size=args.size)
    print(f"barrido de β sobre {X.shape[0]} emojis, {args.epochs} épocas\n")
    print(f"{'beta':>8} {'recon':>10} {'kl':>9}  diagnóstico")
    rows = []
    for beta in args.betas:
        vae = VAE([args.size * args.size, 256, 64], latent_dim=2, activation="relu",
                  output_activation="sigmoid", init="he_normal", loss="bce",
                  seed=args.seed)
        f = train_vae(vae, X, epochs=args.epochs, lr=args.lr, beta=beta,
                      beta_warmup=args.beta_warmup, seed=args.seed)
        diag = "COLAPSO" if f["kl"] < 0.05 else "sano"
        print(f"{beta:>8.2f} {f['recon_det']:>10.2f} {f['kl']:>9.3f}  {diag}")
        rows.append({"beta": beta, "recon": f["recon_det"], "kl": f["kl"]})

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "beta_sweep.csv", index=False)
    plot_beta_sweep(df, path=out_dir / "beta_sweep.png")
    print(f"\n-> {out_dir}/beta_sweep.csv + beta_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
