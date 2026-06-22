"""Barrido de arquitectura del VAE: recon_train vs recon_val por configuración de capas ocultas.

Con latente y β ya fijados por los barridos anteriores, este script aísla el efecto de la
arquitectura en dos dimensiones separadas:
  - Ancho: una sola capa oculta variando entre --widths (default 128, 256, 512)
  - Profundidad: capas fijas de 512 variando la cantidad (512 / 512,256 / 512,256,128)

Genera dos gráficos separados: arch_sweep_width_<tag>.png y arch_sweep_depth_<tag>.png

    uv run python scripts/vae_arch_sweep.py --dataset emoji_multi --latent 16 --beta 1
    uv run python scripts/vae_arch_sweep.py --dataset fashion --latent 16 --beta 1 --batch 256
    uv run python scripts/vae_arch_sweep.py --dataset emoji_multi --latent 16 --beta 1 --widths 64 128 256 512 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from autoencoder.celeba_data import load_celeba
from autoencoder.emoji_data import load_emojis
from autoencoder.emoji_multi_data import load_multi_emojis
from autoencoder.minecraft_data import load_minecraft
from autoencoder.mnist_data import load_mnist
from autoencoder.vae import VAE
from autoencoder.vae_train import EarlyStopping, train_vae


def recon_px(vae, X):
    mu, _ = vae.encode(X)
    return float(vae._loss_value(vae.decode(mu), X))


def plot_arch_sweep(df, path, title):
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    x = range(len(df))
    labels = df["hidden"].tolist()
    ax.plot(x, df["recon_tr"], marker="o", label="train", color="tab:blue")
    ax.plot(x, df["recon_val"], marker="s", label="val", color="tab:orange")
    ax.fill_between(x, df["recon_tr"], df["recon_val"], alpha=0.15, color="tab:orange",
                    label="gap (overfitting)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("capas ocultas (encoder)")
    ax.set_ylabel("recon BCE / px (determinista, z=μ)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_one(hidden_str, args, Xtr, Xval, D):
    hidden = [int(h) for h in hidden_str.split(",")]
    vae = VAE([D, *hidden], latent_dim=args.latent, activation="relu",
              output_activation="sigmoid", init="he_normal", loss="bce", seed=args.seed)
    patience = args.patience if args.patience > 0 else args.epochs
    es = EarlyStopping(val_fn=lambda m: recon_px(m, Xval),
                       train_fn=lambda m: recon_px(m, Xtr),
                       patience=patience)
    train_vae(vae, Xtr, epochs=args.epochs, lr=args.lr, beta=args.beta,
              beta_warmup=args.beta_warmup, seed=args.seed,
              batch_size=args.batch, callback=es, callback_every=args.eval_every)
    if args.patience > 0:
        es.restore(vae)
    recon_tr = recon_px(vae, Xtr)
    recon_va = recon_px(vae, Xval)
    gap = recon_va - recon_tr
    stopped = es.stopped_epoch if es.stopped_epoch >= 0 else args.epochs
    print(f"{hidden_str:>25} {recon_tr:>10.4f} {recon_va:>11.4f} {gap:>8.4f} {stopped:>8}")
    return {"hidden": hidden_str, "recon_tr": recon_tr, "recon_val": recon_va,
            "gap": gap, "epochs": stopped, "n_params": vae.n_params}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Barrido de arquitectura del VAE (ancho y profundidad)")
    parser.add_argument("--dataset", choices=["emojis", "emoji_multi", "celeba", "mnist", "fashion", "minecraft", "minecraft-old"],
                        default="emojis", help="dataset a usar")
    parser.add_argument("--latent", type=int, default=16,
                        help="dimensión del espacio latente (fijar con el barrido de latente)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="β del ELBO (fijar con el barrido de β)")
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512],
                        help="anchos a probar en el barrido de ancho (una sola capa oculta)")
    parser.add_argument("--depths", nargs="+", default=["512", "512,256", "512,256,128"],
                        help="arquitecturas a probar en el barrido de profundidad (ej. '512' '512,256' '512,256,128')")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--color", action="store_true",
                        help="usa texturas RGB en vez de grises (solo para minecraft)")
    parser.add_argument("--blocks-only", action="store_true",
                        help="filtra escaleras, vallas, plantas, etc. (solo minecraft)")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stop: evaluaciones sin mejora (0=off)")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-dir", default="out/vae_arch_sweep")
    args = parser.parse_args(argv)
    tag = args.tag or f"{args.dataset}_L{args.latent}_b{args.beta}"

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = f"{'hidden':>25} {'recon_tr':>10} {'recon_val':>11} {'gap':>8} {'epocas':>8}"

    D = X.shape[1]  # size*size o size*size*3 según --color

    # --- barrido de ANCHO: una sola capa oculta ---
    width_configs = [str(w) for w in args.widths]
    print(f"\n== ANCHO ({args.dataset}, L={args.latent}, β={args.beta}) ==")
    print(header)
    rows_width = [train_one(h, args, Xtr, Xval, D) for h in width_configs]
    df_width = pd.DataFrame(rows_width)
    df_width.to_csv(out_dir / f"arch_sweep_width_{tag}.csv", index=False)
    plot_arch_sweep(df_width, path=out_dir / f"arch_sweep_width_{tag}.png",
                    title=f"Ancho ({args.dataset}, L={args.latent}, β={args.beta}): recon train vs val")

    # --- barrido de PROFUNDIDAD ---
    depth_configs = args.depths
    print(f"\n== PROFUNDIDAD ({args.dataset}, L={args.latent}, β={args.beta}) ==")
    print(header)
    rows_depth = [train_one(h, args, Xtr, Xval, D) for h in depth_configs]
    df_depth = pd.DataFrame(rows_depth)
    df_depth.to_csv(out_dir / f"arch_sweep_depth_{tag}.csv", index=False)
    plot_arch_sweep(df_depth, path=out_dir / f"arch_sweep_depth_{tag}.png",
                    title=f"Profundidad ({args.dataset}, L={args.latent}, β={args.beta}): recon train vs val")

    print(f"\n-> {out_dir}/arch_sweep_width_{tag}.png")
    print(f"-> {out_dir}/arch_sweep_depth_{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
