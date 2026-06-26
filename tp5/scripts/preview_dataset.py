"""Muestra un subset del dataset tal como lo descarga el loader.

Genera una grilla con los primeros N elementos (o una muestra aleatoria) del dataset
y la guarda como imagen. Útil para verificar que el dataset se cargó correctamente.

    uv run python scripts/preview_dataset.py --dataset minecraft --size 16 --color
    uv run python scripts/preview_dataset.py --dataset emoji_multi
    uv run python scripts/preview_dataset.py --dataset fashion --n 30
    uv run python scripts/preview_dataset.py --dataset celeba --size 28
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.celeba_data import load_celeba
from autoencoder.emoji_data import load_emojis
from autoencoder.emoji_multi_data import load_multi_emojis
from autoencoder.minecraft_data import load_minecraft
from autoencoder.mnist_data import load_mnist


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview de un dataset")
    parser.add_argument("--dataset", choices=["emojis", "emoji_multi", "celeba", "mnist", "fashion", "minecraft", "minecraft-old"],
                        default="emojis")
    parser.add_argument("--n", type=int, default=30, help="cantidad de imágenes a mostrar")
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--color", action="store_true",
                        help="usa texturas RGB (solo minecraft)")
    parser.add_argument("--blocks-only", action="store_true",
                        help="filtra escaleras, vallas, plantas, etc. (solo minecraft)")
    parser.add_argument("--max-n", type=int, default=None,
                        help="máximo de muestras a cargar del dataset completo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="out/preview")
    args = parser.parse_args(argv)

    # Cargar dataset
    if args.dataset == "celeba":
        X, labels = load_celeba(n=args.max_n or 3000, size=args.size, seed=args.seed)
    elif args.dataset in ("mnist", "fashion"):
        X, labels = load_mnist(n=args.max_n, seed=args.seed, kind=args.dataset)
    elif args.dataset == "emoji_multi":
        X, labels = load_multi_emojis(size=args.size, seed=args.seed)
    elif args.dataset in ("minecraft", "minecraft-old"):
        X, labels = load_minecraft(size=args.size, color=args.color, n=args.max_n,
                                   seed=args.seed, blocks_only=args.blocks_only,
                                   classic=(args.dataset == "minecraft-old"))
    else:
        X, labels = load_emojis(size=args.size)

    # Tomar subset aleatorio
    rng = np.random.default_rng(args.seed)
    n = min(args.n, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X_show, labels_show = X[idx], [labels[i] for i in idx]

    # Detectar si es color
    color = X.shape[1] == args.size * args.size * 3

    # Layout de la grilla
    ncols = min(10, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.4))
    axes = np.array(axes).reshape(nrows, ncols)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n:
            img = X_show[i]
            if color:
                ax.imshow(img.reshape(args.size, args.size, 3).clip(0, 1), interpolation="nearest")
            else:
                ax.imshow(img.reshape(args.size, args.size), cmap="Greys", vmin=0, vmax=1,
                          interpolation="nearest")
            ax.set_title(labels_show[i], fontsize=5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])

    mode = "rgb" if color else "gray"
    fig.suptitle(f"{args.dataset} — {n} muestras ({args.size}×{args.size} {mode})", fontsize=10)
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preview_{args.dataset}_{args.size}px_{mode}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
