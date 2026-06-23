"""Autoencoder clásico (Ej1) con soporte para datasets arbitrarios.

Entrena un AE MLP y emite los plots canónicos: reconstrucción, scatter latente 2D
e interpolación entre dos muestras elegidas por índice o por label.

    uv run python scripts/ae_run.py --dataset font
    uv run python scripts/ae_run.py --dataset fashion --latent 2 --hidden 256,64
    uv run python scripts/ae_run.py --dataset mnist --latent 2 --interp-from 3 --interp-to 7
    uv run python scripts/ae_run.py --dataset minecraft --size 16 --color --latent 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.celeba_data import load_celeba  # noqa: E402
from autoencoder.data import labels_for_subset, load_font, select_subset  # noqa: E402
from autoencoder.emoji_data import load_emojis  # noqa: E402
from autoencoder.emoji_multi_data import load_multi_emojis  # noqa: E402
from autoencoder.minecraft_data import load_minecraft  # noqa: E402
from autoencoder.mnist_data import load_mnist  # noqa: E402
from autoencoder.network import Autoencoder  # noqa: E402
from autoencoder.train import train_multi_restart  # noqa: E402


# ---- visualización genérica (funciona con font 35px y con imágenes NxN) -------------- #

def _show(ax, vec, size, color):
    ax.set_xticks([]); ax.set_yticks([])
    if size == 0:
        # Font: vector de 35 → grilla 7×5
        from autoencoder.data import to_grid
        ax.imshow(to_grid(vec), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    elif color:
        ax.imshow(vec.reshape(size, size, 3).clip(0, 1), interpolation="nearest")
    else:
        ax.imshow(vec.reshape(size, size), cmap="Greys", vmin=0, vmax=1,
                  interpolation="nearest")


def plot_reconstruction(net, X, labels, size, color, n, path, binarize=True):
    n = min(n, X.shape[0])
    raw = net.forward(X[:n])
    recon = (raw >= 0.5).astype(float) if binarize else raw.clip(0, 1)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 3.2))
    axes = np.atleast_2d(axes)
    for i in range(n):
        _show(axes[0, i], X[i], size, color)
        _show(axes[1, i], recon[i], size, color)
        if labels is not None:
            axes[0, i].set_title(str(labels[i]), fontsize=7)
    axes[0, 0].set_ylabel("in", rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("recon", rotation=0, ha="right", va="center")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"-> {path}")


def plot_latent_scatter(Z, labels, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Z[:, 0], Z[:, 1], c="tab:blue", s=40, alpha=0.7)
    if labels is not None:
        for (x, y), lab in zip(Z, labels):
            ax.annotate(str(lab), (x, y), fontsize=9,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.set_title("Espacio latente 2D"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"-> {path}")


def plot_interpolation(net, X, labels, from_id, to_id, steps, size, color, path,
                       binarize=True):
    """Interpolación lineal en el latente. from_id/to_id pueden ser índices o labels."""
    label_list = [str(l) for l in labels] if labels is not None else []

    def resolve(id_):
        if isinstance(id_, int) or (isinstance(id_, str) and id_.lstrip("-").isdigit()):
            return int(id_)
        if id_ in label_list:
            return label_list.index(id_)
        raise ValueError(f"'{id_}' no encontrado. Labels disponibles: {label_list}")

    i, j = resolve(from_id), resolve(to_id)
    Z = net.encode(X)
    ts = np.linspace(0.0, 1.0, steps)
    zs = np.stack([(1 - t) * Z[i] + t * Z[j] for t in ts])
    raw = net.decode(zs)
    imgs = (raw >= 0.5).astype(float) if binarize else raw.clip(0, 1)

    from_lbl = label_list[i] if label_list else str(i)
    to_lbl = label_list[j] if label_list else str(j)
    step_labels = [f"'{from_lbl}'"] + ["nueva"] * (steps - 2) + [f"'{to_lbl}'"]

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.4, 2.2))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        _show(ax, imgs[k], size, color)
        ax.set_title(step_labels[k], fontsize=8)
    fig.suptitle(f"Interpolación en el latente: '{from_lbl}' -> '{to_lbl}'")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"-> {path}")


def plot_latent_grid(net, Z, size, color, steps, path, z1_range=None, z2_range=None):
    """Decodea una grilla regular en el espacio latente 2D."""
    z1_min, z1_max = z1_range if z1_range else (Z[:, 0].min(), Z[:, 0].max())
    z2_min, z2_max = z2_range if z2_range else (Z[:, 1].min(), Z[:, 1].max())

    z1s = np.linspace(z1_min, z1_max, steps)
    z2s = np.linspace(z2_max, z2_min, steps)  # invertido: arriba = z2 grande
    grid_z = np.array([[z1, z2] for z2 in z2s for z1 in z1s])
    imgs = net.decode(grid_z).clip(0, 1)

    fig, axes = plt.subplots(steps, steps, figsize=(steps * 1.2, steps * 1.2))
    for idx, ax in enumerate(axes.flat):
        _show(ax, imgs[idx], size, color)
    z1_label = f"z1: [{z1_min:.1f}, {z1_max:.1f}]"
    z2_label = f"z2: [{z2_min:.1f}, {z2_max:.1f}]"
    fig.suptitle(f"Grid latente  {z1_label}  {z2_label}", fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"-> {path}")


# ---- main ---------------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description="AE clásico con datasets arbitrarios")
    p.add_argument("--dataset",
                   choices=["font", "mnist", "fashion", "celeba",
                            "emojis", "emoji_multi", "minecraft", "minecraft-old"],
                   default="font")
    p.add_argument("--size", type=int, default=28,
                   help="resolución de imagen (ignorado para font)")
    p.add_argument("--color", action="store_true",
                   help="texturas RGB (solo minecraft)")
    p.add_argument("--blocks-only", action="store_true",
                   help="solo bloques sólidos (solo minecraft)")
    p.add_argument("--max-n", type=int, default=None,
                   help="máximo de muestras a cargar")
    p.add_argument("--latent", type=int, default=2,
                   help="dimensión del espacio latente")
    p.add_argument("--hidden", default=None,
                   help="capas ocultas del encoder, coma-sep (ej: 256,64). "
                        "None = una capa de 64 para font, 256,64 para el resto")
    p.add_argument("--activation", default="tanh")
    p.add_argument("--output-activation", default="sigmoid")
    p.add_argument("--init", default="xavier_normal")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interp-from", default=None,
                   help="origen de la interpolación (label o índice)")
    p.add_argument("--interp-to", default=None,
                   help="destino de la interpolación (label o índice)")
    p.add_argument("--interp-steps", type=int, default=7)
    p.add_argument("--n-recon", type=int, default=16,
                   help="cantidad de muestras en el plot de reconstrucción")
    p.add_argument("--grid", action="store_true",
                   help="genera grid del espacio latente 2D")
    p.add_argument("--grid-steps", type=int, default=10,
                   help="resolución de la grilla (NxN)")
    p.add_argument("--grid-z1", default=None,
                   help="rango manual de z1: 'min,max' (ej: 0,50)")
    p.add_argument("--grid-z2", default=None,
                   help="rango manual de z2: 'min,max' (ej: 0,120)")
    p.add_argument("--out-dir", default="out/ae_run")
    args = p.parse_args(argv)

    # ---- cargar dataset ------------------------------------------------------------ #
    color = False
    if args.dataset == "font":
        X_full = load_font("font/font.h")
        X = select_subset(X_full, None)
        labels = labels_for_subset(None)
        D = X.shape[1]   # 35
        size = 0          # señal para usar to_grid en viz
    elif args.dataset in ("mnist", "fashion"):
        X, labels = load_mnist(n=args.max_n, seed=args.seed, kind=args.dataset)
        D, size = X.shape[1], args.size
    elif args.dataset == "celeba":
        X, labels = load_celeba(n=args.max_n or 3000, size=args.size, seed=args.seed)
        D, size = X.shape[1], args.size
    elif args.dataset == "emoji_multi":
        X, labels = load_multi_emojis(size=args.size, seed=args.seed)
        D, size = X.shape[1], args.size
    elif args.dataset == "emojis":
        X, labels = load_emojis(size=args.size)
        D, size = X.shape[1], args.size
    else:  # minecraft / minecraft-old
        color = args.color
        X, labels = load_minecraft(size=args.size, color=color, n=args.max_n,
                                   seed=args.seed, blocks_only=args.blocks_only,
                                   classic=(args.dataset == "minecraft-old"))
        D, size = X.shape[1], args.size

    # ---- arquitectura -------------------------------------------------------------- #
    if args.hidden:
        hidden = [int(h) for h in args.hidden.split(",")]
    else:
        hidden = [64] if args.dataset == "font" else [256, 64]
    encoder_layers = [D] + hidden + [args.latent]

    print(f"== AE clásico: {args.dataset} ==")
    print(f"  encoder: {encoder_layers}  act={args.activation}  "
          f"salida={args.output_activation}  init={args.init}")
    print(f"  training: adam  epochs={args.epochs}  lr={args.lr}  restarts={args.restarts}")
    print(f"  dataset: {X.shape[0]} muestras de {D} px")

    # ---- entrenar ------------------------------------------------------------------ #
    best_net, _, _ = train_multi_restart(
        X,
        encoder_layers=encoder_layers,
        activation=args.activation,
        output_activation=args.output_activation,
        init=args.init,
        loss="bce",
        optimizer="adam",
        epochs=args.epochs,
        lr=args.lr,
        restarts=args.restarts,
        seed=args.seed,
        stop_at=None,
    )

    # ---- plots --------------------------------------------------------------------- #
    out = Path(args.out_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    binarize = (args.dataset == "font")

    plot_reconstruction(best_net, X, labels, size, color, args.n_recon,
                        out / "reconstruction.png", binarize=binarize)

    Z = best_net.encode(X)
    if args.latent == 2:
        plot_latent_scatter(Z, labels, out / "latent_scatter.png")

    if args.interp_from and args.interp_to:
        plot_interpolation(best_net, X, labels,
                           args.interp_from, args.interp_to, args.interp_steps,
                           size, color, out / "interpolation.png", binarize=binarize)
    elif X.shape[0] >= 2:
        plot_interpolation(best_net, X, labels,
                           0, X.shape[0] // 2, args.interp_steps,
                           size, color, out / "interpolation.png", binarize=binarize)

    if args.grid and args.latent == 2:
        z1_range = tuple(float(v) for v in args.grid_z1.split(",")) if args.grid_z1 else None
        z2_range = tuple(float(v) for v in args.grid_z2.split(",")) if args.grid_z2 else None
        plot_latent_grid(best_net, Z, size, color, args.grid_steps,
                         out / "latent_grid.png", z1_range=z1_range, z2_range=z2_range)


if __name__ == "__main__":
    main()
