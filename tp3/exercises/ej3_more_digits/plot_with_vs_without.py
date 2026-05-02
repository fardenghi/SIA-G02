"""Plot 'con vs sin' una técnica de regularización.

Compara dos modelos guardados (uno SIN la técnica, otro CON la técnica) y
muestra el efecto sobre overfitting con barras agrupadas. Reporta tanto
accuracy como loss, ya que algunas técnicas (como L2) tienen efecto
mayormente sobre la loss y poco sobre el argmax-accuracy.

Usage:
    uv run python -m exercises.ej3_more_digits.plot_with_vs_without \\
        --without outputs/.../baseline_pure.npz \\
        --with    outputs/.../baseline_only_l2.npz \\
        --label   "L2 / Weight Decay" \\
        --out     outputs/.../l2_with_vs_without.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test, load_more_digits, to_one_hot
from common.mlp import MLP


def _evaluate_model(model_path, X_train, Y_train, X_test, Y_test):
    mlp = MLP.load(model_path)
    tr = mlp.evaluate(X_train, Y_train)
    te = mlp.evaluate(X_test, Y_test)
    return {
        "train_acc": tr["accuracy"],
        "train_loss": tr["loss"],
        "test_acc": te["accuracy"],
        "test_loss": te["loss"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--without", required=True, help="Modelo .npz SIN la técnica")
    parser.add_argument("--with_", "--with", dest="with_", required=True, help="Modelo .npz CON la técnica")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Datos: mismo split que se usó para entrenar (seed=42, val_split=0.15)
    X_all, y_all = load_more_digits()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.15)
    X_train = X_all[idx[n_val:]]
    Y_train = to_one_hot(y_all[idx[n_val:]], 10, encoding="zero_one")

    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    sin = _evaluate_model(args.without, X_train, Y_train, X_test, Y_test)
    con = _evaluate_model(args.with_, X_train, Y_train, X_test, Y_test)

    sin_acc_gap = sin["train_acc"] - sin["test_acc"]
    con_acc_gap = con["train_acc"] - con["test_acc"]
    sin_loss_gap = sin["test_loss"] - sin["train_loss"]
    con_loss_gap = con["test_loss"] - con["train_loss"]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Panel izquierdo: accuracy =====
    x_sin, x_con = [0, 1], [3, 4]
    ax_acc.bar(x_sin, [sin["train_acc"], sin["test_acc"]], width=0.7,
               color=["#d97a7a", "#a83232"], edgecolor="black", linewidth=0.6,
               label=f"SIN {args.label}")
    ax_acc.bar(x_con, [con["train_acc"], con["test_acc"]], width=0.7,
               color=["#7aa3d9", "#3262a8"], edgecolor="black", linewidth=0.6,
               label=f"CON {args.label}")

    for x, v in zip(x_sin + x_con,
                    [sin["train_acc"], sin["test_acc"], con["train_acc"], con["test_acc"]]):
        ax_acc.text(x, v - 0.005, f"{v:.4f}", ha="center", va="top",
                    fontsize=9, color="white", fontweight="bold")

    # Anotación de gap (flechita)
    def _arrow_gap(ax, x_left, top, bottom, gap, color):
        ax.annotate("", xy=(x_left, top), xytext=(x_left, bottom),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2))
        ax.text(x_left + 0.35, (top + bottom) / 2,
                f"gap\n{gap:+.4f}", ha="left", va="center",
                fontsize=10, color=color, fontweight="bold")

    _arrow_gap(ax_acc, x_sin[0] - 0.55, sin["train_acc"], sin["test_acc"], sin_acc_gap, "darkred")
    _arrow_gap(ax_acc, x_con[0] - 0.55, con["train_acc"], con["test_acc"], con_acc_gap, "navy")

    ax_acc.set_xticks([0, 1, 3, 4])
    ax_acc.set_xticklabels(["train", "test", "train", "test"], fontsize=9.5)
    ax_acc.text(0.5, ax_acc.get_ylim()[0] - 0.02, f"SIN {args.label}",
                ha="center", fontsize=11, fontweight="bold", color="darkred",
                transform=ax_acc.get_xaxis_transform())
    ax_acc.text(3.5, ax_acc.get_ylim()[0] - 0.02, f"CON {args.label}",
                ha="center", fontsize=11, fontweight="bold", color="navy",
                transform=ax_acc.get_xaxis_transform())

    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.set_title("Accuracy y gap (overfitting)", fontsize=12)
    ax_acc.set_ylim(min(sin["test_acc"], con["test_acc"]) - 0.02, 1.005)
    ax_acc.grid(axis="y", alpha=0.3)
    ax_acc.set_axisbelow(True)

    # ===== Panel derecho: loss =====
    ax_loss.bar(x_sin, [sin["train_loss"], sin["test_loss"]], width=0.7,
                color=["#d97a7a", "#a83232"], edgecolor="black", linewidth=0.6)
    ax_loss.bar(x_con, [con["train_loss"], con["test_loss"]], width=0.7,
                color=["#7aa3d9", "#3262a8"], edgecolor="black", linewidth=0.6)

    for x, v in zip(x_sin + x_con,
                    [sin["train_loss"], sin["test_loss"], con["train_loss"], con["test_loss"]]):
        offset = 0.005 * max(sin["test_loss"], con["test_loss"])
        ax_loss.text(x, v + offset, f"{v:.4f}", ha="center", va="bottom",
                     fontsize=9, color="black", fontweight="bold")

    _arrow_gap(ax_loss, x_sin[0] - 0.55, sin["test_loss"], sin["train_loss"], sin_loss_gap, "darkred")
    _arrow_gap(ax_loss, x_con[0] - 0.55, con["test_loss"], con["train_loss"], con_loss_gap, "navy")

    ax_loss.set_xticks([0, 1, 3, 4])
    ax_loss.set_xticklabels(["train", "test", "train", "test"], fontsize=9.5)
    ax_loss.text(0.5, ax_loss.get_ylim()[0] - 0.02, f"SIN {args.label}",
                 ha="center", fontsize=11, fontweight="bold", color="darkred",
                 transform=ax_loss.get_xaxis_transform())
    ax_loss.text(3.5, ax_loss.get_ylim()[0] - 0.02, f"CON {args.label}",
                 ha="center", fontsize=11, fontweight="bold", color="navy",
                 transform=ax_loss.get_xaxis_transform())

    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.set_title("Loss y gap", fontsize=12)
    ax_loss.set_ylim(0, max(sin["test_loss"], con["test_loss"]) * 1.15)
    ax_loss.grid(axis="y", alpha=0.3)
    ax_loss.set_axisbelow(True)

    fig.suptitle(f"Efecto de {args.label} sobre overfitting (todo lo demás idéntico)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Plot saved to {out}")
    print(f"  SIN {args.label}: train_acc={sin['train_acc']:.4f} test_acc={sin['test_acc']:.4f}  "
          f"acc_gap={sin_acc_gap:+.4f}  loss_gap={sin_loss_gap:+.4f}")
    print(f"  CON {args.label}: train_acc={con['train_acc']:.4f} test_acc={con['test_acc']:.4f}  "
          f"acc_gap={con_acc_gap:+.4f}  loss_gap={con_loss_gap:+.4f}")


if __name__ == "__main__":
    main()
