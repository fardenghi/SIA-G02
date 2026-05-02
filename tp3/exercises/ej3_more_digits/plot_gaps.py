"""Plot de los dos gaps clave de overfitting (accuracy y loss),
comparando 'sin regularización' vs 'con regularización (L2 + Aug + ES)'.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test, load_more_digits, to_one_hot
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS = _ROOT / "outputs" / "ej3_more_digits" / "models"
_OUT = _ROOT / "outputs" / "ej3_more_digits" / "metrics" / "regularizacion_gaps.png"


def _eval(model_path, X_train, Y_train, X_test, Y_test):
    mlp = MLP.load(str(model_path))
    tr = mlp.evaluate(X_train, Y_train)
    te = mlp.evaluate(X_test, Y_test)
    return tr, te


def main():
    X_all, y_all = load_more_digits()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.15)
    X_train = X_all[idx[n_val:]]
    Y_train = to_one_hot(y_all[idx[n_val:]], 10, encoding="zero_one")
    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    sin_tr, sin_te = _eval(_MODELS / "baseline_pure.npz", X_train, Y_train, X_test, Y_test)
    con_tr, con_te = _eval(_MODELS / "best_l2_aug.npz", X_train, Y_train, X_test, Y_test)

    sin_acc_gap = sin_tr["accuracy"] - sin_te["accuracy"]
    con_acc_gap = con_tr["accuracy"] - con_te["accuracy"]
    sin_loss_gap = sin_te["loss"] - sin_tr["loss"]
    con_loss_gap = con_te["loss"] - con_tr["loss"]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ===== Accuracy gap =====
    bars1 = ax_acc.bar(["SIN regularización", "CON regularización"],
                       [sin_acc_gap, con_acc_gap],
                       color=["#a83232", "#3262a8"], edgecolor="black", linewidth=0.6,
                       width=0.55)
    for bar, v in zip(bars1, [sin_acc_gap, con_acc_gap]):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
    pct_acc = (sin_acc_gap - con_acc_gap) / sin_acc_gap * 100
    ax_acc.text(0.5, max(sin_acc_gap, con_acc_gap) * 0.55,
                f"−{pct_acc:.0f}%", ha="center", va="center",
                fontsize=20, fontweight="bold", color="darkgreen",
                transform=ax_acc.transData)
    ax_acc.annotate("", xy=(0.95, con_acc_gap + 0.001), xytext=(0.05, sin_acc_gap - 0.001),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))
    ax_acc.set_ylabel("Gap de accuracy (train − test)", fontsize=11)
    ax_acc.set_title("Gap de accuracy", fontsize=13)
    ax_acc.set_ylim(0, sin_acc_gap * 1.15)
    ax_acc.grid(axis="y", alpha=0.3)
    ax_acc.set_axisbelow(True)

    # ===== Loss gap =====
    bars2 = ax_loss.bar(["SIN regularización", "CON regularización"],
                        [sin_loss_gap, con_loss_gap],
                        color=["#a83232", "#3262a8"], edgecolor="black", linewidth=0.6,
                        width=0.55)
    for bar, v in zip(bars2, [sin_loss_gap, con_loss_gap]):
        ax_loss.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f"{v:.4f}", ha="center", va="bottom",
                     fontsize=12, fontweight="bold")
    pct_loss = (sin_loss_gap - con_loss_gap) / sin_loss_gap * 100
    ax_loss.text(0.5, max(sin_loss_gap, con_loss_gap) * 0.55,
                 f"−{pct_loss:.0f}%", ha="center", va="center",
                 fontsize=20, fontweight="bold", color="darkgreen",
                 transform=ax_loss.transData)
    ax_loss.annotate("", xy=(0.95, con_loss_gap + 0.005), xytext=(0.05, sin_loss_gap - 0.005),
                     arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))
    ax_loss.set_ylabel("Gap de loss (test − train)", fontsize=11)
    ax_loss.set_title("Gap de loss", fontsize=13)
    ax_loss.set_ylim(0, sin_loss_gap * 1.15)
    ax_loss.grid(axis="y", alpha=0.3)
    ax_loss.set_axisbelow(True)

    fig.suptitle("Reducción del overfitting con regularización (L2 + Aug + Early Stopping)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_OUT, dpi=130, bbox_inches="tight")
    print(f"Plot saved to {_OUT}")
    print(f"  acc_gap:  {sin_acc_gap:.4f} → {con_acc_gap:.4f}  ({-pct_acc:+.0f}%)")
    print(f"  loss_gap: {sin_loss_gap:.4f} → {con_loss_gap:.4f}  ({-pct_loss:+.0f}%)")


if __name__ == "__main__":
    main()
