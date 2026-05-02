"""Plots para presentar las técnicas de regularización del Ej3:

  1) L2 / weight decay sweep: muestra cómo el gap train-test cambia con λ.
     Sweet spot intermedio, underfitting si λ es muy alto.
  2) Augmentation sweep: muestra cómo aug más agresiva reduce overfitting
     hasta cierto punto.

Ambos plots usan los modelos ya entrenados de configs/ej3_more_digits/ensembles/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test, load_more_digits, to_one_hot
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"
_OUT_DIR = _ROOT / "outputs" / "ej3_more_digits" / "metrics"


def _split_train(seed=42, val_split=0.15):
    X, y = load_more_digits()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_split)
    train_idx = idx[n_val:]
    return X[train_idx], to_one_hot(y[train_idx], 10, encoding="zero_one")


def _eval_pair(model_name, X_train, Y_train, X_test, Y_test):
    mlp = MLP.load(str(_MODELS_DIR / f"{model_name}.npz"))
    return (
        mlp.evaluate(X_train, Y_train)["accuracy"],
        mlp.evaluate(X_test, Y_test)["accuracy"],
    )


def plot_sweep(ax, x_labels, x_pos, train_accs, test_accs, xlabel, title):
    gaps = [tr - te for tr, te in zip(train_accs, test_accs)]

    ax.plot(x_pos, train_accs, "o-", color="lightcoral", lw=2, ms=8,
            label="Train acc", markeredgecolor="black")
    ax.plot(x_pos, test_accs, "s-", color="steelblue", lw=2, ms=8,
            label="Test acc", markeredgecolor="black")

    # Gap shading
    ax.fill_between(x_pos, train_accs, test_accs, alpha=0.15, color="gray", label="Gap (overfitting)")

    # Anotar gap encima de cada par de puntos
    for xp, tr, g in zip(x_pos, train_accs, gaps):
        ax.text(xp, tr + 0.0015, f"Δ {g:+.3f}", ha="center", fontsize=8.5,
                color="darkred" if g > 0.015 else "dimgray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


def main():
    X_train, Y_train = _split_train()
    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    # === L2 sweep ===
    wd_models = [("1e-4", "wd_1e-4"), ("5e-4", "wd_5e-4"),
                 ("1e-3", "wd_1e-3"), ("5e-3", "wd_5e-3")]
    wd_train, wd_test = [], []
    for _, name in wd_models:
        tr, te = _eval_pair(name, X_train, Y_train, X_test, Y_test)
        wd_train.append(tr); wd_test.append(te)

    # === Aug sweep ===
    aug_models = [
        ("shifts", "aug_shifts"),
        ("rot 5°", "aug_rot5"),
        ("rot 10°+s", "aug_rot10"),
        ("rot 15°+s", "aug_rot15_scale"),
    ]
    aug_train, aug_test = [], []
    for _, name in aug_models:
        tr, te = _eval_pair(name, X_train, Y_train, X_test, Y_test)
        aug_train.append(tr); aug_test.append(te)

    # Plot
    fig, (ax_l2, ax_aug) = plt.subplots(1, 2, figsize=(14, 5.5))

    plot_sweep(ax_l2, [m[0] for m in wd_models], list(range(len(wd_models))),
               wd_train, wd_test,
               xlabel="weight_decay (λ)",
               title="L2 / Weight Decay")

    plot_sweep(ax_aug, [m[0] for m in aug_models], list(range(len(aug_models))),
               aug_train, aug_test,
               xlabel="Intensidad de augmentation (creciente)",
               title="Data Augmentation")

    fig.suptitle("Efecto de las técnicas de regularización sobre overfitting",
                 fontsize=14, y=1.01)

    out_path = _OUT_DIR / "regularization_sweeps.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Plot saved to {out_path}\n")

    # Tablas
    print("L2 sweep:")
    print(f"  {'λ':>6s} {'train':>8s} {'test':>8s} {'gap':>8s}")
    for (lab, _), tr, te in zip(wd_models, wd_train, wd_test):
        print(f"  {lab:>6s} {tr:>8.4f} {te:>8.4f} {tr - te:>+8.4f}")

    print("\nAug sweep:")
    print(f"  {'aug':>10s} {'train':>8s} {'test':>8s} {'gap':>8s}")
    for (lab, _), tr, te in zip(aug_models, aug_train, aug_test):
        print(f"  {lab:>10s} {tr:>8.4f} {te:>8.4f} {tr - te:>+8.4f}")


if __name__ == "__main__":
    main()
