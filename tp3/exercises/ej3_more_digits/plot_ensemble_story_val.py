"""Versión 'val' de plot_ensemble_story.py: evalúa cada estrategia de ensemble
contra el split de validación interno (15% de more_digits.csv, seed=42) en lugar
de digits_test.csv. digits_test.csv se reserva para reportar el modelo final
elegido como métrica de producción.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_more_digits, to_one_hot
from common.ensemble import Ensemble

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"
_METRICS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "metrics"
_OUT_PATH = _METRICS_DIR / "ensemble_story_val.png"


SINGLE_AXIS = {
    "Arquitecturas\ndiversas": [
        ("default", "arch_default"),
        ("wide",    "arch_wide"),
        ("deep",    "arch_deep"),
        ("thin",    "arch_thin"),
    ],
    "4 seeds\n(mismo config)": [
        ("seed 42", "best_l2_aug_4seeds_seed42"),
        ("seed 0",  "best_l2_aug_4seeds_seed0"),
        ("seed 7",  "best_l2_aug_4seeds_seed7"),
        ("seed 13", "best_l2_aug_4seeds_seed13"),
    ],
    "Variaciones\nweight decay": [
        ("1e-4", "wd_1e-4"),
        ("5e-4", "wd_5e-4"),
        ("1e-3", "wd_1e-3"),
        ("5e-3", "wd_5e-3"),
    ],
    "Variaciones\naugmentation": [
        ("shifts", "aug_shifts"),
        ("rot 5°", "aug_rot5"),
        ("rot 10°+s", "aug_rot10"),
        ("rot 15°+s", "aug_rot15_scale"),
    ],
}

HETERO_MODELS = [
    "wd_1e-3",
    "arch_deep",
    "best",
]
HETERO_LABEL = "Heterogéneo\n(3 modelos)"


def main():
    X_all, y_all = load_more_digits()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.15)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    X_train = X_all[train_idx]
    Y_train = to_one_hot(y_all[train_idx], 10, encoding="zero_one")
    X_val = X_all[val_idx]
    Y_val = to_one_hot(y_all[val_idx], 10, encoding="zero_one")

    rows = []
    for label, members in SINGLE_AXIS.items():
        files = [f for _, f in members]
        ens = Ensemble.from_paths([str(_MODELS_DIR / f"{f}.npz") for f in files])
        ens_val = ens.evaluate(X_val, Y_val)["accuracy"]
        ens_train = ens.evaluate(X_train, Y_train)["accuracy"]
        rows.append((label, ens_val, ens_train))

    hetero_ens = Ensemble.from_paths([str(_MODELS_DIR / f"{n}.npz") for n in HETERO_MODELS])
    hetero_val = hetero_ens.evaluate(X_val, Y_val)["accuracy"]
    hetero_train = hetero_ens.evaluate(X_train, Y_train)["accuracy"]
    rows.append((HETERO_LABEL, hetero_val, hetero_train))

    fig, ax = plt.subplots(figsize=(13, 6))

    strategies = [r[0] for r in rows]
    val_accs = [r[1] for r in rows]
    train_accs = [r[2] for r in rows]
    gaps = [tr - va for va, tr in zip(val_accs, train_accs)]
    best_idx = int(np.argmax(val_accs))
    colors = ["goldenrod" if i == best_idx else "steelblue" for i in range(len(rows))]

    bars = ax.bar(strategies, val_accs, color=colors, edgecolor="black", linewidth=0.6)

    for bar, va, tr, g in zip(bars, val_accs, train_accs, gaps):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, va + 0.0004,
                f"val {va:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="black")
        ax.text(x, va - 0.0006,
                f"train {tr:.4f}\nΔ {g:+.4f}",
                ha="center", va="top", fontsize=8.5, color="white")

    ax.axhline(0.98, color="green", linestyle="--", lw=1.2, alpha=0.7, label="Goal 98%")
    ymin = min(val_accs) - 0.002
    ymax = max(val_accs) + 0.003
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Validation accuracy", fontsize=11)
    ax.set_title("Comparación de estrategias de ensemble en Ej3 (validation interno, sin digits_test)",
                 fontsize=12.5, pad=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=10)

    fig.text(0.5, 0.005,
             "Δ = train_acc − val_acc.   Δ alto → overfitting.   "
             "train_acc bajo → underfitting.",
             ha="center", fontsize=9, style="italic", color="dimgray")
    plt.tight_layout()

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"Plot saved to {_OUT_PATH}\n")

    print(f"{'Estrategia':<26s} {'val':>9s} {'train':>9s} {'Δ (gap)':>10s}")
    print("-" * 60)
    for label, va, tr in rows:
        clean = label.replace("\n", " ")
        gap = tr - va
        print(f"{clean:<26s} {va:>9.4f} {tr:>9.4f} {gap:>+10.4f}")


if __name__ == "__main__":
    main()
