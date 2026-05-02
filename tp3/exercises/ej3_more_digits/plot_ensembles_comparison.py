"""Comparación entre 4 estrategias de ensemble.

Muestra para cada estrategia:
  - test accuracy de los 4 modelos individuales (barras claras)
  - test accuracy del ensemble que los combina (barra oscura)

Estrategias:
  1. Arquitecturas diversas (default, wide, deep, thin)
  2. 4 seeds, mismo config (42, 0, 7, 13)
  3. Variaciones de weight_decay (1e-4, 5e-4, 1e-3, 5e-3)
  4. Variaciones de augmentation (shifts, rot5, rot10, rot15+scale)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test, to_one_hot
from common.ensemble import Ensemble
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"
_OUT_PATH = _ROOT / "outputs" / "ej3_more_digits" / "metrics" / "ensembles_comparison.png"


_ENSEMBLES = {
    "Arquitecturas diversas": [
        ("default", "arch_default.npz"),
        ("wide",    "arch_wide.npz"),
        ("deep",    "arch_deep.npz"),
        ("thin",    "arch_thin.npz"),
    ],
    "4 seeds (mismo config)": [
        ("seed 42", "best_l2_aug_4seeds_seed42.npz"),
        ("seed 0",  "best_l2_aug_4seeds_seed0.npz"),
        ("seed 7",  "best_l2_aug_4seeds_seed7.npz"),
        ("seed 13", "best_l2_aug_4seeds_seed13.npz"),
    ],
    "Variaciones weight_decay": [
        ("1e-4", "wd_1e-4.npz"),
        ("5e-4", "wd_5e-4.npz"),
        ("1e-3", "wd_1e-3.npz"),
        ("5e-3", "wd_5e-3.npz"),
    ],
    "Variaciones augmentation": [
        ("shifts only", "aug_shifts.npz"),
        ("rot 5°",      "aug_rot5.npz"),
        ("rot 10°+s",   "aug_rot10.npz"),
        ("rot 15°+s",   "aug_rot15_scale.npz"),
    ],
}


def _eval(path, X, Y):
    return MLP.load(path).evaluate(X, Y)["accuracy"]


def main():
    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    summary = []
    for ax, (label, members) in zip(axes, _ENSEMBLES.items()):
        names = [n for n, _ in members]
        paths = [str(_MODELS_DIR / fn) for _, fn in members]

        # Individuals
        accs = [_eval(p, X_test, Y_test) for p in paths]
        # Ensemble
        ens = Ensemble.from_paths(paths)
        ens_acc = ens.evaluate(X_test, Y_test)["accuracy"]

        summary.append({
            "label": label,
            "individuals": list(zip(names, accs)),
            "ensemble": ens_acc,
        })

        x = np.arange(len(names) + 1)
        bar_accs = accs + [ens_acc]
        bar_labels = names + ["ENS"]
        colors = ["steelblue"] * len(names) + ["tomato"]

        bars = ax.bar(x, bar_accs, color=colors, edgecolor="black", linewidth=0.5)
        for bar, acc in zip(bars, bar_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.0003,
                    f"{acc:.4f}", ha="center", va="bottom", fontsize=8, rotation=0)

        ax.axhline(0.98, color="green", linestyle="--", lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Test accuracy")

    # Y limits: focus on relevant range
    ymin = min(min(s["individuals"], key=lambda t: t[1])[1] for s in summary) - 0.003
    ymax = max(s["ensemble"] for s in summary) + 0.005
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.suptitle("Comparación de estrategias de ensemble (test accuracy)", fontsize=13, y=1.02)
    plt.tight_layout()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to {_OUT_PATH}\n")

    # Console table
    print(f"{'Estrategia':<28s} {'min':>7s} {'max':>7s} {'mean':>7s} {'ENS':>7s} {'Δ vs best':>10s}")
    print("-" * 70)
    for s in summary:
        accs = [a for _, a in s["individuals"]]
        delta = s["ensemble"] - max(accs)
        sign = "+" if delta >= 0 else ""
        print(f"{s['label']:<28s} {min(accs):>7.4f} {max(accs):>7.4f} {np.mean(accs):>7.4f} "
              f"{s['ensemble']:>7.4f} {sign}{delta:>9.4f}")


if __name__ == "__main__":
    main()
