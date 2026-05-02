"""Bar chart comparando test accuracy entre modelos individuales y ensembles.

Carga modelos guardados, evalúa cada uno en digits_test.csv y plotea
un bar chart con los resultados.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test, to_one_hot
from common.ensemble import Ensemble
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"
_OUT_PATH = _ROOT / "outputs" / "ej3_more_digits" / "metrics" / "test_acc_comparison.png"


def main():
    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    models = {
        "best": ["best.npz"],
        "best_decay": ["best_decay.npz"],
        "best_l2": ["best_l2(+pat_weight).npz"],
        "best_l2_aug": ["best_l2_aug.npz"],
        "l2_4s_42": ["best_l2_4seeds_seed42.npz"],
        "l2_4s_0":  ["best_l2_4seeds_seed0.npz"],
        "l2_4s_7":  ["best_l2_4seeds_seed7.npz"],
        "l2_4s_13": ["best_l2_4seeds_seed13.npz"],
        "aug_4s_42": ["best_l2_aug_4seeds_seed42.npz"],
        "aug_4s_0":  ["best_l2_aug_4seeds_seed0.npz"],
        "aug_4s_7":  ["best_l2_aug_4seeds_seed7.npz"],
        "aug_4s_13": ["best_l2_aug_4seeds_seed13.npz"],
        "ENS 4 seeds": [
            "best_l2_4seeds_seed42.npz", "best_l2_4seeds_seed0.npz",
            "best_l2_4seeds_seed7.npz", "best_l2_4seeds_seed13.npz",
        ],
        "ENS aug+4 seeds": [
            "best_l2_aug_4seeds_seed42.npz", "best_l2_aug_4seeds_seed0.npz",
            "best_l2_aug_4seeds_seed7.npz", "best_l2_aug_4seeds_seed13.npz",
        ],
        "ENS 8 diversos": [
            "best.npz", "best_decay.npz", "best_l2(+pat_weight).npz", "best_l2_aug.npz",
            "best_l2_4seeds_seed42.npz", "best_l2_4seeds_seed0.npz",
            "best_l2_4seeds_seed7.npz", "best_l2_4seeds_seed13.npz",
        ],
    }

    names, accs, kinds = [], [], []
    for name, files in models.items():
        paths = [str(_MODELS_DIR / f) for f in files]
        if len(paths) == 1:
            mlp = MLP.load(paths[0])
            kind = "individual"
        else:
            mlp = Ensemble.from_paths(paths)
            kind = "ensemble"
        m = mlp.evaluate(X_test, Y_test)
        names.append(name)
        accs.append(m["accuracy"])
        kinds.append(kind)
        print(f"  {name:<20s} {kind:<10s}  test_acc={m['accuracy']:.4f}")

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["steelblue" if k == "individual" else "tomato" for k in kinds]
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)

    # Goal line
    ax.axhline(0.98, color="green", linestyle="--", lw=1.5, alpha=0.7, label="Goal 98%")

    # Labels on top of bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.0005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.97, 0.995)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Comparación de test accuracy: modelos individuales vs ensembles")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="Individual"),
        Patch(facecolor="tomato", edgecolor="black", label="Ensemble"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="green", linestyle="--", label="Goal 98%")
    ], loc="lower right")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=120)
    print(f"\nPlot saved to {_OUT_PATH}")


if __name__ == "__main__":
    main()
