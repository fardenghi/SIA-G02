"""Plot único para presentación: comparación de estrategias de ensemble +
indicadores de overfitting/underfitting.

Layout:
  Subplot 1 (arriba):  Test accuracy de cada ensemble (5 estrategias, incluyendo
                       el ensemble heterogéneo de 8 modelos que es el ganador).
  Subplot 2 (abajo):   Por estrategia, train_acc vs test_acc de cada modelo
                       constituyente. El gap (train - test) indica overfitting;
                       train_acc bajo de por sí indica underfitting.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test, load_more_digits, to_one_hot
from common.ensemble import Ensemble

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"
_METRICS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "metrics"
_OUT_PATH = _METRICS_DIR / "ensemble_story.png"


# Cuatro estrategias "single-axis" de diversidad
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

# Ensemble heterogéneo ganador: 3 modelos diversos en ejes complementarios
# (L2 fuerte, arquitectura profunda, sin L2). Encontrado por busqueda exhaustiva
# en ensemble_search.py → 99.04% test acc.
HETERO_MODELS = [
    "wd_1e-3",     # L2 fuerte (λ=1e-3) + Aug + ES, arch default
    "arch_deep",   # arch profunda 256x256x256x128, L2+Aug+ES
    "best",        # sin L2, solo Aug + ES, arch default
]
HETERO_LABEL = "Heterogéneo\n(3 modelos)"


def main():
    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    # Reproducir el mismo split train/val que usaron las corridas (seed=42, val_split=0.15)
    X_all, y_all = load_more_digits()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.15)
    train_idx = idx[n_val:]
    X_train = X_all[train_idx]
    Y_train = to_one_hot(y_all[train_idx], 10, encoding="zero_one")

    # Recolectar accuracies por ensemble
    rows = []
    for label, members in SINGLE_AXIS.items():
        files = [f for _, f in members]
        ens = Ensemble.from_paths([str(_MODELS_DIR / f"{f}.npz") for f in files])
        ens_test = ens.evaluate(X_test, Y_test)["accuracy"]
        ens_train = ens.evaluate(X_train, Y_train)["accuracy"]
        rows.append((label, ens_test, ens_train))

    hetero_ens = Ensemble.from_paths([str(_MODELS_DIR / f"{n}.npz") for n in HETERO_MODELS])
    hetero_test = hetero_ens.evaluate(X_test, Y_test)["accuracy"]
    hetero_train = hetero_ens.evaluate(X_train, Y_train)["accuracy"]
    rows.append((HETERO_LABEL, hetero_test, hetero_train))

    # Plot: solo el panel superior con barras + anotaciones
    fig, ax = plt.subplots(figsize=(13, 6))

    strategies = [r[0] for r in rows]
    test_accs = [r[1] for r in rows]
    train_accs = [r[2] for r in rows]
    gaps = [tr - te for te, tr in zip(test_accs, train_accs)]
    colors = ["steelblue"] * (len(rows) - 1) + ["goldenrod"]

    bars = ax.bar(strategies, test_accs, color=colors, edgecolor="black", linewidth=0.6)

    # Anotaciones: test arriba (label de la barra), train + gap adentro
    for bar, te, tr, g in zip(bars, test_accs, train_accs, gaps):
        x = bar.get_x() + bar.get_width() / 2
        # Test acc afuera, sobre la barra
        ax.text(x, te + 0.0004,
                f"test {te:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="black")
        # Train acc + gap adentro de la barra, cerca del tope
        ax.text(x, te - 0.0006,
                f"train {tr:.4f}\nΔ {g:+.4f}",
                ha="center", va="top", fontsize=8.5, color="white")

    ax.axhline(0.98, color="green", linestyle="--", lw=1.2, alpha=0.7, label="Goal 98%")
    ymin = min(test_accs) - 0.002
    ymax = max(test_accs) + 0.003
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Comparación de estrategias de ensemble en Ej3", fontsize=13, pad=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=10)

    fig.text(0.5, 0.005,
             "Δ = train_acc − test_acc.   Δ alto → overfitting.   "
             "train_acc bajo → underfitting.",
             ha="center", fontsize=9, style="italic", color="dimgray")
    plt.tight_layout()

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"Plot saved to {_OUT_PATH}\n")

    # Tabla resumen en consola
    print(f"{'Estrategia':<26s} {'test':>9s} {'train':>9s} {'Δ (gap)':>10s}")
    print("-" * 60)
    for label, te, tr in rows:
        clean = label.replace("\n", " ")
        gap = tr - te
        print(f"{clean:<26s} {te:>9.4f} {tr:>9.4f} {gap:>+10.4f}")


if __name__ == "__main__":
    main()
