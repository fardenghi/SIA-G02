"""Ensemble heterogeneo: combina modelos de arquitecturas/regularizaciones
distintas (no solo seeds). Promedia probabilidades softmax y reporta test acc
sobre digits_test.csv.

Uso:
    uv run python -m exercises.ej3_more_digits.ensemble_heterogeneo
"""
from pathlib import Path

import numpy as np

from common.datasets import load_digits_test, to_one_hot
from common.mlp import MLP


_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"


# Combos a probar — cada lista es una variante de ensemble heterogeneo
COMBOS = {
    "4_arch (thin+default+wide+deep)": [
        "arch_thin", "arch_default", "arch_wide", "arch_deep",
    ],
    "best_l2_aug + best (no L2) + softmax (tanh)": [
        "best_l2_aug", "best", "softmax",
    ],
    "wd_sweep (1e-4, 5e-4, 1e-3)": [
        "wd_1e-4", "wd_5e-4", "wd_1e-3",
    ],
    "aug_sweep (rot5, rot10, rot15, shifts)": [
        "aug_rot5", "aug_rot10", "aug_rot15_scale", "aug_shifts",
    ],
    "MEGA (4_arch + 4_aug + 3_wd) — 11 modelos diversos": [
        "arch_thin", "arch_default", "arch_wide", "arch_deep",
        "aug_rot5", "aug_rot10", "aug_rot15_scale", "aug_shifts",
        "wd_1e-4", "wd_5e-4", "wd_1e-3",
    ],
}


def evaluate_ensemble(model_names, X_test, y_test, n_classes=10):
    probs_list = []
    individual_accs = {}
    for name in model_names:
        path = _MODELS_DIR / f"{name}.npz"
        if not path.exists():
            print(f"  [skip] {path.name} no existe")
            continue
        mlp = MLP.load(str(path))
        probs = mlp.forward(X_test)
        pred = np.argmax(probs, axis=1)
        individual_accs[name] = float(np.mean(pred == y_test))
        probs_list.append(probs)

    if len(probs_list) < 2:
        return None, individual_accs

    avg_probs = np.mean(probs_list, axis=0)
    pred = np.argmax(avg_probs, axis=1)
    ens_acc = float(np.mean(pred == y_test))
    return ens_acc, individual_accs


def main():
    X_test, y_test = load_digits_test()

    print("=" * 80)
    print(f"ENSEMBLE HETEROGENEO  ({len(y_test)} muestras de digits_test.csv)")
    print("=" * 80)

    results = []
    for label, names in COMBOS.items():
        print(f"\n--- {label} ---")
        ens_acc, individual = evaluate_ensemble(names, X_test, y_test)
        if ens_acc is None:
            print("  insuficientes modelos cargados")
            continue
        for name, acc in individual.items():
            print(f"    {acc*100:6.2f}%  {name}")
        best_indiv = max(individual.values())
        delta = (ens_acc - best_indiv) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  >> ENSEMBLE: {ens_acc*100:6.2f}%   (vs mejor individual: {sign}{delta:.2f} pp)")
        results.append((label, ens_acc, best_indiv))

    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"{'combo':<55s} {'ensemble':>10s} {'mejor indiv':>12s}  {'delta':>8s}")
    print("-" * 90)
    for label, ens_acc, best_indiv in sorted(results, key=lambda x: -x[1]):
        delta = (ens_acc - best_indiv) * 100
        sign = "+" if delta >= 0 else ""
        print(f"{label:<55s} {ens_acc*100:9.2f}% {best_indiv*100:11.2f}%  {sign}{delta:6.2f}pp")


if __name__ == "__main__":
    main()
