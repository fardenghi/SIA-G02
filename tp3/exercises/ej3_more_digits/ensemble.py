"""Ensemble por promedio de probabilidades sobre el test set de Ej3.

Carga varios modelos guardados, promedia sus salidas softmax y reporta
test accuracy + accuracy por clase. Comparación contra cada modelo individual.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej3_more_digits.ensemble
    uv run python -m exercises.ej3_more_digits.ensemble best best_l2
"""

import sys
from pathlib import Path

import numpy as np

from common.datasets import load_digits_test, to_one_hot
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"


def _evaluate(probs, y_true, n_classes):
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y_true))
    per_class = {}
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class[c] = float(np.mean(pred[mask] == c))
    return acc, per_class


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ["best", "best_decay", "best_l2"]

    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")
    n_classes = 10

    models, individual = [], {}
    for name in names:
        path = _MODELS_DIR / f"{name}.npz"
        if not path.exists():
            print(f"  [skip] {path} no existe")
            continue
        mlp = MLP.load(path)
        probs = mlp.forward(X_test)  # output_activation=softmax → ya son probs
        acc, per_class = _evaluate(probs, y_test, n_classes)
        m = mlp.evaluate(X_test, Y_test)
        models.append((name, probs))
        individual[name] = {"acc": acc, "loss": m["loss"], "per_class": per_class}

    if len(models) < 2:
        print("Se necesitan al menos 2 modelos para ensemblar.")
        return

    avg_probs = np.mean([p for _, p in models], axis=0)
    ens_acc, ens_per_class = _evaluate(avg_probs, y_test, n_classes)

    print("\nMODELOS INDIVIDUALES")
    print("-" * 35)
    for name, info in individual.items():
        print(f"  {name:<14s}  test_acc={info['acc']:.4f}  test_loss={info['loss']:.4f}")

    print("\nENSEMBLE (promedio de probabilidades)")
    print("-" * 35)
    print(f"  models combined: {[n for n,_ in models]}")
    print(f"  test_acc={ens_acc:.4f}")

    best_indiv = max(individual.values(), key=lambda d: d["acc"])
    delta = ens_acc - best_indiv["acc"]
    sign = "+" if delta >= 0 else ""
    print(f"  vs mejor individual: {sign}{delta:.4f}")

    print("\nACCURACY POR CLASE")
    header = f"  {'class':>5s} " + " ".join(f"{n:>10s}" for n in individual) + f" {'ensemble':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in range(n_classes):
        row = f"  {c:>5d} " + " ".join(
            f"{individual[n]['per_class'].get(c, 0.0):>10.4f}" for n in individual
        ) + f" {ens_per_class.get(c, 0.0):>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
