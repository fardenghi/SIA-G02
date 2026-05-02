"""Evaluar un modelo Ej3 ya entrenado sobre digits_test.csv.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej3_more_digits.eval outputs/ej3_more_digits/models/best_l2_aug.npz
"""
import sys
from pathlib import Path

import numpy as np

from common.datasets import load_digits_test, to_one_hot
from common.mlp import MLP


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m exercises.ej3_more_digits.eval <path/al/modelo.npz>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"Error: no existe {model_path}")
        sys.exit(1)

    mlp = MLP.load(model_path)
    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    m = mlp.evaluate(X_test, Y_test)
    print(f"\nModelo: {model_path}")
    print(f"Arquitectura: {mlp.layer_sizes}")
    print(f"Test accuracy: {m['accuracy']:.4f}")
    print(f"Test loss:     {m['loss']:.4f}")

    pred = np.argmax(mlp.forward(X_test), axis=1)
    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_test == c
        if mask.sum() > 0:
            acc = float(np.mean(pred[mask] == c))
            print(f"  Class {c}: {acc:.4f}  ({mask.sum()} samples)")


if __name__ == "__main__":
    main()
