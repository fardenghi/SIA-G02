"""Evalúa un ensemble (lista de modelos) sobre train, val y test.

Train/val: split interno de more_digits.csv (15% val, seed=42).
Test: digits_test.csv (producción).

Uso:
    uv run python -m exercises.ej3_more_digits.eval_ensemble \\
        aug_shifts aug_rot5 aug_rot10 aug_rot15_scale
"""
import sys
from pathlib import Path

import numpy as np

from common.datasets import load_digits_test, load_more_digits, to_one_hot
from common.ensemble import Ensemble

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m exercises.ej3_more_digits.eval_ensemble <modelo1> <modelo2> ...")
        sys.exit(1)

    names = sys.argv[1:]
    paths = []
    for n in names:
        p = _MODELS_DIR / (n if n.endswith(".npz") else f"{n}.npz")
        if not p.exists():
            print(f"Error: no existe {p}")
            sys.exit(1)
        paths.append(str(p))

    # Split interno (mismo que se usó para entrenar)
    X_all, y_all = load_more_digits()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.15)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    Y_train = to_one_hot(y_train, 10, encoding="zero_one")
    Y_val = to_one_hot(y_val, 10, encoding="zero_one")

    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    ens = Ensemble.from_paths(paths)
    tr = ens.evaluate(X_train, Y_train)
    va = ens.evaluate(X_val, Y_val)
    te = ens.evaluate(X_test, Y_test)

    print("\nEnsemble:")
    for n in names:
        print(f"  - {n}")

    print("\n{:<10s} {:>10s} {:>10s}".format("split", "accuracy", "loss"))
    print("-" * 34)
    print(f"{'train':<10s} {tr['accuracy']:>10.4f} {tr['loss']:>10.4f}")
    print(f"{'val':<10s} {va['accuracy']:>10.4f} {va['loss']:>10.4f}")
    print(f"{'test':<10s} {te['accuracy']:>10.4f} {te['loss']:>10.4f}")

    print(f"\nΔ train-val  = {tr['accuracy'] - va['accuracy']:+.4f}")
    print(f"Δ train-test = {tr['accuracy'] - te['accuracy']:+.4f}")
    print(f"Δ val-test   = {va['accuracy'] - te['accuracy']:+.4f}")


if __name__ == "__main__":
    main()
