"""Rankea todos los modelos guardados en outputs/ej3_more_digits/models por val accuracy
sobre el split de validación interno (15% de more_digits.csv, seed=42)."""
from pathlib import Path

import numpy as np

from common.datasets import load_more_digits, to_one_hot
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"


def main():
    X_all, y_all = load_more_digits()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.15)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_val = X_all[val_idx]
    Y_val = to_one_hot(y_all[val_idx], 10, encoding="zero_one")
    X_train = X_all[train_idx]
    Y_train = to_one_hot(y_all[train_idx], 10, encoding="zero_one")

    rows = []
    for path in sorted(_MODELS_DIR.glob("*.npz")):
        try:
            mlp = MLP.load(str(path))
            va = mlp.evaluate(X_val, Y_val)
            tr = mlp.evaluate(X_train, Y_train)
            rows.append((path.stem, va["accuracy"], va["loss"],
                         tr["accuracy"], tr["accuracy"] - va["accuracy"]))
        except Exception as e:
            print(f"[skip] {path.name}: {e}")

    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"\n{'Modelo':<38s} {'val_acc':>8s} {'val_loss':>9s} {'train_acc':>10s} {'Δ':>9s}")
    print("-" * 80)
    for name, va, vl, ta, gap in rows:
        print(f"{name:<38s} {va:>8.4f} {vl:>9.4f} {ta:>10.4f} {gap:>+9.4f}")

    best = rows[0]
    print("\n" + "=" * 80)
    print(f"GANADOR (val accuracy): {best[0]}")
    print(f"  val_acc   = {best[1]:.4f}")
    print(f"  val_loss  = {best[2]:.4f}")
    print(f"  train_acc = {best[3]:.4f}")
    print(f"  gap (Δ)   = {best[4]:+.4f}")


if __name__ == "__main__":
    main()
