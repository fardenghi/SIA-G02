"""Busqueda del mejor ensemble heterogeneo entre modelos ya entrenados.

Estrategia:
  1) Pre-computa probs softmax de cada modelo sobre digits_test.csv (1 vez).
  2) Forward selection greedy: arranca con el mejor single y va agregando
     el modelo que mas sube el accuracy del ensemble.
  3) Busqueda exhaustiva sobre combinaciones de tamano K=2..6 de los
     top-N modelos individuales.
  4) Reporta los mejores combos encontrados.

Uso:
    uv run python -m exercises.ej3_more_digits.ensemble_search
"""
from itertools import combinations
from pathlib import Path

import numpy as np

from common.datasets import load_digits_test
from common.mlp import MLP


_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"

# Modelos a considerar — solo los que tienen alguna regularizacion
# (excluyo los vanilla porque rinden < 97% y no aportan al mejor ensemble)
EXCLUDE_PREFIXES = ("vanilla_",)
EXCLUDE_NAMES = {"baseline_only_es", "baseline_only_l2", "baseline_pure",
                 "baseline_no_aug", "softmax", "wd_5e-3", "best_decay"}


def load_all_probs(X_test):
    probs_by_name = {}
    for path in sorted(_MODELS_DIR.glob("*.npz")):
        name = path.stem
        if any(name.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if name in EXCLUDE_NAMES:
            continue
        try:
            mlp = MLP.load(str(path))
            probs_by_name[name] = mlp.forward(X_test)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return probs_by_name


def acc_of(probs_avg, y_test):
    pred = np.argmax(probs_avg, axis=1)
    return float(np.mean(pred == y_test))


def ensemble_acc(names, probs_by_name, y_test):
    avg = np.mean([probs_by_name[n] for n in names], axis=0)
    return acc_of(avg, y_test)


def greedy_forward(probs_by_name, y_test, max_size=10):
    """Empieza vacio y va agregando el modelo que mas sube el accuracy."""
    selected = []
    available = list(probs_by_name.keys())
    history = []

    while len(selected) < max_size and available:
        best_acc = -1
        best_pick = None
        for cand in available:
            acc = ensemble_acc(selected + [cand], probs_by_name, y_test)
            if acc > best_acc:
                best_acc = acc
                best_pick = cand
        selected.append(best_pick)
        available.remove(best_pick)
        history.append((tuple(selected), best_acc))
    return history


def exhaustive(top_names, probs_by_name, y_test, k_range=(2, 3, 4, 5, 6)):
    """Prueba todas las combinaciones de tamano K entre top_names."""
    results = []
    for k in k_range:
        for combo in combinations(top_names, k):
            acc = ensemble_acc(combo, probs_by_name, y_test)
            results.append((acc, combo))
    return results


def main():
    X_test, y_test = load_digits_test()
    print("Cargando modelos y precomputando probs sobre test...")
    probs_by_name = load_all_probs(X_test)
    print(f"  {len(probs_by_name)} modelos cargados\n")

    # Singles
    singles = sorted(
        [(acc_of(probs_by_name[n], y_test), n) for n in probs_by_name],
        reverse=True,
    )
    print("=" * 70)
    print(f"TOP 15 MODELOS INDIVIDUALES")
    print("=" * 70)
    for acc, n in singles[:15]:
        print(f"  {acc*100:6.2f}%  {n}")
    best_single = singles[0]
    print(f"\nMejor single: {best_single[1]}  →  {best_single[0]*100:.2f}%")

    # Greedy forward
    print("\n" + "=" * 70)
    print("GREEDY FORWARD SELECTION")
    print("=" * 70)
    history = greedy_forward(probs_by_name, y_test, max_size=12)
    for combo, acc in history:
        print(f"  k={len(combo):>2}  acc={acc*100:6.2f}%  +{combo[-1]}")
    best_greedy = max(history, key=lambda x: x[1])
    print(f"\nMejor greedy: k={len(best_greedy[0])}  acc={best_greedy[1]*100:.2f}%")
    print(f"  modelos: {list(best_greedy[0])}")

    # Exhaustive entre top-12
    top12 = [n for _, n in singles[:12]]
    print("\n" + "=" * 70)
    print(f"EXHAUSTIVO sobre top-12 (K=2..6)")
    print("=" * 70)
    results = exhaustive(top12, probs_by_name, y_test, k_range=(2, 3, 4, 5, 6))
    results.sort(reverse=True)
    print(f"\nTOP 15 COMBOS de top-12:")
    for acc, combo in results[:15]:
        print(f"  {acc*100:6.2f}%  k={len(combo)}  {list(combo)}")

    # Exhaustive K=3,4 sobre top-20 (si hay)
    top20 = [n for _, n in singles[: min(20, len(singles))]]
    print("\n" + "=" * 70)
    print(f"EXHAUSTIVO sobre top-{len(top20)} (K=3,4)")
    print("=" * 70)
    results20 = exhaustive(top20, probs_by_name, y_test, k_range=(3, 4))
    results20.sort(reverse=True)
    print(f"\nTOP 10 COMBOS de top-{len(top20)} con K=3,4:")
    for acc, combo in results20[:10]:
        print(f"  {acc*100:6.2f}%  k={len(combo)}  {list(combo)}")

    # Best overall
    all_results = results + results20
    best_overall = max(all_results, key=lambda x: x[0])
    print("\n" + "=" * 70)
    print(f"MEJOR ENSEMBLE ENCONTRADO: {best_overall[0]*100:.2f}%")
    print("=" * 70)
    print(f"  K={len(best_overall[1])}")
    for n in best_overall[1]:
        print(f"    - {n}")


if __name__ == "__main__":
    main()
