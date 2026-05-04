"""Versión 'val' de ensemble_search.py: busca el mejor ensemble heterogéneo
optimizando validation accuracy (split interno 15% de more_digits.csv, seed=42)
en lugar de digits_test.csv. digits_test.csv queda reservado para evaluación
final del ganador.

Uso:
    uv run python -m exercises.ej3_more_digits.ensemble_search_val
"""
from itertools import combinations
from pathlib import Path

import numpy as np

from common.datasets import load_more_digits, to_one_hot
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej3_more_digits" / "models"

EXCLUDE_PREFIXES = ("vanilla_",)
EXCLUDE_NAMES = {"baseline_only_es", "baseline_only_l2", "baseline_pure",
                 "baseline_no_aug", "wd_5e-3", "best_decay"}


def load_all_probs(X_val):
    probs_by_name = {}
    for path in sorted(_MODELS_DIR.glob("*.npz")):
        name = path.stem
        if any(name.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if name in EXCLUDE_NAMES:
            continue
        try:
            mlp = MLP.load(str(path))
            probs_by_name[name] = mlp.forward(X_val)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return probs_by_name


def acc_of(probs_avg, y_true):
    pred = np.argmax(probs_avg, axis=1)
    return float(np.mean(pred == y_true))


def ensemble_acc(names, probs_by_name, y_true):
    avg = np.mean([probs_by_name[n] for n in names], axis=0)
    return acc_of(avg, y_true)


def greedy_forward(probs_by_name, y_true, max_size=10):
    selected = []
    available = list(probs_by_name.keys())
    history = []

    while len(selected) < max_size and available:
        best_acc = -1
        best_pick = None
        for cand in available:
            acc = ensemble_acc(selected + [cand], probs_by_name, y_true)
            if acc > best_acc:
                best_acc = acc
                best_pick = cand
        selected.append(best_pick)
        available.remove(best_pick)
        history.append((tuple(selected), best_acc))
    return history


def exhaustive(top_names, probs_by_name, y_true, k_range=(2, 3, 4, 5, 6)):
    results = []
    for k in k_range:
        for combo in combinations(top_names, k):
            acc = ensemble_acc(combo, probs_by_name, y_true)
            results.append((acc, combo))
    return results


def main():
    X_all, y_all = load_more_digits()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = int(len(X_all) * 0.15)
    val_idx = idx[:n_val]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]

    print("Cargando modelos y precomputando probs sobre VAL...")
    probs_by_name = load_all_probs(X_val)
    print(f"  {len(probs_by_name)} modelos cargados\n")

    singles = sorted(
        [(acc_of(probs_by_name[n], y_val), n) for n in probs_by_name],
        reverse=True,
    )
    print("=" * 70)
    print("TOP 15 MODELOS INDIVIDUALES (val)")
    print("=" * 70)
    for acc, n in singles[:15]:
        print(f"  {acc*100:6.2f}%  {n}")
    best_single = singles[0]
    print(f"\nMejor single: {best_single[1]}  →  {best_single[0]*100:.2f}%")

    print("\n" + "=" * 70)
    print("GREEDY FORWARD SELECTION (val)")
    print("=" * 70)
    history = greedy_forward(probs_by_name, y_val, max_size=12)
    for combo, acc in history:
        print(f"  k={len(combo):>2}  val_acc={acc*100:6.2f}%  +{combo[-1]}")
    best_greedy = max(history, key=lambda x: x[1])
    print(f"\nMejor greedy: k={len(best_greedy[0])}  val_acc={best_greedy[1]*100:.2f}%")
    print(f"  modelos: {list(best_greedy[0])}")

    top12 = [n for _, n in singles[:12]]
    print("\n" + "=" * 70)
    print("EXHAUSTIVO sobre top-12 (K=2..6, val)")
    print("=" * 70)
    results = exhaustive(top12, probs_by_name, y_val, k_range=(2, 3, 4, 5, 6))
    results.sort(reverse=True)
    print("\nTOP 15 COMBOS de top-12:")
    for acc, combo in results[:15]:
        print(f"  {acc*100:6.2f}%  k={len(combo)}  {list(combo)}")

    top20 = [n for _, n in singles[: min(20, len(singles))]]
    print("\n" + "=" * 70)
    print(f"EXHAUSTIVO sobre top-{len(top20)} (K=3,4, val)")
    print("=" * 70)
    results20 = exhaustive(top20, probs_by_name, y_val, k_range=(3, 4))
    results20.sort(reverse=True)
    print(f"\nTOP 10 COMBOS de top-{len(top20)} con K=3,4:")
    for acc, combo in results20[:10]:
        print(f"  {acc*100:6.2f}%  k={len(combo)}  {list(combo)}")

    all_results = results + results20
    best_overall = max(all_results, key=lambda x: x[0])
    print("\n" + "=" * 70)
    print(f"MEJOR ENSEMBLE POR VAL ACCURACY: {best_overall[0]*100:.2f}%")
    print("=" * 70)
    print(f"  K={len(best_overall[1])}")
    for n in best_overall[1]:
        print(f"    - {n}")


if __name__ == "__main__":
    main()
