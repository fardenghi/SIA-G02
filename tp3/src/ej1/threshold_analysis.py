"""Fase 7 — Análisis de umbral de detección de fraude.

Ejecutar desde la raíz del proyecto:
    uv run python -m src.ej1.threshold_analysis
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.ej1.data import load_fraud_dataset

_OUT_DIR  = "experiments/ej1"
_PLOTS_DIR = f"{_OUT_DIR}/plots"

with open(f"{_OUT_DIR}/generalization_config.json") as f:
    _CFG = json.load(f)


def _threshold_metrics(preds, labels, thresholds):
    """Return arrays of precision, recall, F1 for each threshold."""
    precision, recall, f1 = [], [], []
    for t in thresholds:
        pred_bin = (preds >= t).astype(int)
        tp = int(((pred_bin == 1) & (labels == 1)).sum())
        fp = int(((pred_bin == 1) & (labels == 0)).sum())
        fn = int(((pred_bin == 0) & (labels == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return np.array(precision), np.array(recall), np.array(f1)


def run():
    # ── 1. Cargar Predicciones OOF ───────────────────────────────────────────
    oof_path = Path(_OUT_DIR) / "oof_predictions.npz"
    if not oof_path.exists():
        print(f"Error: No se encontró {oof_path}. Ejecute la Fase 6 primero.")
        return
        
    data = np.load(oof_path)
    preds = data["preds"]
    l_v = data["labels"]

    print("ANÁLISIS DE PREDICCIONES — OUT-OF-FOLD (Dataset completo)")
    print("-" * 45)
    print(f"  Muestras totales:     {len(preds)}")
    print(f"  Fraudes reales:       {l_v.sum()} ({l_v.mean()*100:.1f}%)")
    print(f"  Pred media:           {preds.mean():.4f}")
    print(f"  Pred std:             {preds.std():.4f}")
    print(f"  Pred percentiles:     "
          f"p25={np.percentile(preds,25):.3f}  "
          f"p50={np.percentile(preds,50):.3f}  "
          f"p75={np.percentile(preds,75):.3f}")

    # ── 3. Métricas por umbral ────────────────────────────────────────────────
    thresholds = np.arange(0.05, 0.96, 0.05)
    prec, rec, f1 = _threshold_metrics(preds, l_v, thresholds)

    print("\nPRECISION / RECALL / F1 POR UMBRAL")
    print(f"{'Umbral':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 38)
    for t, p, r, f in zip(thresholds, prec, rec, f1):
        print(f"  {t:.2f}   {p:>9.4f}  {r:>7.4f}  {f:>7.4f}")

    # ── 4. Umbral óptimo (max F1) ─────────────────────────────────────────────
    best_idx = int(np.argmax(f1))
    best_t   = thresholds[best_idx]
    best_p, best_r, best_f = prec[best_idx], rec[best_idx], f1[best_idx]

    print(f"\nUMBRAL ÓPTIMO (max F1): {best_t:.2f}")
    print(f"  Precision: {best_p:.4f}")
    print(f"  Recall:    {best_r:.4f}")
    print(f"  F1-score:  {best_f:.4f}")

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    _plot_pr_vs_threshold(thresholds, prec, rec, f1, best_t)
    _plot_pr_curve(prec, rec, best_p, best_r, best_f)

    # ── 6. Recomendación final ────────────────────────────────────────────────
    print("\nRECOMENDACIÓN FINAL")
    print("=" * 45)
    print(f"  Umbral sugerido:  {best_t:.2f}")
    print(f"  Precision:        {best_p:.4f}  — de cada 100 alertas, "
          f"~{best_p*100:.0f} son fraude real")
    print(f"  Recall:           {best_r:.4f}  — se detecta el "
          f"~{best_r*100:.0f}% de los fraudes")
    print(f"  F1-score:         {best_f:.4f}")
    print()
    print("  Justificación: umbral que maximiza F1, balanceando el costo")
    print("  de falsos positivos (alertas innecesarias) y falsos negativos")
    print("  (fraudes no detectados). El cliente puede ajustar hacia un")
    print("  umbral menor para priorizar recall (detectar más fraudes)")
    print("  o mayor para priorizar precisión (menos falsas alarmas).")

    # ── 7. Guardar resultados ─────────────────────────────────────────────────
    results = {
        "optimal_threshold": float(best_t),
        "precision": float(best_p),
        "recall":    float(best_r),
        "f1":        float(best_f),
    }
    out_path = Path(_OUT_DIR) / "threshold_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {out_path}")


def _plot_pr_vs_threshold(thresholds, prec, rec, f1, best_t):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, prec, label="Precision", color="steelblue",  marker="o", ms=4)
    ax.plot(thresholds, rec,  label="Recall",    color="tomato",     marker="s", ms=4)
    ax.plot(thresholds, f1,   label="F1-score",  color="seagreen",   marker="^", ms=4, lw=2)
    ax.axvline(best_t, color="gray", linestyle="--", alpha=0.7,
               label=f"Óptimo ({best_t:.2f})")
    ax.set_xlabel("Umbral de clasificación")
    ax.set_ylabel("Métrica")
    ax.set_title("Fase 7 — Precision, Recall y F1 vs Umbral")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, f"{_PLOTS_DIR}/threshold_metrics.png")


def _plot_pr_curve(prec, rec, best_p, best_r, best_f):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="steelblue", lw=2, label="Curva P-R")
    ax.scatter([best_r], [best_p], color="tomato", zorder=5, s=80,
               label=f"Óptimo F1={best_f:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Fase 7 — Curva Precision-Recall")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, f"{_PLOTS_DIR}/precision_recall_curve.png")


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
