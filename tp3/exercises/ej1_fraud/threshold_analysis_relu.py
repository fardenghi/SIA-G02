"""Análisis de umbral de detección de fraude para ReLU.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej1_fraud.threshold_analysis_relu
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.metrics import threshold_sweep

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _ROOT / "outputs" / "ej1_fraud"
_PLOTS_DIR = _OUT_DIR / "plots"


def run():
    oof_path = _OUT_DIR / "oof_predictions_relu.npz"
    if not oof_path.exists():
        print(f"Error: No se encontró {oof_path}. Ejecute train_generalization_relu primero.")
        return

    data = np.load(oof_path)
    preds = data["preds"]
    labels = data["labels"]

    print("ANÁLISIS DE PREDICCIONES ReLU — OUT-OF-FOLD (Dataset completo)")
    print("-" * 45)
    print(f"  Muestras totales:     {len(preds)}")
    print(f"  Fraudes reales:       {labels.sum()} ({labels.mean() * 100:.1f}%)")
    print(f"  Pred media:           {preds.mean():.4f}")
    print(f"  Pred std:             {preds.std():.4f}")
    print(
        "  Pred percentiles:     "
        f"p25={np.percentile(preds, 25):.3f}  "
        f"p50={np.percentile(preds, 50):.3f}  "
        f"p75={np.percentile(preds, 75):.3f}"
    )
    print(f"  Pred max:             {preds.max():.4f}")

    # Como ReLU no está acotado en (0,1), determinamos los thresholds de forma dinámica
    # basados en los percentiles o un rango desde 0 hasta el max predict.
    # Dado que y está en {-1, 1} (o {0, 1} internamente, asumiendo {-1, 1} en linear pero en sigmoid el mapeo era [0,1]).
    # De hecho, y es probablemente {-1, 1} en este ej1? No, `labels` son 0 y 1.
    max_pred = float(preds.max())
    # Generar 20 thresholds entre el 5% del max y el 95% del max
    if max_pred <= 0:
        max_pred = 1.0 # fallback
        
    thresholds = np.linspace(max_pred * 0.05, max_pred * 0.95, 20)
    prec, rec, f1 = threshold_sweep(preds, labels, thresholds)

    print("\nPRECISION / RECALL / F1 POR UMBRAL (ReLU)")
    print(f"{'Umbral':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 38)
    for t, p, r, f in zip(thresholds, prec, rec, f1):
        print(f"  {t:.2f}   {p:>9.4f}  {r:>7.4f}  {f:>7.4f}")

    best_idx = int(np.argmax(f1))
    best_t = thresholds[best_idx]
    best_p, best_r, best_f = prec[best_idx], rec[best_idx], f1[best_idx]

    print(f"\nUMBRAL ÓPTIMO ReLU (max F1): {best_t:.2f}")
    print(f"  Precision: {best_p:.4f}")
    print(f"  Recall:    {best_r:.4f}")
    print(f"  F1-score:  {best_f:.4f}")

    _plot_pr_vs_threshold(thresholds, prec, rec, f1, best_t)
    _plot_pr_curve(prec, rec, best_p, best_r, best_f)

    print("\nRECOMENDACIÓN FINAL (ReLU)")
    print("=" * 45)
    print(f"  Umbral sugerido:  {best_t:.2f}")
    print(f"  Precision:        {best_p:.4f}  — de cada 100 alertas, ~{best_p * 100:.0f} son fraude real")
    print(f"  Recall:           {best_r:.4f}  — se detecta el ~{best_r * 100:.0f}% de los fraudes")
    print(f"  F1-score:         {best_f:.4f}")
    print()
    print("  Nota para ReLU: Los umbrales no están en el rango [0,1] como en Sigmoide, ")
    print("  sino que dependen de la amplitud de las predicciones arrojadas por la activación.")

    results = {
        "optimal_threshold": float(best_t),
        "precision": float(best_p),
        "recall": float(best_r),
        "f1": float(best_f),
    }
    out_path = _OUT_DIR / "threshold_results_relu.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {out_path}")


def _plot_pr_vs_threshold(thresholds, prec, rec, f1, best_t):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, prec, label="Precision", color="steelblue", marker="o", ms=4)
    ax.plot(thresholds, rec, label="Recall", color="tomato", marker="s", ms=4)
    ax.plot(thresholds, f1, label="F1-score", color="seagreen", marker="^", ms=4, lw=2)
    ax.axvline(best_t, color="gray", linestyle="--", alpha=0.7, label=f"Óptimo ({best_t:.2f})")
    ax.set_xlabel("Umbral de clasificación")
    ax.set_ylabel("Métrica")
    ax.set_title("Precision, Recall y F1 vs Umbral (ReLU)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, _PLOTS_DIR / "threshold_metrics_relu.png")


def _plot_pr_curve(prec, rec, best_p, best_r, best_f):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="steelblue", lw=2, label="Curva P-R")
    ax.scatter([best_r], [best_p], color="tomato", zorder=5, s=80, label=f"Óptimo F1={best_f:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall (ReLU)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, _PLOTS_DIR / "precision_recall_curve_relu.png")


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
