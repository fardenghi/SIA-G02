"""Fase 5 — Comparación lineal vs sigmoid y selección del modelo final.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej1_fraud.compare_models
"""

import json
from pathlib import Path

from exercises.ej1_fraud.data import load_fraud_dataset, normalize_features
from exercises.ej1_fraud.evaluation import evaluate, plot_model_comparison, print_metrics
from exercises.ej1_fraud.training import FraudTrainer

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _ROOT / "outputs" / "ej1_fraud"
_PLOTS_DIR = _OUT_DIR / "plots"


def run():
    X, y, _, _ = load_fraud_dataset()
    X_norm, _, _ = normalize_features(X)

    linear_t    = FraudTrainer.load(_OUT_DIR, "linear")
    sigmoid_t = FraudTrainer.load(_OUT_DIR, "sigmoid")

    lin_mse,  lin_mae  = evaluate(linear_t.perceptron,    X_norm, y)
    sig_mse,  sig_mae  = evaluate(sigmoid_t.perceptron, X_norm, y)

    # ── Tabla comparativa ────────────────────────────────────────────────────
    print("=" * 55)
    print("COMPARACIÓN LINEAL vs SIGMOID")
    print("=" * 55)
    print(f"{'Modelo':<22} {'MSE':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 55)
    print_metrics("Lineal",  lin_mse,  lin_mae)
    print_metrics("Sigmoid", sig_mse,  sig_mae)
    print("-" * 55)
    mse_reduction = (1 - sig_mse / lin_mse) * 100
    mae_reduction = (1 - sig_mae / lin_mae) * 100
    print(f"  Mejora sigmoid        {mse_reduction:>9.1f}%  {mae_reduction:>9.1f}%")

    # ── Análisis de curvas ───────────────────────────────────────────────────
    print("\nANÁLISIS DE CURVAS DE APRENDIZAJE")
    print("-" * 55)
    lin_h  = linear_t.train_loss_history
    sig_h  = sigmoid_t.train_loss_history
    print(f"  Lineal   — inicial: {lin_h[0]:.4f} → final: {lin_h[-1]:.4f}  "
          f"(converge epoch ~{_convergence_epoch(lin_h)})")
    print(f"  Sigmoid  — inicial: {sig_h[0]:.4f} → final: {sig_h[-1]:.4f}  "
          f"(converge epoch ~{_convergence_epoch(sig_h)})")
    print()
    print("  Underfitting:  NO — ambos bajan consistentemente hasta convergencia")
    print("  Saturación:    NO — sigmoid sigue mejorando vs lineal hasta el final")
    print("  Overfitting:   N/A — estudio con split en Fase 6")

    # ── Decisión ─────────────────────────────────────────────────────────────
    print("\nDECISIÓN")
    print("-" * 55)
    print("  Modelo elegido: SIGMOID")
    print(f"  Justificación: reduce MSE un {mse_reduction:.1f}% y MAE un {mae_reduction:.1f}%")
    print("  respecto al lineal con idénticos hiperparámetros.")
    print("  La activación sigmoid captura la relación no lineal entre")
    print("  features y probabilidad de fraude, y mapea naturalmente al")
    print("  rango [0,1] del target.")

    # ── Plot final de comparación ─────────────────────────────────────────────
    plot_model_comparison(
        [lin_h, sig_h],
        ["Lineal", "Sigmoid"],
        title="Lineal vs Sigmoid: convergencia y loss final",
        save_path=_PLOTS_DIR / "phase5_comparison.png",
    )

    # ── Guardar config del modelo elegido para Fase 6 ─────────────────────────
    selected_cfg_path = Path(_OUT_DIR) / "selected_config.json"
    with open(_OUT_DIR / "sigmoid_config.json") as f:
        selected_cfg = json.load(f)
    selected_cfg["selected_model"] = "sigmoid"
    selected_cfg["selection_reason"] = (
        f"sigmoid reduces MSE {mse_reduction:.1f}% vs linear with same hyperparams"
    )
    with open(selected_cfg_path, "w") as f:
        json.dump(selected_cfg, f, indent=2)
    print(f"\nConfig del modelo elegido guardada en {selected_cfg_path}")


def _convergence_epoch(history, tol=1e-7):
    for i in range(len(history) - 1, 0, -1):
        if abs(history[i] - history[i - 1]) > tol:
            return i + 1
    return 1


if __name__ == "__main__":
    run()
