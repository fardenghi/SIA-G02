"""Fase 5b — Comparación lineal vs sigmoid vs relu y selección del modelo final.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej1_fraud.compare_models_relu
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

    try:
        linear_t = FraudTrainer.load(_OUT_DIR, "linear")
        sigmoid_t = FraudTrainer.load(_OUT_DIR, "sigmoid")
        relu_t = FraudTrainer.load(_OUT_DIR, "relu")
    except FileNotFoundError as e:
        print(f"Error cargando modelos: {e}. Asegúrese de ejecutar train_linear, train_sigmoid y train_relu.")
        return

    lin_mse,  lin_mae  = evaluate(linear_t.perceptron, X_norm, y)
    sig_mse,  sig_mae  = evaluate(sigmoid_t.perceptron, X_norm, y)
    relu_mse, relu_mae = evaluate(relu_t.perceptron, X_norm, y)

    # ── Tabla comparativa ────────────────────────────────────────────────────
    print("=" * 55)
    print("COMPARACIÓN LINEAL vs SIGMOID vs RELU")
    print("=" * 55)
    print(f"{'Modelo':<22} {'MSE':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 55)
    print_metrics("Lineal",  lin_mse,  lin_mae)
    print_metrics("Sigmoid", sig_mse,  sig_mae)
    print_metrics("ReLU",    relu_mse, relu_mae)
    print("-" * 55)
    
    # ── Análisis de curvas ───────────────────────────────────────────────────
    print("\nANÁLISIS DE CURVAS DE APRENDIZAJE")
    print("-" * 55)
    lin_h  = linear_t.train_loss_history
    sig_h  = sigmoid_t.train_loss_history
    relu_h = relu_t.train_loss_history
    
    print(f"  Lineal   — inicial: {lin_h[0]:.4f} → final: {lin_h[-1]:.4f}  (converge epoch ~{_convergence_epoch(lin_h)})")
    print(f"  Sigmoid  — inicial: {sig_h[0]:.4f} → final: {sig_h[-1]:.4f}  (converge epoch ~{_convergence_epoch(sig_h)})")
    print(f"  ReLU     — inicial: {relu_h[0]:.4f} → final: {relu_h[-1]:.4f}  (converge epoch ~{_convergence_epoch(relu_h)})")

    # ── Plot final de comparación ─────────────────────────────────────────────
    plot_model_comparison(
        [lin_h, sig_h, relu_h],
        ["Lineal", "Sigmoid", "ReLU"],
        title="Lineal vs Sigmoid vs ReLU: convergencia",
        save_path=_PLOTS_DIR / "phase5b_comparison_relu.png",
    )
    print(f"\nGráfico comparativo guardado en {_PLOTS_DIR}/phase5b_comparison_relu.png")


def _convergence_epoch(history, tol=1e-7):
    for i in range(len(history) - 1, 0, -1):
        if abs(history[i] - history[i - 1]) > tol:
            return i + 1
    return 1


if __name__ == "__main__":
    run()
