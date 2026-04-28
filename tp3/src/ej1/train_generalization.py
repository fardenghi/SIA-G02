"""Fase 6 — Estudio de generalización: sigmoid con split 80/20.

Ejecutar desde la raíz del proyecto:
    uv run python -m src.ej1.train_generalization
"""

import json

import numpy as np

from src.activation import sigmoid, sigmoid_prime
from src.ej1.config import save_config
from src.ej1.data import load_fraud_dataset, normalize_features, train_val_split
from src.ej1.evaluation import (
    evaluate,
    plot_loss_curves,
    plot_predictions_distribution,
    print_metrics,
)
from src.ej1.training import FraudTrainer
from src.perceptron import SimplePerceptron

_OUT_DIR = "experiments/ej1"
_PLOTS_DIR = f"{_OUT_DIR}/plots"
_MODEL_NAME = "generalization"

with open(f"{_OUT_DIR}/selected_config.json") as f:
    _SELECTED = json.load(f)

CFG = {
    "learning_rate": _SELECTED["learning_rate"],
    "max_epochs":    _SELECTED["max_epochs"],
    "activation":    _SELECTED["activation"],
    "val_ratio":     _SELECTED["val_ratio"],
    "seed":          _SELECTED["seed"],
}


def run():
    # ── 1. Datos y split ─────────────────────────────────────────────────────
    X, y, labels, _ = load_fraud_dataset()
    X_tr, y_tr, l_tr, X_v, y_v, l_v = train_val_split(
        X, y, labels, val_ratio=CFG["val_ratio"], seed=CFG["seed"]
    )

    # Normalizar con stats del train únicamente
    X_tr_n, mean, std = normalize_features(X_tr)
    X_v_n, _, _       = normalize_features(X_v, mean=mean, std=std)

    print(f"Train: {X_tr_n.shape[0]} muestras  |  Val: {X_v_n.shape[0]} muestras")

    # ── 2. Modelo ────────────────────────────────────────────────────────────
    np.random.seed(CFG["seed"])
    perceptron = SimplePerceptron(
        input_size=X_tr_n.shape[1],
        learning_rate=CFG["learning_rate"],
        max_epochs=CFG["max_epochs"],
        activation=sigmoid,
        activation_prime=sigmoid_prime,
    )

    # ── 3. Entrenamiento con monitoreo de validación ─────────────────────────
    trainer = FraudTrainer(perceptron)
    trainer.train_with_validation(X_tr_n, y_tr, X_v_n, y_v)

    # ── 4. Métricas finales ──────────────────────────────────────────────────
    tr_mse, tr_mae = evaluate(perceptron, X_tr_n, y_tr)
    v_mse,  v_mae  = evaluate(perceptron, X_v_n,  y_v)

    print("\nMétricas finales:")
    print_metrics("Train", tr_mse, tr_mae)
    print_metrics("Val",   v_mse,  v_mae)
    gap = v_mse - tr_mse
    print(f"\n  Gap val-train (MSE): {gap:+.6f}  ({gap/tr_mse*100:+.1f}%)")

    # ── 5. Análisis de generalización ────────────────────────────────────────
    print("\nANÁLISIS DE GENERALIZACIÓN")
    print("-" * 45)
    if gap / tr_mse < 0.10:
        verdict = "Buen balance — sin overfitting apreciable"
    elif gap / tr_mse < 0.25:
        verdict = "Leve overfitting — generalización aceptable"
    else:
        verdict = "Overfitting significativo"
    print(f"  {verdict}")
    last_tr = trainer.train_loss_history[-1]
    last_v  = trainer.val_loss_history[-1]
    if last_tr > 0.05:
        print("  Posible underfitting — loss de train elevado")
    else:
        print("  Sin underfitting — loss de train bajo")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    plot_loss_curves(
        trainer.train_loss_history,
        trainer.val_loss_history,
        title="Fase 6 — Generalización: Train vs Val MSE",
        save_path=f"{_PLOTS_DIR}/generalization_loss.png",
    )

    preds_v = np.array([perceptron.predict(x) for x in X_v_n])
    plot_predictions_distribution(
        preds_v, y_v,
        title="Fase 6 — Distribución de predicciones (val set)",
        save_path=f"{_PLOTS_DIR}/generalization_pred_dist.png",
    )

    # ── 7. Guardar ───────────────────────────────────────────────────────────
    trainer.save(_OUT_DIR, _MODEL_NAME)
    save_config({
        **CFG,
        "input_size": X_tr_n.shape[1],
        "norm_mean": mean.tolist(),
        "norm_std":  std.tolist(),
        "final_train_mse": tr_mse,
        "final_val_mse":   v_mse,
        "final_train_mae": tr_mae,
        "final_val_mae":   v_mae,
    }, f"{_OUT_DIR}/{_MODEL_NAME}_config.json")
    print(f"\nModelo guardado en {_OUT_DIR}/{_MODEL_NAME}.*")


if __name__ == "__main__":
    run()
