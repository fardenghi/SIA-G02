"""Fase 6 — Estudio de generalización: sigmoid con split 80/20.

Ejecutar desde la raíz del proyecto:
    uv run python -m src.ej1.train_generalization
"""

import json

import numpy as np

from src.activation import sigmoid, sigmoid_prime
from src.ej1.config import save_config
from src.ej1.data import load_fraud_dataset, normalize_features, k_fold_split
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
    "k_folds":       5,
    "seed":          _SELECTED["seed"],
}


def run():
    # ── 1. Datos ─────────────────────────────────────────────────────────────
    X, y, labels, _ = load_fraud_dataset()
    
    print(f"Iniciando K-Fold Cross-Validation (K={CFG['k_folds']})")
    print("-" * 45)
    print(f"Total muestras: {X.shape[0]}")
    
    all_train_loss = []
    all_val_loss = []
    oof_preds = np.zeros(len(X))
    tr_mses, tr_maes, v_mses, v_maes = [], [], [], []
    
    # ── 2. Entrenamiento K-Fold ──────────────────────────────────────────────
    for fold, (X_tr, y_tr, l_tr, X_v, y_v, l_v, val_idx) in enumerate(k_fold_split(
        X, y, labels, k=CFG["k_folds"], seed=CFG["seed"]
    )):
        X_tr_n, mean, std = normalize_features(X_tr)
        X_v_n, _, _       = normalize_features(X_v, mean=mean, std=std)

        np.random.seed(CFG["seed"] + fold)
        perceptron = SimplePerceptron(
            input_size=X_tr_n.shape[1],
            learning_rate=CFG["learning_rate"],
            max_epochs=CFG["max_epochs"],
            activation=sigmoid,
            activation_prime=sigmoid_prime,
        )

        trainer = FraudTrainer(perceptron)
        trainer.train_with_validation(X_tr_n, y_tr, X_v_n, y_v)
        
        all_train_loss.append(trainer.train_loss_history)
        all_val_loss.append(trainer.val_loss_history)
        
        tr_mse, tr_mae = evaluate(perceptron, X_tr_n, y_tr)
        v_mse,  v_mae  = evaluate(perceptron, X_v_n,  y_v)
        
        tr_mses.append(tr_mse)
        tr_maes.append(tr_mae)
        v_mses.append(v_mse)
        v_maes.append(v_mae)
        
        preds_v = np.array([perceptron.predict(x) for x in X_v_n])
        oof_preds[val_idx] = preds_v
        
        print(f"  Fold {fold+1}/{CFG['k_folds']} | Train MSE: {tr_mse:.4f} | Val MSE: {v_mse:.4f}")
        
    # ── 3. Métricas finales (Promedio K-Fold) ────────────────────────────────
    print("\nMétricas finales (Promedio K-Fold):")
    tr_mse_avg = np.mean(tr_mses)
    v_mse_avg = np.mean(v_mses)
    tr_mae_avg = np.mean(tr_maes)
    v_mae_avg = np.mean(v_maes)
    
    print_metrics("Train (media)", tr_mse_avg, tr_mae_avg)
    print_metrics("Val (media)",   v_mse_avg,  v_mae_avg)
    gap = v_mse_avg - tr_mse_avg
    print(f"\n  Gap val-train (MSE): {gap:+.6f}  ({gap/tr_mse_avg*100:+.1f}%)")

    # ── 4. Análisis de generalización ────────────────────────────────────────
    print("\nANÁLISIS DE GENERALIZACIÓN")
    print("-" * 45)
    if gap / tr_mse_avg < 0.10:
        verdict = "Buen balance — sin overfitting apreciable"
    elif gap / tr_mse_avg < 0.25:
        verdict = "Leve overfitting — generalización aceptable"
    else:
        verdict = "Overfitting significativo"
    print(f"  {verdict}")
    if tr_mse_avg > 0.05:
        print("  Posible underfitting — loss de train promedio elevado")
    else:
        print("  Sin underfitting — loss de train promedio bajo")

    # ── 5. Modelo Final (100% de datos) ──────────────────────────────────────
    print("\nEntrenando modelo final con 100% de los datos...")
    X_n, final_mean, final_std = normalize_features(X)
    np.random.seed(CFG["seed"])
    final_perceptron = SimplePerceptron(
        input_size=X_n.shape[1],
        learning_rate=CFG["learning_rate"],
        max_epochs=CFG["max_epochs"],
        activation=sigmoid,
        activation_prime=sigmoid_prime,
    )
    final_trainer = FraudTrainer(final_perceptron)
    final_trainer.train(X_n, y)

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    plot_loss_curves(
        all_train_loss,
        all_val_loss,
        title=f"Fase 6 — Generalización (K-Fold K={CFG['k_folds']})",
        save_path=f"{_PLOTS_DIR}/generalization_loss.png",
    )

    plot_predictions_distribution(
        oof_preds, y,
        title="Fase 6 — Distribución predicciones (Out-Of-Fold)",
        save_path=f"{_PLOTS_DIR}/generalization_pred_dist.png",
    )

    # ── 7. Guardar ───────────────────────────────────────────────────────────
    final_trainer.save(_OUT_DIR, _MODEL_NAME)
    np.savez(f"{_OUT_DIR}/oof_predictions.npz", preds=oof_preds, labels=labels)
    
    save_config({
        **CFG,
        "input_size": X_n.shape[1],
        "norm_mean": final_mean.tolist(),
        "norm_std":  final_std.tolist(),
        "final_train_mse_avg": float(tr_mse_avg),
        "final_val_mse_avg":   float(v_mse_avg),
        "final_train_mae_avg": float(tr_mae_avg),
        "final_val_mae_avg":   float(v_mae_avg),
    }, f"{_OUT_DIR}/{_MODEL_NAME}_config.json")
    print(f"\nModelo final guardado en {_OUT_DIR}/{_MODEL_NAME}.*")
    print(f"Predicciones OOF guardadas en {_OUT_DIR}/oof_predictions.npz")


if __name__ == "__main__":
    run()
