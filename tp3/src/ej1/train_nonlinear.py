"""Fase 4 — Perceptrón con activación sigmoid sobre el dataset completo.

Ejecutar desde la raíz del proyecto:
    uv run python -m src.ej1.train_nonlinear
"""

import numpy as np

from src.activation import sigmoid, sigmoid_prime
from src.ej1.config import DEFAULT_CONFIG, save_config
from src.ej1.data import load_fraud_dataset, normalize_features
from src.ej1.evaluation import (
    evaluate,
    plot_model_comparison,
    print_metrics,
)
from src.ej1.training import FraudTrainer
from src.perceptron import SimplePerceptron

_OUT_DIR = "experiments/ej1"
_PLOTS_DIR = f"{_OUT_DIR}/plots"
_MODEL_NAME = "nonlinear"

CFG = {**DEFAULT_CONFIG, "activation": "sigmoid", "max_epochs": 100}


def run():
    # ── 1. Datos (mismos que Fase 3) ─────────────────────────────────────────
    X, y, _, _ = load_fraud_dataset()
    X_norm, mean, std = normalize_features(X)
    print(f"Dataset: {X_norm.shape[0]} muestras, {X_norm.shape[1]} features")

    # ── 2. Modelo ────────────────────────────────────────────────────────────
    np.random.seed(CFG["seed"])
    perceptron = SimplePerceptron(
        input_size=X_norm.shape[1],
        learning_rate=CFG["learning_rate"],
        max_epochs=CFG["max_epochs"],
        activation=sigmoid,
        activation_prime=sigmoid_prime,
    )

    # ── 3. Entrenamiento ─────────────────────────────────────────────────────
    trainer = FraudTrainer(perceptron)
    trainer.train(X_norm, y)

    # ── 4. Resultados ────────────────────────────────────────────────────────
    mse, mae = evaluate(perceptron, X_norm, y)
    print("\nMétricas finales (dataset completo):")
    print_metrics("No lineal (sigmoid)", mse, mae)
    print(f"Loss inicial:  {trainer.train_loss_history[0]:.6f}")
    print(f"Loss final:    {trainer.train_loss_history[-1]:.6f}")
    print(f"Reducción:     {(1 - trainer.train_loss_history[-1] / trainer.train_loss_history[0]) * 100:.1f}%")

    # ── 5. Comparación con modelo lineal ─────────────────────────────────────
    linear_trainer = FraudTrainer.load(_OUT_DIR, "linear")
    print("\nComparación:")
    print_metrics("  Lineal", *evaluate(linear_trainer.perceptron, X_norm, y))
    print_metrics("  Sigmoid", mse, mae)

    plot_model_comparison(
        [linear_trainer.train_loss_history, trainer.train_loss_history],
        ["Lineal", "Sigmoid"],
        title="Lineal vs Sigmoid — Loss vs Epochs",
        save_path=f"{_PLOTS_DIR}/linear_vs_nonlinear_loss.png",
    )

    # ── 6. Guardar ───────────────────────────────────────────────────────────
    trainer.save(_OUT_DIR, _MODEL_NAME)
    save_config({**CFG, "input_size": X_norm.shape[1],
                 "norm_mean": mean.tolist(), "norm_std": std.tolist()},
                f"{_OUT_DIR}/{_MODEL_NAME}_config.json")
    print(f"\nModelo guardado en {_OUT_DIR}/{_MODEL_NAME}.*")


if __name__ == "__main__":
    run()
