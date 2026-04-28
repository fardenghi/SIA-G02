"""Fase 3 — Perceptrón con activación lineal sobre el dataset completo.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej1_fraud.train_linear
"""

from pathlib import Path

import numpy as np

from common.activations import linear
from common.simple_perceptron import SimplePerceptron
from exercises.ej1_fraud.config import DEFAULT_CONFIG, save_config
from exercises.ej1_fraud.data import load_fraud_dataset, normalize_features
from exercises.ej1_fraud.evaluation import evaluate, plot_loss_curves, print_metrics
from exercises.ej1_fraud.training import FraudTrainer

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _ROOT / "outputs" / "ej1_fraud"
_PLOTS_DIR = _OUT_DIR / "plots"
_MODEL_NAME = "linear"

CFG = {**DEFAULT_CONFIG, "activation": "linear", "max_epochs": 100}


def run():
    # ── 1. Datos ─────────────────────────────────────────────────────────────
    X, y, _, feature_names = load_fraud_dataset()
    X_norm, mean, std = normalize_features(X)
    print(f"Dataset: {X_norm.shape[0]} muestras, {X_norm.shape[1]} features")

    # ── 2. Modelo ────────────────────────────────────────────────────────────
    np.random.seed(CFG["seed"])
    perceptron = SimplePerceptron(
        input_size=X_norm.shape[1],
        learning_rate=CFG["learning_rate"],
        max_epochs=CFG["max_epochs"],
        activation=linear,
        activation_prime=None,
    )

    # ── 3. Entrenamiento ─────────────────────────────────────────────────────
    trainer = FraudTrainer(perceptron)
    trainer.train(X_norm, y)

    # ── 4. Resultados ────────────────────────────────────────────────────────
    mse, mae = evaluate(perceptron, X_norm, y)
    print("\nMétricas finales (dataset completo):")
    print_metrics("Lineal", mse, mae)
    print(f"Loss inicial:  {trainer.train_loss_history[0]:.6f}")
    print(f"Loss final:    {trainer.train_loss_history[-1]:.6f}")
    print(f"Reducción:     {(1 - trainer.train_loss_history[-1] / trainer.train_loss_history[0]) * 100:.1f}%")

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    plot_loss_curves(
        trainer.train_loss_history,
        title="Perceptrón Lineal — Loss vs Epochs",
        save_path=_PLOTS_DIR / "linear_loss.png",
    )

    # ── 6. Guardar ───────────────────────────────────────────────────────────
    trainer.save(_OUT_DIR, _MODEL_NAME)
    save_config({**CFG, "input_size": X_norm.shape[1],
                 "norm_mean": mean.tolist(), "norm_std": std.tolist()},
                _OUT_DIR / f"{_MODEL_NAME}_config.json")
    print(f"\nModelo guardado en {_OUT_DIR}/{_MODEL_NAME}.*")


if __name__ == "__main__":
    run()
