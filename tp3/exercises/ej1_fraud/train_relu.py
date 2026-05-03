"""Fase 4b — Perceptrón con activación ReLU sobre el dataset completo y análisis de Learning Rate.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej1_fraud.train_relu
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.activations import relu, relu_prime
from common.simple_perceptron import SimplePerceptron
from exercises.ej1_fraud.config import DEFAULT_CONFIG, save_config
from exercises.ej1_fraud.data import load_fraud_dataset, normalize_features
from exercises.ej1_fraud.evaluation import evaluate, print_metrics
from exercises.ej1_fraud.training import FraudTrainer

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _ROOT / "outputs" / "ej1_fraud"
_PLOTS_DIR = _OUT_DIR / "plots"
_MODEL_NAME = "relu"

CFG = {**DEFAULT_CONFIG, "activation": "relu", "max_epochs": 100}

def plot_lr_comparison(histories, lrs, best_lr, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for hist, lr in zip(histories, lrs):
        lw = 2.5 if lr == best_lr else 1.5
        alpha = 1.0 if lr == best_lr else 0.7
        ax.plot(hist, label=f"LR = {lr}", linewidth=lw, alpha=alpha)
    
    ax.set_title("ReLU: Loss vs Epochs para distintos Learning Rates")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def run():
    # ── 1. Datos ─────────────────────────────────────────────────────────────
    X, y, _, _ = load_fraud_dataset()
    X_norm, mean, std = normalize_features(X)
    print(f"Dataset: {X_norm.shape[0]} muestras, {X_norm.shape[1]} features")

    # ── 2. Análisis de Learning Rates para ReLU ──────────────────────────────
    learning_rates = [0.01, 0.001, 0.0001]
    histories = []
    trainers = []
    final_losses = []

    print("\nEvaluando Learning Rates para ReLU:")
    for lr in learning_rates:
        np.random.seed(CFG["seed"])
        perceptron = SimplePerceptron(
            input_size=X_norm.shape[1],
            learning_rate=lr,
            max_epochs=CFG["max_epochs"],
            activation=relu,
            activation_prime=relu_prime,
        )
        trainer = FraudTrainer(perceptron)
        trainer.train(X_norm, y)
        
        histories.append(trainer.train_loss_history)
        trainers.append(trainer)
        final_loss = trainer.train_loss_history[-1]
        final_losses.append(final_loss)
        
        print(f"  LR = {lr:<6} -> Loss final: {final_loss:.6f}")

    # ── 3. Seleccionar el mejor LR ───────────────────────────────────────────
    best_idx = np.argmin(final_losses)
    best_lr = learning_rates[best_idx]
    best_trainer = trainers[best_idx]
    best_perceptron = best_trainer.perceptron
    
    print(f"\nMejor Learning Rate: {best_lr} (Loss: {final_losses[best_idx]:.6f})")

    # ── 4. Plot de comparación de LRs ────────────────────────────────────────
    plot_path = _PLOTS_DIR / "relu_lr_comparison.png"
    plot_lr_comparison(histories, learning_rates, best_lr, plot_path)
    print(f"Gráfico comparativo de LRs guardado en {plot_path}")

    # ── 5. Resultados del mejor modelo ReLU ──────────────────────────────────
    mse, mae = evaluate(best_perceptron, X_norm, y)
    print("\nMétricas finales del mejor ReLU (dataset completo):")
    print_metrics(f"No lineal (ReLU, lr={best_lr})", mse, mae)

    # ── 6. Guardar ───────────────────────────────────────────────────────────
    best_trainer.save(_OUT_DIR, _MODEL_NAME)
    
    final_cfg = {
        **CFG, 
        "learning_rate": best_lr,
        "input_size": X_norm.shape[1],
        "norm_mean": mean.tolist(), 
        "norm_std": std.tolist()
    }
    save_config(final_cfg, _OUT_DIR / f"{_MODEL_NAME}_config.json")
    print(f"\nModelo ReLU guardado en {_OUT_DIR}/{_MODEL_NAME}.*")


if __name__ == "__main__":
    run()
