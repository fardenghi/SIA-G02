"""Evaluation metrics and plots for the fraud-detection perceptron."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def evaluate(perceptron, X, y):
    """Return (mse, mae) for a dataset."""
    preds = np.array([perceptron.predict(x) for x in X])
    errors = y - preds
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    return mse, mae


def plot_loss_curves(train_loss, val_loss=None, title="Loss vs Epochs",
                     save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, label="Train MSE", color="steelblue")
    if val_loss:
        ax.plot(val_loss, label="Val MSE", color="tomato", linestyle="--")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(fig, save_path)


def plot_predictions_distribution(predictions, targets, title="Distribución de predicciones",
                                  save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(targets,      bins=40, alpha=0.6, label="Target (BigModel)", color="steelblue")
    ax.hist(predictions,  bins=40, alpha=0.6, label="Predicciones (TinyModel)", color="tomato")
    ax.set_xlabel("Probabilidad de fraude")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(fig, save_path)


def print_metrics(label, mse, mae):
    print(f"  {label:<20} MSE={mse:.6f}  MAE={mae:.6f}  RMSE={mse**0.5:.6f}")


def _save_or_show(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  guardado: {save_path}")
    else:
        plt.show()
    plt.close(fig)
