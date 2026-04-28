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
    
    train_loss = np.array(train_loss)
    if train_loss.ndim == 2:
        epochs = np.arange(train_loss.shape[1])
        train_mean = train_loss.mean(axis=0)
        train_std = train_loss.std(axis=0)
        ax.plot(epochs, train_mean, label="Train MSE (media)", color="steelblue")
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="steelblue", alpha=0.2)
        
        if val_loss is not None:
            val_loss = np.array(val_loss)
            val_mean = val_loss.mean(axis=0)
            val_std = val_loss.std(axis=0)
            ax.plot(epochs, val_mean, label="Val MSE (media)", color="tomato", linestyle="--")
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="tomato", alpha=0.2)
    else:
        ax.plot(train_loss, label="Train MSE", color="steelblue")
        if val_loss is not None:
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


def plot_model_comparison(histories, labels, title="Comparación de modelos — Loss vs Epochs",
                          save_path=None):
    """Overlay loss curves from multiple models on a single plot.

    histories: list of loss-history lists
    labels:    list of display names (same length)
    """
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for hist, label, color in zip(histories, labels, colors):
        ax.plot(hist, label=label, color=color)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
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
