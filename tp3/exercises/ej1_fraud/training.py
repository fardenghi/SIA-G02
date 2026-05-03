"""Training utilities for the fraud-detection perceptron."""

import json
from pathlib import Path

import numpy as np

from common.activations import linear, sigmoid, sigmoid_prime, step, tanh_act, tanh_prime, relu, relu_prime
from common.simple_perceptron import SimplePerceptron

_ACTIVATION_MAP = {
    "linear":  (linear,   None),
    "sigmoid": (sigmoid,  sigmoid_prime),
    "tanh":    (tanh_act, tanh_prime),
    "step":    (step,     None),
    "relu":    (relu,     relu_prime),
}


class FraudTrainer:
    """Wraps SimplePerceptron and adds validation monitoring + persistence."""

    def __init__(self, perceptron):
        self.perceptron = perceptron
        self.train_loss_history = []
        self.val_loss_history = []

    def train(self, X, y):
        """Full training without validation (delegates to SimplePerceptron.train)."""
        self.perceptron.train(X, y)
        self.train_loss_history = list(self.perceptron.loss_history)

    def train_with_validation(self, X_tr, y_tr, X_val, y_val):
        """Per-epoch SGD with validation loss tracked each epoch."""
        p = self.perceptron
        for _ in range(p.max_epochs):
            errors = []
            for x_i, y_i in zip(X_tr, y_tr):
                h = float(np.dot(p.w, x_i) + p.b)
                y_pred = p.activation(h)
                error = y_i - y_pred
                d = p.activation_prime(h) if p.activation_prime else 1.0
                p.w += p.lr * error * d * x_i
                p.b += p.lr * error * d
                errors.append(error ** 2)

            train_mse = float(np.mean(errors))
            self.train_loss_history.append(train_mse)
            p.loss_history.append(train_mse)

            val_preds = np.array([p.predict(x) for x in X_val])
            val_mse = float(np.mean((y_val - val_preds) ** 2))
            self.val_loss_history.append(val_mse)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, out_dir, name):
        """Save weights (.npz) and config (.json) to out_dir/name.*"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = self.perceptron

        np.savez(
            out / f"{name}.npz",
            w=p.w,
            b=np.array([p.b]),
            train_loss=np.array(self.train_loss_history),
            val_loss=np.array(self.val_loss_history),
        )

        activation_name = next(
            k for k, (fn, _) in _ACTIVATION_MAP.items() if fn is p.activation
        )
        cfg = {
            "input_size": len(p.w),
            "learning_rate": p.lr,
            "max_epochs": p.max_epochs,
            "activation": activation_name,
        }
        with open(out / f"{name}.json", "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, out_dir, name):
        """Reconstruct a FraudTrainer from saved files."""
        out = Path(out_dir)
        with open(out / f"{name}.json") as f:
            cfg = json.load(f)

        activation, activation_prime = _ACTIVATION_MAP[cfg["activation"]]
        p = SimplePerceptron(
            input_size=cfg["input_size"],
            learning_rate=cfg["learning_rate"],
            max_epochs=cfg["max_epochs"],
            activation=activation,
            activation_prime=activation_prime,
        )

        data = np.load(out / f"{name}.npz")
        p.w = data["w"]
        p.b = float(data["b"][0])

        trainer = cls(p)
        trainer.train_loss_history = data["train_loss"].tolist()
        trainer.val_loss_history = data["val_loss"].tolist()
        return trainer
