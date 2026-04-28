import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from common.activations import linear, sigmoid, sigmoid_prime
from common.simple_perceptron import SimplePerceptron
from exercises.ej1_fraud.config import DEFAULT_CONFIG, load_config, save_config
from exercises.ej1_fraud.evaluation import evaluate
from exercises.ej1_fraud.training import FraudTrainer


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_perceptron(activation=linear, activation_prime=None, epochs=5):
    return SimplePerceptron(
        input_size=4,
        learning_rate=0.01,
        max_epochs=epochs,
        activation=activation,
        activation_prime=activation_prime,
    )


def _toy_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = rng.uniform(0, 1, n)
    return X, y


# ── config ────────────────────────────────────────────────────────────────────

def test_default_config_keys():
    for key in ("learning_rate", "max_epochs", "activation", "val_ratio", "seed"):
        assert key in DEFAULT_CONFIG


def test_save_load_config_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        save_config({"learning_rate": 0.05, "max_epochs": 50}, path)
        cfg = load_config(path)
    assert cfg["learning_rate"] == 0.05
    assert cfg["max_epochs"] == 50
    assert "activation" in cfg           # defaults filled in


# ── FraudTrainer.train ────────────────────────────────────────────────────────

def test_train_fills_loss_history():
    X, y = _toy_data()
    p = _make_perceptron(epochs=10)
    trainer = FraudTrainer(p)
    trainer.train(X, y)
    assert len(trainer.train_loss_history) == 10
    assert len(trainer.val_loss_history) == 0


def test_train_loss_decreases_or_stays():
    X, y = _toy_data(n=200)
    p = _make_perceptron(activation=sigmoid, activation_prime=sigmoid_prime, epochs=50)
    trainer = FraudTrainer(p)
    trainer.train(X, y)
    assert trainer.train_loss_history[-1] <= trainer.train_loss_history[0] + 0.1


# ── FraudTrainer.train_with_validation ───────────────────────────────────────

def test_train_with_validation_histories():
    X, y = _toy_data(n=100)
    X_tr, y_tr, X_v, y_v = X[:80], y[:80], X[80:], y[80:]
    p = _make_perceptron(activation=sigmoid, activation_prime=sigmoid_prime, epochs=10)
    trainer = FraudTrainer(p)
    trainer.train_with_validation(X_tr, y_tr, X_v, y_v)
    assert len(trainer.train_loss_history) == 10
    assert len(trainer.val_loss_history) == 10


def test_train_with_validation_positive_losses():
    X, y = _toy_data(n=100)
    p = _make_perceptron(activation=sigmoid, activation_prime=sigmoid_prime, epochs=5)
    trainer = FraudTrainer(p)
    trainer.train_with_validation(X[:80], y[:80], X[80:], y[80:])
    assert all(v >= 0 for v in trainer.train_loss_history)
    assert all(v >= 0 for v in trainer.val_loss_history)


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    X, y = _toy_data(n=60)
    p = _make_perceptron(activation=sigmoid, activation_prime=sigmoid_prime, epochs=5)
    trainer = FraudTrainer(p)
    trainer.train(X, y)

    with tempfile.TemporaryDirectory() as tmp:
        trainer.save(tmp, "test_model")
        assert (Path(tmp) / "test_model.npz").exists()
        assert (Path(tmp) / "test_model.json").exists()

        loaded = FraudTrainer.load(tmp, "test_model")

    np.testing.assert_array_equal(loaded.perceptron.w, p.w)
    assert loaded.perceptron.b == pytest.approx(p.b)
    assert loaded.train_loss_history == trainer.train_loss_history


# ── evaluate ──────────────────────────────────────────────────────────────────

def test_evaluate_returns_positive():
    X, y = _toy_data(n=30)
    p = _make_perceptron(activation=sigmoid, activation_prime=sigmoid_prime, epochs=3)
    FraudTrainer(p).train(X, y)
    mse, mae = evaluate(p, X, y)
    assert mse >= 0
    assert mae >= 0


def test_evaluate_perfect_predictions():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0.5, 0.5])
    p = SimplePerceptron(input_size=2, activation=linear)
    p.w = np.array([0.0, 0.0])
    p.b = 0.5
    mse, mae = evaluate(p, X, y)
    assert mse == pytest.approx(0.0)
    assert mae == pytest.approx(0.0)
