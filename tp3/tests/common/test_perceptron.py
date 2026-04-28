import numpy as np
import pytest
from common.activations import linear, sigmoid, sigmoid_prime, step
from common.simple_perceptron import SimplePerceptron


def test_predict_returns_scalar():
    p = SimplePerceptron(input_size=2)
    result = p.predict(np.array([1.0, 0.5]))
    assert np.isscalar(result) or result.ndim == 0

def test_predict_step_output_values():
    p = SimplePerceptron(input_size=2)
    result = p.predict(np.array([1.0, 0.5]))
    assert result in (1, -1)

def test_train_reduces_loss_and_linear():
    """Con activación lineal y datos linealmente separables la pérdida debe bajar."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 2], dtype=float)
    p = SimplePerceptron(input_size=2, learning_rate=0.1, max_epochs=100,
                         activation=linear, activation_prime=lambda h: 1)
    p.train(X, y)
    assert len(p.loss_history) == 100
    assert p.loss_history[-1] < p.loss_history[0]

def test_train_and_learn_and_gate():
    """Perceptron con step debe aprender AND en pocas épocas."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, -1, -1, 1], dtype=float)
    np.random.seed(0)
    p = SimplePerceptron(input_size=2, learning_rate=0.5, max_epochs=50)
    p.train(X, y)
    predictions = [p.predict(x) for x in X]
    assert predictions == [-1, -1, -1, 1]

def test_loss_history_length():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, -1.0])
    p = SimplePerceptron(input_size=2, max_epochs=10)
    p.train(X, y)
    assert len(p.loss_history) == 10

def test_weights_update_after_training():
    np.random.seed(42)
    p = SimplePerceptron(input_size=2, learning_rate=0.1, max_epochs=5)
    w_before = p.w.copy()
    b_before = p.b
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, -1.0])
    p.train(X, y)
    assert not np.allclose(p.w, w_before) or p.b != b_before
