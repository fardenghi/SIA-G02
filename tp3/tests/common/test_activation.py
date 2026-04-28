import numpy as np
import pytest
from common.activations import linear, sigmoid, sigmoid_prime, step, tanh_act, tanh_prime


def test_step_positive():
    assert step(1.0) == 1

def test_step_zero():
    assert step(0.0) == 1

def test_step_negative():
    assert step(-1.0) == -1

def test_linear_identity():
    assert linear(3.5) == 3.5
    assert linear(-2.0) == -2.0
    assert linear(0.0) == 0.0

def test_sigmoid_range():
    for h in [-10, -1, 0, 1, 10]:
        s = sigmoid(h)
        assert 0 < s < 1

def test_sigmoid_midpoint():
    assert sigmoid(0) == pytest.approx(0.5)

def test_sigmoid_prime_positive():
    assert sigmoid_prime(0) > 0

def test_tanh_act_range():
    for h in [-5, -1, 0, 1, 5]:
        assert -1 <= tanh_act(h) <= 1

def test_tanh_act_zero():
    assert tanh_act(0) == pytest.approx(0.0)

def test_tanh_prime_positive():
    assert tanh_prime(0) > 0
