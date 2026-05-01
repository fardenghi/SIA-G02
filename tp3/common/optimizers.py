import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self, layer, dW, db):
        """Update layer.W and layer.b in-place."""


def _apply_l2(dW, layer, weight_decay):
    """L2 / weight decay: adds λ·W to gradient. Biases are not regularized."""
    if weight_decay > 0:
        return dW + weight_decay * layer.W
    return dW


class SGD(Optimizer):
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layer, dW, db):
        dW = _apply_l2(dW, layer, self.weight_decay)
        layer.W -= self.lr * dW
        layer.b -= self.lr * db


class Momentum(Optimizer):
    def __init__(self, lr, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self._state = {}

    def step(self, layer, dW, db):
        dW = _apply_l2(dW, layer, self.weight_decay)
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
            }
        s = self._state[lid]
        s["vW"] = self.beta * s["vW"] + self.lr * dW
        s["vb"] = self.beta * s["vb"] + self.lr * db
        layer.W -= s["vW"]
        layer.b -= s["vb"]


class RMSProp(Optimizer):
    def __init__(self, lr, gamma=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.weight_decay = weight_decay
        self._state = {}

    def step(self, layer, dW, db):
        dW = _apply_l2(dW, layer, self.weight_decay)
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "SW": np.zeros_like(layer.W),
                "Sb": np.zeros_like(layer.b),
            }
        s = self._state[lid]
        s["SW"] = self.gamma * s["SW"] + (1 - self.gamma) * dW ** 2
        s["Sb"] = self.gamma * s["Sb"] + (1 - self.gamma) * db ** 2
        layer.W -= self.lr * dW / (np.sqrt(s["SW"]) + self.eps)
        layer.b -= self.lr * db / (np.sqrt(s["Sb"]) + self.eps)


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._state = {}

    def step(self, layer, dW, db):
        dW = _apply_l2(dW, layer, self.weight_decay)
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "mW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b),
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
                "t": 0,
            }
        s = self._state[lid]
        s["t"] += 1
        t = s["t"]

        s["mW"] = self.beta1 * s["mW"] + (1 - self.beta1) * dW
        s["mb"] = self.beta1 * s["mb"] + (1 - self.beta1) * db
        s["vW"] = self.beta2 * s["vW"] + (1 - self.beta2) * dW ** 2
        s["vb"] = self.beta2 * s["vb"] + (1 - self.beta2) * db ** 2

        mW_hat = s["mW"] / (1 - self.beta1 ** t)
        mb_hat = s["mb"] / (1 - self.beta1 ** t)
        vW_hat = s["vW"] / (1 - self.beta2 ** t)
        vb_hat = s["vb"] / (1 - self.beta2 ** t)

        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class StepDecay:
    """Multiplies optimizer.lr by decay_rate every step_size epochs."""

    def __init__(self, decay_rate=0.5, step_size=50, lr_min=1e-6):
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.lr_min = lr_min
        self._epoch = 0

    def step(self, optimizer, epoch_loss):
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            optimizer.lr = max(optimizer.lr * self.decay_rate, self.lr_min)
        return optimizer.lr


class ExponentialDecay:
    """Multiplies optimizer.lr by decay_rate every epoch."""

    def __init__(self, decay_rate=0.99, lr_min=1e-6):
        self.decay_rate = decay_rate
        self.lr_min = lr_min

    def step(self, optimizer, epoch_loss):
        optimizer.lr = max(optimizer.lr * self.decay_rate, self.lr_min)
        return optimizer.lr


class AdaptiveLR:
    """Adapts optimizer.lr based on loss trend, per the rule from class:

      - if loss decreased on the last `k` epochs in a row → lr += a
      - if loss increased on the last `k` epochs in a row → lr *= (1 - b)
    """

    def __init__(self, k=5, a=1e-4, b=0.1, lr_min=1e-6, lr_max=1.0):
        self.k = k
        self.a = a
        self.b = b
        self.lr_min = lr_min
        self.lr_max = lr_max
        self._losses = []

    def step(self, optimizer, epoch_loss):
        self._losses.append(epoch_loss)
        if len(self._losses) <= self.k:
            return optimizer.lr

        diffs = np.diff(self._losses[-(self.k + 1):])
        if np.all(diffs < 0):
            optimizer.lr = min(optimizer.lr + self.a, self.lr_max)
        elif np.all(diffs > 0):
            optimizer.lr = max(optimizer.lr * (1 - self.b), self.lr_min)
        return optimizer.lr
