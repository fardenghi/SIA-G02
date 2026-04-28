import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self, layer, dW, db):
        """Update layer.W and layer.b in-place."""


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def step(self, layer, dW, db):
        layer.W -= self.lr * dW
        layer.b -= self.lr * db


class Momentum(Optimizer):
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self._state = {}

    def step(self, layer, dW, db):
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


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._state = {}

    def step(self, layer, dW, db):
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
