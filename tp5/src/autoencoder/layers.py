"""Capa densa, activaciones e inicializaciones.

Una capa densa aplica `z = X @ W + b` seguido de una activación `a = f(z)`.
El backward recibe `da` (gradiente de la pérdida respecto a la salida activada) y
devuelve `dx` (gradiente respecto a la entrada), acumulando `dW`/`db`.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Activaciones: cada una devuelve (forward, derivada respecto a la pre-activación)
# La derivada se expresa en términos de la salida activada `a` cuando conviene.
# --------------------------------------------------------------------------- #


def _tanh(z):
    return np.tanh(z)


def _tanh_grad(z, a):
    return 1.0 - a * a


def _sigmoid(z):
    # Estable numéricamente: evita overflow de exp para |z| grande (p. ej. tras relu).
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_grad(z, a):
    return a * (1.0 - a)


def _relu(z):
    return np.maximum(0.0, z)


def _relu_grad(z, a):
    return (z > 0.0).astype(z.dtype)


def _linear(z):
    return z


def _linear_grad(z, a):
    return np.ones_like(z)


ACTIVATIONS = {
    "tanh": (_tanh, _tanh_grad),
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "relu": (_relu, _relu_grad),
    "linear": (_linear, _linear_grad),
}


def get_activation(name: str):
    if name not in ACTIVATIONS:
        raise ValueError(f"Activación desconocida: {name!r}")
    return ACTIVATIONS[name]


# --------------------------------------------------------------------------- #
# Inicializaciones de pesos (reproducibles por semilla vía el rng provisto).
# --------------------------------------------------------------------------- #


def init_weights(n_in: int, n_out: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Inicializa una matriz de pesos `(n_in, n_out)` según `kind`."""
    if kind == "xavier_uniform":
        limit = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-limit, limit, size=(n_in, n_out))
    if kind == "xavier_normal":
        std = np.sqrt(2.0 / (n_in + n_out))
        return rng.normal(0.0, std, size=(n_in, n_out))
    if kind == "he_uniform":
        limit = np.sqrt(6.0 / n_in)
        return rng.uniform(-limit, limit, size=(n_in, n_out))
    if kind == "he_normal":
        std = np.sqrt(2.0 / n_in)
        return rng.normal(0.0, std, size=(n_in, n_out))
    if kind == "normal":
        return rng.normal(0.0, 1.0, size=(n_in, n_out))
    if kind == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n_in, n_out))
    raise ValueError(f"Inicialización desconocida: {kind!r}")


class Dense:
    """Capa densa con activación, forward/backward y gradientes propios."""

    def __init__(self, n_in: int, n_out: int, activation: str, init: str,
                 rng: np.random.Generator):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation
        self.W = init_weights(n_in, n_out, init, rng)
        self.b = np.zeros(n_out)
        self._fwd, self._grad = get_activation(activation)
        # gradientes acumulados
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # cache para backward
        self._x = None
        self._z = None
        self._a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        self._a = self._fwd(self._z)
        return self._a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """Propaga `da` (grad respecto a la salida activada) hacia la entrada."""
        dz = da * self._grad(self._z, self._a)
        self.dW = self._x.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T

    def backward_from_dz(self, dz: np.ndarray) -> np.ndarray:
        """Backward cuando ya se tiene el gradiente respecto a la pre-activación `z`.

        Útil para la capa de salida cuando la pérdida entrega `dL/dz` directamente
        (p. ej. BCE+sigmoid se simplifica a `(ŷ - y)`).
        """
        self.dW = self._x.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size
