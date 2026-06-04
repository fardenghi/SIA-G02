"""Optimizadores: Adam propio y wrapper L-BFGS-B (scipy). Nunca Powell.

Ambos trabajan sobre el vector plano de pesos provisto por `network.get_params()`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize


class Adam:
    """Adam implementado a mano (momentos de 1er/2do orden + corrección de sesgo)."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads * grads)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def lbfgs_minimize(
    fun_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    maxiter: int = 20000,
    callback: Callable[[np.ndarray], None] | None = None,
):
    """Minimiza con L-BFGS-B usando el gradiente analítico propio vía `jac`.

    `fun_and_grad(theta)` devuelve `(loss, grad)`. Scipy solo decide el paso; toda la
    matemática del autoencoder es propia.
    """
    result = minimize(
        fun=lambda theta: fun_and_grad(theta),
        x0=x0,
        method="L-BFGS-B",
        jac=True,  # `fun` devuelve (valor, gradiente)
        callback=callback,
        options={"maxiter": maxiter, "maxfun": maxiter * 2},
    )
    return result
