"""Pérdidas con gradiente analítico propio y validación de coherencia.

Convención: cada pérdida expone `value(y_pred, y_true)` y `grad(y_pred, y_true)`,
donde `grad` devuelve `dL/d(y_pred)` (gradiente respecto a la salida ACTIVADA de la
red). La red luego multiplica por la derivada de la activación de salida en su backward.
Todas las pérdidas se normalizan por el número total de elementos (`y_pred.size`).
"""

from __future__ import annotations

import warnings

import numpy as np

EPS = 1e-7


def bce_value(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary cross-entropy con clamp a `[ε, 1−ε]` para evitar `log(0)`."""
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def bce_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """dL/d(y_pred) de la BCE, con el mismo clamp que el forward."""
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    return (p - y_true) / (p * (1.0 - p)) / y_pred.size


def mse_value(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2.0 * (y_pred - y_true) / y_pred.size


LOSSES = {
    "bce": (bce_value, bce_grad),
    "mse": (mse_value, mse_grad),
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Pérdida desconocida: {name!r}")
    return LOSSES[name]


def validate_coherence(loss: str, output_activation: str, activation: str,
                       init: str, latent_activation: str | None = None) -> list[str]:
    """Valida combinaciones loss↔activación↔init y emite warnings (no errores).

    Devuelve la lista de mensajes emitidos (útil para tests).
    """
    messages: list[str] = []

    if latent_activation in ("relu", "sigmoid"):
        messages.append(
            f"latent_activation='{latent_activation}' acota/descentra el cuello "
            "(relu lo colapsa a >=0, sigmoid a [0,1]); para un latente sin restringir "
            "se sugiere 'linear' (o 'tanh' acotado y centrado)."
        )

    if loss == "bce" and output_activation != "sigmoid":
        messages.append(
            f"BCE asume salida 'sigmoid' pero output_activation='{output_activation}'. "
            "Puede producir gradientes inconsistentes."
        )
    if loss == "mse" and output_activation == "linear":
        messages.append(
            "MSE con salida 'linear' no acota la salida a [0,1]; la métrica de píxeles "
            "usa umbral 0.5 igualmente."
        )
    if activation == "relu" and init.startswith("xavier"):
        messages.append(
            "Activación 'relu' con init 'xavier'; se sugiere una init 'he' para mantener "
            "la varianza."
        )
    if activation in ("tanh", "sigmoid") and init.startswith("he"):
        messages.append(
            f"Activación '{activation}' con init 'he'; se sugiere 'xavier' para "
            "saturar menos."
        )

    for msg in messages:
        warnings.warn(msg, stacklevel=2)
    return messages
