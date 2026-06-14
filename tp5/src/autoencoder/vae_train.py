"""Entrenamiento del VAE: bucle full-batch con Adam, β-warmup y tracker de métricas.

Reutiliza el optimizador `Adam` del Ej1. Cada época se muestrea un `ε` fresco (el muestreo
estocástico del posterior es justamente lo que distingue al VAE), se evalúa el ELBO
`recon + β·KL` y se da un paso de Adam sobre el vector plano de pesos del `VAE`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .optim import Adam
from .vae import VAE


def beta_schedule(epoch: int, beta: float, warmup: int) -> float:
    """β efectivo en la época `epoch`: warmup lineal de 0 a `beta` en `warmup` épocas.

    Con `warmup <= 0` devuelve `beta` constante. El warmup arranca dando casi todo el peso
    a la reconstrucción (β≈0) y sube la KL gradualmente, mitigando el *posterior collapse*.
    """
    if warmup <= 0:
        return beta
    return beta * min(1.0, epoch / warmup)


class VAEMetricsTracker:
    """Acumula `total`/`recon`/`kl`/`beta` por época en un DataFrame; exporta a CSV."""

    def __init__(self, run_label: str = "vae"):
        self.run_label = run_label
        self._rows: list[dict] = []

    def log(self, epoch: int, total: float, recon: float, kl: float, beta: float) -> None:
        self._rows.append({
            "run": self.run_label,
            "epoch": epoch,
            "elbo": total,
            "recon": recon,
            "kl": kl,
            "beta": beta,
        })

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)


def _eval_recon(vae: VAE, X: np.ndarray) -> float:
    """Reconstrucción determinista (z = μ, ε=0): aísla la calidad de reconstrucción."""
    vae.forward(X, eps=np.zeros((X.shape[0], vae.latent_dim)))
    return vae.elbo(0.0)[1]


def train_vae(
    vae: VAE,
    X: np.ndarray,
    epochs: int = 3000,
    lr: float = 1e-3,
    beta: float = 1.0,
    beta_warmup: int = 0,
    seed: int = 0,
    tracker: VAEMetricsTracker | None = None,
    log_every: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    """Entrena `vae` in-place sobre `X` (full-batch, Adam). Devuelve métricas finales.

    `beta`/`beta_warmup` controlan el peso de la KL (ver `beta_schedule`). El muestreo de
    `ε` es estocástico por época con `rng` (determinista dada `seed`).
    """
    rng = rng or np.random.default_rng(seed)
    opt = Adam(lr=lr)
    params = vae.get_params()

    for epoch in range(epochs):
        b = beta_schedule(epoch, beta, beta_warmup)
        vae.set_params(params)
        eps = rng.standard_normal((X.shape[0], vae.latent_dim))
        vae.forward(X, eps=eps)
        total, recon, kl = vae.elbo(b)
        vae.backward(b)
        params = opt.step(params, vae.get_grads())
        if tracker is not None and (epoch % log_every == 0):
            tracker.log(epoch, total, recon, kl, b)
    vae.set_params(params)

    # Métricas finales: ELBO con β objetivo y reconstrucción determinista.
    eps = rng.standard_normal((X.shape[0], vae.latent_dim))
    vae.forward(X, eps=eps)
    total, recon, kl = vae.elbo(beta)
    return {
        "elbo": total,
        "recon": recon,
        "kl": kl,
        "recon_det": _eval_recon(vae, X),
    }
