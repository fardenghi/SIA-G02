"""Entrenamiento del VAE: bucle full-batch con Adam, β-warmup y tracker de métricas.

Reutiliza el optimizador `Adam` del Ej1. Cada época se muestrea un `ε` fresco (el muestreo
estocástico del posterior es justamente lo que distingue al VAE), se evalúa el ELBO
`recon + β·KL` y se da un paso de Adam sobre el vector plano de pesos del `VAE`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

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


class EarlyStopping:
    """Early stopping por paciencia sobre una métrica de validación (callback de `train_vae`).

    Es **independiente del tamaño del dataset**: el criterio no es un número fijo de épocas
    sino relativo —se detiene cuando la val deja de mejorar (en más de `min_delta`) durante
    `patience` evaluaciones seguidas—. Con datos chicos la val sube antes → corta antes; con
    muchos datos mejora más tiempo → corta después. Guarda los pesos del mejor val y los
    restaura con `restore()`, así los plots/finales reflejan el modelo óptimo y no el
    sobreajustado. Registra `epochs`/`train`/`val` para la curva.

    `start_epoch` evita seleccionar/cortar **antes** de que termine el warmup de β: si no, como
    la val (recon) mejora con β bajo, el "óptimo" caería a mitad del warmup → se restauraría un
    modelo sub-regularizado (β incompleto), malo para la generación. Las evaluaciones previas a
    `start_epoch` se registran para la curva pero no cuentan para best/paciencia.

    Uso:
        es = EarlyStopping(val_fn=lambda m: recon_px(m, Xval),
                           train_fn=lambda m: recon_px(m, Xtr), patience=10,
                           start_epoch=beta_warmup)
        train_vae(vae, Xtr, ..., callback=es, callback_every=25)
        es.restore(vae)        # deja en `vae` los pesos del mínimo de val
    """

    def __init__(self, val_fn, train_fn=None, patience: int = 10, min_delta: float = 1e-4,
                 start_epoch: int = 0):
        self.val_fn = val_fn
        self.train_fn = train_fn
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.best = float("inf")
        self.best_epoch = -1
        self.best_params = None
        self.wait = 0
        self.stopped_epoch = -1
        self.epochs: list[int] = []
        self.train: list[float] = []
        self.val: list[float] = []

    def __call__(self, epoch: int, vae: VAE, metrics: dict) -> bool:
        v = float(self.val_fn(vae))
        self.epochs.append(epoch)
        self.val.append(v)
        if self.train_fn is not None:
            self.train.append(float(self.train_fn(vae)))
        if epoch < self.start_epoch:        # durante el warmup: solo registrar la curva
            return False
        if v < self.best - self.min_delta:
            self.best = v
            self.best_epoch = epoch
            self.best_params = vae.get_params().copy()
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def restore(self, vae: VAE) -> None:
        """Carga en `vae` los pesos del mejor val (no-op si nunca mejoró)."""
        if self.best_params is not None:
            vae.set_params(self.best_params)


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
    batch_size: int | None = None,
    lr_schedule: str = "constant",
    callback: Callable[[int, object, dict], None] | None = None,
    callback_every: int | None = None,
) -> dict:
    """Entrena `vae` in-place sobre `X` (Adam). Devuelve métricas finales.

    `beta`/`beta_warmup` controlan el peso de la KL (ver `beta_schedule`). El muestreo de
    `ε` es estocástico por época con `rng` (determinista dada `seed`). Con `batch_size=None`
    es full-batch (un paso por época); con `batch_size < N` hace mini-batch SGD (barajando
    cada época), lo que desacopla el costo de `N` y hace tratables datasets grandes.

    `lr_schedule="cosine"` decae el LR de `lr` a ~0 (annealing, útil en corridas largas).
    `callback(epoch, vae, metrics)` se llama cada `callback_every` épocas (default `log_every`)
    con el `vae` ya actualizado: sirve para checkpointing y figuras de monitoreo.
    """
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule debe ser 'constant' o 'cosine', no {lr_schedule!r}")
    rng = rng or np.random.default_rng(seed)
    opt = Adam(lr=lr)
    params = vae.get_params()
    n = X.shape[0]
    bs = n if batch_size is None else min(batch_size, n)
    cb_every = callback_every or log_every

    for epoch in range(epochs):
        if lr_schedule == "cosine":
            opt.lr = lr * 0.5 * (1.0 + np.cos(np.pi * epoch / max(1, epochs - 1)))
        b = beta_schedule(epoch, beta, beta_warmup)
        # Full-batch (bs==n): orden fijo y sin permutación -> mismo stream de rng que antes.
        order = np.arange(n) if bs == n else rng.permutation(n)
        total = recon = kl = 0.0
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            vae.set_params(params)
            eps = rng.standard_normal((idx.size, vae.latent_dim))
            vae.forward(X[idx], eps=eps)
            total, recon, kl = vae.elbo(b)
            vae.backward(b)
            params = opt.step(params, vae.get_grads())
        if tracker is not None and (epoch % log_every == 0):
            tracker.log(epoch, total, recon, kl, b)
        if callback is not None and (epoch % cb_every == 0):
            vae.set_params(params)
            # Si el callback devuelve truthy (p. ej. EarlyStopping) se corta el entrenamiento.
            if callback(epoch, vae, {"elbo": total, "recon": recon, "kl": kl, "lr": opt.lr}):
                break
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
