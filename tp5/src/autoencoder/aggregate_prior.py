"""Prior empírico sobre el POSTERIOR AGREGADO `q(z) = ⅟ₙ Σ q(z|xᵢ)` (Ej2c, mejora de samples).

El VAE genera con `z ~ p(z) -> decode`. Con el prior estándar `p(z)=N(0,I)` eso solo funciona
si el posterior agregado se le parece; en latente alto **no** (los códigos viven en una variedad
finísima y `N(0,I)` reparte masa por un volumen enorme → la mayoría de los `z` caen en zonas no
entrenadas → manchones). El arreglo clásico (*ex-post density estimation*, Ghosh et al. 2020) es
ajustar una densidad a los `μ` del dataset **después** de entrenar y muestrear de ahí.

Es un **tratamiento posterior**: no toca el decoder ni el entrenamiento (el VAE se entrena igual
con `N(0,I)`); solo reemplaza la distribución de la que se muestrea `z` al generar. Se elige por
config (`generation.prior`), con `"gaussian"` revirtiendo al `N(0,I)` habitual.

Dos densidades, ambas numpy desde cero:
  - **gaussian**: una Gaussiana full-cov (capta media y correlaciones del posterior agregado).
  - **gmm**: mezcla de Gaussianas de covarianza diagonal por EM (capta multimodalidad, p. ej.
    caras vs animales como clusters distintos). Diagonal = robusta en alta dim con pocos puntos.
"""

from __future__ import annotations

import numpy as np


def _fit_gaussian(mu: np.ndarray, ridge: float = 1e-4):
    """Media y factor de Cholesky de la covarianza empírica de `mu` (+ridge por estabilidad)."""
    mean = mu.mean(axis=0)
    cov = np.cov(mu, rowvar=False) + ridge * np.eye(mu.shape[1])
    return mean, np.linalg.cholesky(cov)


def _fit_gmm(mu: np.ndarray, k: int, iters: int = 80, var_floor: float = 1e-3, seed: int = 0):
    """GMM de covarianza **diagonal** por EM. Devuelve `(weights, means, vars)`."""
    rng = np.random.default_rng(seed)
    n, d = mu.shape
    k = max(1, min(k, n))
    means = mu[rng.choice(n, k, replace=False)].copy()
    variances = np.tile(mu.var(axis=0) + var_floor, (k, 1))
    weights = np.full(k, 1.0 / k)

    for _ in range(iters):
        # E-step en log-espacio (estable): log N(x | μ_c, diag(σ²_c)) por componente.
        diff = mu[:, None, :] - means[None, :, :]                       # (n,k,d)
        log_norm = -0.5 * (np.log(2 * np.pi * variances)[None] + diff**2 / variances[None])
        log_prob = np.log(weights)[None] + log_norm.sum(axis=2)         # (n,k)
        log_prob -= log_prob.max(axis=1, keepdims=True)
        resp = np.exp(log_prob)
        resp /= resp.sum(axis=1, keepdims=True)                         # (n,k)
        # M-step.
        nk = resp.sum(axis=0) + 1e-8
        weights = nk / n
        means = (resp.T @ mu) / nk[:, None]
        for c in range(k):
            dc = mu - means[c]
            variances[c] = (resp[:, c][:, None] * dc**2).sum(axis=0) / nk[c] + var_floor
    return weights, means, variances


class AggregatePrior:
    """Densidad ajustada a los `μ` del posterior agregado, para muestrear `z` en generación.

    Uso: `AggregatePrior.fit(mu, kind="gmm", k=8).sample(n, rng)`. La firma de `sample` es
    `(n, rng) -> (n, latent)`, compatible con `VAE.sample_prior` (drop-in en `plot_samples`).
    """

    def __init__(self, kind: str, params):
        self.kind = kind
        self.params = params

    @classmethod
    def fit(cls, mu: np.ndarray, kind: str = "gmm", k: int = 8, seed: int = 0) -> "AggregatePrior":
        mu = np.asarray(mu, dtype=float)
        if kind == "gaussian":
            return cls("gaussian", _fit_gaussian(mu))
        if kind == "gmm":
            return cls("gmm", _fit_gmm(mu, k=k, seed=seed))
        raise ValueError(f"kind debe ser 'gaussian' o 'gmm', no {kind!r}")

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self.kind == "gaussian":
            mean, L = self.params
            return mean + rng.standard_normal((n, mean.shape[0])) @ L.T
        weights, means, variances = self.params
        comps = rng.choice(len(weights), size=n, p=weights)
        return means[comps] + rng.standard_normal((n, means.shape[1])) * np.sqrt(variances[comps])


def posterior_prior_mismatch(mu: np.ndarray) -> dict:
    """Diagnóstico del desajuste posterior-agregado vs prior `N(0,I)`.

    Si `q(z) ≈ N(0,I)`, muestrear del prior genera bien. Devuelve la norma de la media y los
    desvíos por dimensión (objetivo: `‖mean‖≈0`, `std≈1`); cuanto más lejos, peores los samples.
    """
    mu = np.asarray(mu, dtype=float)
    per_dim_std = mu.std(axis=0)
    return {
        "mean_norm": float(np.linalg.norm(mu.mean(axis=0))),
        "std_mean": float(per_dim_std.mean()),
        "std_min": float(per_dim_std.min()),
        "std_max": float(per_dim_std.max()),
    }
