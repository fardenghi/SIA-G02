"""Gráficos de diagnóstico del VAE para validar el encoder.

A diferencia de `vae_viz.py` (que dibuja las imágenes generadas), acá se grafican las
**métricas** que dicen si el encoder produce un posterior sano:

  - curvas de entrenamiento (ELBO/recon/KL + warmup de β): convergencia y no-colapso;
  - KL por dimensión latente: detecta dimensiones "muertas" (sin usar);
  - distribución de μ y σ del posterior: ni colapso (σ→0) ni ignorar la entrada (σ→1, μ→0);
  - posterior agregado vs prior N(0,I): si q(z) matchea p(z), muestrear del prior genera bien.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .vae import VAE  # noqa: E402


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)


def kl_per_dim(vae: VAE, X: np.ndarray) -> np.ndarray:
    """KL aportada por cada dimensión latente (media sobre el batch).

    La suma sobre dimensiones coincide con `vae.kl_divergence(μ, logσ²)`. Una dimensión con
    KL≈0 es una "unidad muerta": el encoder no la usa.
    """
    mu, logvar = vae.encode(X)
    kld = -0.5 * (1.0 + logvar - mu**2 - np.exp(logvar))  # (N, L)
    return kld.mean(axis=0)


def active_units(vae: VAE, X: np.ndarray, threshold: float = 0.1) -> int:
    """Cuenta las dimensiones latentes "activas": KL media por dim > `threshold` (nats).

    Es el conteo del pruning automático del VAE: con `latent_dim` grande, varias dims
    colapsan a KL≈0 (el encoder no las usa) y solo unas pocas quedan activas. Ese número es
    una estimación de la dimensionalidad intrínseca que el dataset le pide al cuello latente.
    """
    return int((kl_per_dim(vae, X) > threshold).sum())


def project_pca(mu: np.ndarray, k: int = 2):
    """Proyecta `μ` (N, L) a sus `k` componentes principales (PCA desde cero con SVD).

    Devuelve `(proj, var_ratio)` donde `proj` es (N, k) y `var_ratio` es la fracción de
    varianza explicada por cada uno de los `k` componentes (ordenados de mayor a menor).
    Con `k == L` la proyección es invertible: `proj @ Vt + media == μ`.
    """
    mu = np.asarray(mu, dtype=float)
    mean = mu.mean(axis=0)
    centered = mu - mean
    # SVD del centrado: las filas de Vt son los componentes principales (autovectores de la
    # covarianza), con valores singulares S en orden decreciente.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s**2
    total = variance.sum()
    var_ratio = variance / total if total > 0 else np.zeros_like(variance)
    k = min(k, vt.shape[0])
    proj = centered @ vt[:k].T  # (N, k)
    return proj, var_ratio[:k]


def plot_training_curves(df, path=None, title="Curvas de entrenamiento del VAE"):
    """ELBO/recon (arriba) y KL/β (abajo) vs época, desde el DataFrame del tracker."""
    epochs = np.asarray(df["epoch"], dtype=float)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_top.plot(epochs, df["elbo"], "-", color="tab:blue", label="ELBO (recon + β·KL)")
    ax_top.plot(epochs, df["recon"], "-", color="tab:green", label="reconstrucción")
    ax_top.set_ylabel("nats / muestra")
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(title)

    ax_bot.plot(epochs, df["kl"], "-", color="tab:red", label="KL")
    ax_bot.set_ylabel("KL")
    ax_bot.set_xlabel("época")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_ylim(bottom=0)
    # β en eje secundario: muestra el warmup.
    ax_beta = ax_bot.twinx()
    ax_beta.plot(epochs, df["beta"], "--", color="tab:gray", alpha=0.8, label="β (warmup)")
    ax_beta.set_ylabel("β")
    ax_beta.set_ylim(bottom=0)
    lines = ax_bot.get_lines() + ax_beta.get_lines()
    ax_bot.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_kl_per_dim(vae: VAE, X: np.ndarray, path=None, threshold: float = 0.1,
                    title="KL por dimensión latente (uso del cuello)"):
    """Barras de KL por dimensión, ordenadas de mayor a menor, con umbral de unidad activa.

    Con `latent_dim > 2` es la figura estrella: muestra el pruning automático del VAE. Las
    dims sobre `threshold` (nats) se pintan llenas (activas); las que colapsaron a KL≈0, en
    gris. El título anota cuántas quedaron activas.
    """
    kld = kl_per_dim(vae, X)
    order = np.argsort(kld)[::-1]  # de mayor a menor KL
    kld_sorted = kld[order]
    active = kld_sorted > threshold
    n_active = int(active.sum())

    fig, ax = plt.subplots(figsize=(max(5, 0.4 * len(kld) + 1), 4))
    dims = np.arange(len(kld))
    colors = np.where(active, "tab:purple", "lightgray")
    ax.bar(dims, kld_sorted, color=colors, alpha=0.85)
    ax.set_xticks(dims)
    ax.set_xticklabels([f"z{i + 1}" for i in order], rotation=90 if len(kld) > 12 else 0,
                       fontsize=8 if len(kld) > 12 else 9)
    ax.axhline(threshold, color="tab:red", ls="--", lw=1,
               label=f"umbral unidad activa ({threshold:g} nats)")
    ax.set_ylabel("KL media")
    ax.set_xlabel("dimensión latente (ordenada por KL)")
    ax.set_title(f"{title}\n{n_active}/{len(kld)} dims activas")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_posterior_stats(vae: VAE, X: np.ndarray, path=None,
                         title="Distribución del posterior q(z|x)"):
    """Histogramas de μ y de σ = e^{logσ²/2} sobre el dataset.

    Sano: σ claramente en (0, 1) — ni σ→0 (colapso) ni σ→1 con μ→0 (la red ignora la entrada).
    """
    mu, logvar = vae.encode(X)
    sigma = np.exp(0.5 * logvar)
    fig, (ax_mu, ax_sig) = plt.subplots(1, 2, figsize=(9, 4))
    ax_mu.hist(mu.reshape(-1), bins=30, color="tab:blue", alpha=0.8)
    ax_mu.set_title("μ (medias)")
    ax_mu.set_xlabel("valor de μ")
    ax_mu.grid(True, alpha=0.3)
    ax_sig.hist(sigma.reshape(-1), bins=30, color="tab:orange", alpha=0.8)
    ax_sig.axvline(1.0, color="tab:gray", ls="--", label="σ=1 (prior)")
    ax_sig.set_title("σ (desvíos)")
    ax_sig.set_xlabel("valor de σ")
    ax_sig.legend(fontsize=9)
    ax_sig.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_aggregate_posterior(vae: VAE, X: np.ndarray, path=None,
                             title="Posterior agregado vs prior N(0, I)"):
    """Scatter de todos los μ con los contornos del prior N(0,I) (radios 1σ/2σ/3σ).

    Si la nube de μ queda contenida y centrada en los contornos, q(z) ≈ p(z) y muestrear del
    prior cae en zonas pobladas (buena generación). Requiere latent_dim == 2.
    """
    if vae.latent_dim != 2:
        raise ValueError("plot_aggregate_posterior requiere latent_dim == 2")
    mu, _ = vae.encode(X)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mu[:, 0], mu[:, 1], s=15, alpha=0.5, color="tab:purple", label="μ(x)")
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in (1, 2, 3):
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="tab:gray", ls="--",
                alpha=0.6, lw=1)
    ax.axhline(0, color="tab:gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="tab:gray", lw=0.5, alpha=0.4)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
    ax.set_title(title + "  (círculos: 1σ/2σ/3σ)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_beta_sweep(df, path=None, title="Barrido de β: recon vs KL"):
    """Recon y KL vs β (escala log) — frontera del trade-off y zona de colapso.

    `df` con columnas `beta`, `recon`, `kl` (lo que produce `scripts/vae_beta_sweep.py`).
    """
    beta = np.asarray(df["beta"], dtype=float)
    fig, ax_recon = plt.subplots(figsize=(7, 4.5))
    ax_recon.plot(beta, df["recon"], "o-", color="tab:green", label="reconstrucción")
    ax_recon.set_xscale("log")
    ax_recon.set_xlabel("β (escala log)")
    ax_recon.set_ylabel("reconstrucción (nats/muestra)", color="tab:green")
    ax_recon.grid(True, alpha=0.3)
    ax_kl = ax_recon.twinx()
    ax_kl.plot(beta, df["kl"], "s--", color="tab:red", label="KL")
    ax_kl.set_ylabel("KL", color="tab:red")
    ax_kl.set_ylim(bottom=0)
    ax_kl.axhline(0.05, color="tab:gray", ls=":", alpha=0.7)
    lines = ax_recon.get_lines() + ax_kl.get_lines()
    ax_recon.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=9)
    ax_recon.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_latent_sweep(df, path=None, title="Barrido de latent_dim: recon vs unidades activas"):
    """Recon (det) y nº de unidades activas vs `latent_dim` (escala log₂).

    Responde "¿cuántas dimensiones efectivas piden los emojis?": la recon mejora al subir el
    latente y luego satura, mientras las unidades activas se estabilizan en la dimensionalidad
    intrínseca. `df` con columnas `latent_dim`, `recon`, `active_units` (lo que produce
    `scripts/vae_latent_sweep.py`).
    """
    from matplotlib.ticker import ScalarFormatter

    ld = np.asarray(df["latent_dim"], dtype=float)
    fig, ax_recon = plt.subplots(figsize=(7, 4.5))
    ax_recon.plot(ld, df["recon"], "o-", color="tab:green", label="reconstrucción (det)")
    ax_recon.set_xscale("log", base=2)
    ax_recon.set_xticks(ld)
    ax_recon.xaxis.set_major_formatter(ScalarFormatter())
    ax_recon.set_xlabel("latent_dim (escala log₂)")
    ax_recon.set_ylabel("reconstrucción (nats/muestra)", color="tab:green")
    ax_recon.grid(True, alpha=0.3)
    ax_act = ax_recon.twinx()
    ax_act.plot(ld, df["active_units"], "s--", color="tab:blue", label="unidades activas")
    ax_act.set_ylabel("nº unidades activas", color="tab:blue")
    ax_act.set_ylim(bottom=0)
    lines = ax_recon.get_lines() + ax_act.get_lines()
    ax_recon.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=9)
    ax_recon.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig
