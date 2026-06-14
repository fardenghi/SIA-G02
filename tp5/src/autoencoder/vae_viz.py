"""Visualizaciones del VAE (Ej2): scatter de medias, manifold 2D, muestras nuevas (2c),
interpolación y reconstrucción en escala de grises.

Las imágenes son vectores de `size·size` que se muestran como grillas `size×size`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display, apto para generar archivos
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .vae import VAE  # noqa: E402


def _img(vec: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(vec).reshape(size, size)


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)


def plot_image_grid(images: np.ndarray, size: int, ncols: int = 8, titles=None,
                    path=None, suptitle: str | None = None):
    """Renderiza varias imágenes (filas de `size·size`) en una grilla de grises."""
    images = np.asarray(images)
    n = images.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.3))
    axes = np.atleast_1d(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(_img(images[i], size), cmap="Greys", vmin=0, vmax=1,
                      interpolation="nearest")
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_latent_means(mu: np.ndarray, labels, path=None,
                      title="Espacio latente del VAE (medias μ)"):
    """Scatter 2D de las medias `μ` del posterior, anotando cada punto con su etiqueta."""
    mu = np.asarray(mu)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mu[:, 0], mu[:, 1], c="tab:purple", s=30, alpha=0.6)
    if labels is not None:
        for (x, y), lab in zip(mu, labels):
            ax.annotate(str(lab), (x, y), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_latent_manifold(vae: VAE, size: int, n: int = 15, span: float = 2.5, path=None,
                         title="Manifold generativo (grid de z decodificado)"):
    """Decodifica una grilla `n×n` de `z` en `[-span, span]²` y la muestra como un mosaico.

    Es la visualización clásica del VAE de latente 2D: recorre el espacio latente y muestra
    cómo varía la imagen generada de forma continua. Solo aplica con `latent_dim == 2`.
    """
    if vae.latent_dim != 2:
        raise ValueError("plot_latent_manifold requiere latent_dim == 2")
    grid = np.linspace(-span, span, n)
    canvas = np.zeros((n * size, n * size))
    for i, zy in enumerate(grid[::-1]):  # de arriba (zy alto) hacia abajo
        zs = np.column_stack([grid, np.full(n, zy)])
        imgs = vae.decode(zs)
        for j in range(n):
            canvas[i * size:(i + 1) * size, j * size:(j + 1) * size] = _img(imgs[j], size)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(canvas, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("z1 →")
    ax.set_ylabel("z2 →")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_samples(vae: VAE, size: int, n: int = 16, rng=None, path=None,
                 title="Muestras nuevas del prior z ~ N(0, I)"):
    """Genera `n` muestras nuevas muestreando `z ~ N(0,I)` y decodificando (Ej2c)."""
    z = vae.sample_prior(n, rng=rng)
    samples = vae.decode(z)
    return plot_image_grid(samples, size, ncols=min(8, n), path=path, suptitle=title)


def plot_interpolation(vae: VAE, x_start: np.ndarray, x_end: np.ndarray, size: int,
                       steps: int = 10, path=None,
                       title="Interpolación en el espacio latente"):
    """Interpola en el latente entre dos imágenes (vía sus medias) y decodifica cada paso."""
    mu_start, _ = vae.encode(np.asarray(x_start).reshape(1, -1))
    mu_end, _ = vae.encode(np.asarray(x_end).reshape(1, -1))
    alphas = np.linspace(0.0, 1.0, steps).reshape(-1, 1)
    zs = (1.0 - alphas) * mu_start + alphas * mu_end
    imgs = vae.decode(zs)
    return plot_image_grid(imgs, size, ncols=steps, path=path, suptitle=title)


def plot_reconstruction_gray(X: np.ndarray, X_hat: np.ndarray, labels=None, size: int = 28,
                             n: int | None = None, path=None):
    """Compara entrada vs reconstrucción (grises) en dos filas."""
    X = np.asarray(X)
    X_hat = np.asarray(X_hat)
    n = X.shape[0] if n is None else min(n, X.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.8))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[0, i].imshow(_img(X[i], size), cmap="Greys", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[1, i].imshow(_img(X_hat[i], size), cmap="Greys", vmin=0, vmax=1,
                          interpolation="nearest")
        if labels is not None:
            axes[0, i].set_title(str(labels[i]), fontsize=8)
        for r in (0, 1):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    axes[0, 0].set_ylabel("in", rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("out", rotation=0, ha="right", va="center")
    fig.tight_layout()
    _save(fig, path)
    return fig
