"""Visualizaciones: heatmaps de glifos, scatter latente (1a3) y letra nueva (1a4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display, apto para generar archivos
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import data as data_mod  # noqa: E402


def plot_glyph(vector: np.ndarray, ax=None, title: str | None = None):
    """Renderiza un vector de 35 valores como una grilla 7×5 (heatmap)."""
    grid = data_mod.to_grid(vector)
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2.5))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    return ax


def plot_glyph_grid(vectors: np.ndarray, labels=None, ncols: int = 8, path=None):
    """Renderiza varios glifos en una grilla y opcionalmente guarda a `path`."""
    vectors = np.asarray(vectors)
    n = vectors.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.5))
    axes = np.atleast_1d(axes).reshape(-1)
    for i in range(len(axes)):
        if i < n:
            title = None if labels is None else str(labels[i])
            plot_glyph(vectors[i], ax=axes[i], title=title)
        else:
            axes[i].axis("off")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def plot_latent_scatter(Z: np.ndarray, labels, path=None, title="Espacio latente 2D"):
    """Scatter 2D de los códigos latentes, anotando cada punto con su carácter (1a3)."""
    Z = np.asarray(Z)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Z[:, 0], Z[:, 1], c="tab:blue", s=40, alpha=0.7)
    for (x, y), lab in zip(Z, labels):
        ax.annotate(str(lab), (x, y), fontsize=11,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def plot_reconstruction(X: np.ndarray, X_hat: np.ndarray, labels=None, path=None,
                        n: int | None = None):
    """Render comparativo entrada vs reconstrucción (binarizada a 0.5)."""
    X = np.asarray(X)
    X_hat = np.asarray(X_hat)
    n = X.shape[0] if n is None else min(n, X.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 3.2))
    axes = np.atleast_2d(axes)
    for i in range(n):
        title = None if labels is None else str(labels[i])
        plot_glyph(X[i], ax=axes[0, i], title=title)
        plot_glyph((X_hat[i] >= 0.5).astype(float), ax=axes[1, i])
    axes[0, 0].set_ylabel("in", rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("out", rotation=0, ha="right", va="center")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def plot_denoise_triplet(noisy, recon, clean, labels=None, path=None, n: int = 8):
    """Visualiza ruidoso → reconstruido → limpio por columna (8.4)."""
    noisy, recon, clean = map(np.asarray, (noisy, recon, clean))
    n = min(n, noisy.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(n * 1.2, 4.5))
    axes = np.atleast_2d(axes)
    rows = [("ruidoso", noisy), ("recon", (recon >= 0.5).astype(float)), ("limpio", clean)]
    for r, (name, mat) in enumerate(rows):
        for i in range(n):
            title = None if (labels is None or r != 0) else str(labels[i])
            plot_glyph(mat[i], ax=axes[r, i], title=title)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig
