"""Visualizaciones didácticas para la red de Hopfield.

Reutiliza ideas del runner del compañero (curva de recuperación por letra,
evolución de energía, crosstalk) integradas con nuestro `HopfieldNetwork`.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from hopfield.hopfield import HopfieldNetwork, add_noise

# Paleta inspirada en la del compañero: crema (inactivo) → azul oscuro (activo)
PATTERN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hopfield", ["#F5F0E8", "#1E3A5F"]
)


def plot_energy_evolution(
    energies: list[float],
    letter_name: str,
    output: str,
) -> None:
    """Curva de energía vs iteración para una sola consulta."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(len(energies)), energies, marker="o", linewidth=2, color="#1E3A5F")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Energía H")
    ax.set_title(f"Energía durante la recuperación de '{letter_name}'")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def plot_crosstalk(
    stored: dict[str, np.ndarray],
    output: str,
) -> None:
    """Heatmap de correlación normalizada xi·xj / N entre patrones almacenados."""
    names = list(stored.keys())
    N = len(next(iter(stored.values())))
    mat = np.array([
        [float(np.dot(stored[a], stored[b])) / N for b in names]
        for a in names
    ])
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(names) + 2),
                                    max(3.5, 0.5 * len(names) + 1.5)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    fontsize = 9 if len(names) <= 8 else 6
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=fontsize)
    fig.colorbar(im, ax=ax, label="Correlación normalizada")
    ax.set_title("Correlación entre patrones almacenados")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def plot_recovery_rate_vs_noise(
    net: HopfieldNetwork,
    stored: dict[str, np.ndarray],
    noise_levels: list[float],
    n_trials: int,
    rng: np.random.Generator,
    output: str,
    mode: str = "sync",
    max_steps: int = 50,
) -> dict[str, list[float]]:
    """Curva de tasa de recuperación vs ruido, una línea por letra."""
    names = list(stored.keys())
    rates: dict[str, list[float]] = {n: [] for n in names}

    for noise in noise_levels:
        for name in names:
            v = stored[name]
            correct = 0
            for _ in range(n_trials):
                noisy = add_noise(v, noise, rng)
                final, _h, _e, _c = net.recall(
                    noisy, mode=mode, max_steps=max_steps, rng=rng
                )
                if np.array_equal(final, v):
                    correct += 1
            rates[name].append(correct / n_trials)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        ax.plot(noise_levels, rates[name], marker="o", label=name, linewidth=2)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="50% ruido")
    ax.set_xlabel("Nivel de ruido (fracción de bits invertidos)")
    ax.set_ylabel("Tasa de recuperación")
    ax.set_title("Tasa de recuperación vs nivel de ruido")
    ax.legend(fontsize=8, ncol=2 if len(names) > 6 else 1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return rates


def plot_pattern_grid(
    patterns: list[tuple[str, np.ndarray]],
    output: str,
    grid: int = 5,
    title: str | None = None,
) -> None:
    """Grilla de patrones con la paleta crema/azul."""
    n = len(patterns)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes_flat = np.atleast_2d(axes).ravel() if n > 1 else [axes]
    for ax in axes_flat:
        ax.axis("off")
    for ax, (label, pat) in zip(axes_flat, patterns):
        ax.axis("on")
        ax.imshow(pat.reshape(grid, grid), cmap=PATTERN_CMAP, vmin=-1, vmax=1,
                  interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
