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

    good = [n for n in names if rates[n][0] >= 0.5]
    rest = [n for n in names if n not in good]
    highlight_colors = ["#1E3A5F", "#E63946", "#2A9D8F", "#E9C46A", "#F4A261"]

    fig, ax = plt.subplots(figsize=(7, 4))
    rest_plotted = False
    for name in rest:
        ax.plot(noise_levels, rates[name], marker="o", linewidth=1.5,
                color="#BBBBBB", alpha=0.6, label="Resto" if not rest_plotted else "_nolegend_")
        rest_plotted = True
    for name, color in zip(good, highlight_colors):
        ax.plot(noise_levels, rates[name], marker="o", label=name, linewidth=2, color=color)
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


def plot_crosstalk_per_neuron(
    stored: dict[str, np.ndarray],
    output: str,
    subset: list[str] | None = None,
) -> None:
    """Para cada patrón ν en subset (o todos si None), calcula el crosstalk real por neurona
    usando todos los patrones almacenados:
       crosstalk_i^ν = Σ_{μ≠ν} ξ_i^μ · <ξ^μ, ξ^ν> / N
    """
    names = list(stored.keys())
    patterns = np.array([stored[n] for n in names], dtype=np.float64)
    p, N = patterns.shape
    grid = int(np.sqrt(N))

    show = subset if subset is not None else names
    show_idx = [names.index(n) for n in show]

    cell_size = 0.4 if grid <= 5 else 0.25
    fig_size = grid * cell_size
    fig, axes = plt.subplots(1, len(show), figsize=(len(show) * fig_size + 1.5, fig_size))
    axes = np.atleast_1d(axes)

    vmax = 0.0
    crosstalks = []
    for nu in show_idx:
        ct = np.zeros(N)
        for mu in range(p):
            if mu != nu:
                dot = np.dot(patterns[mu], patterns[nu]) / N
                ct += patterns[mu] * dot
        crosstalks.append(ct)
        vmax = max(vmax, np.abs(ct).max())

    for (ax, ct, name) in zip(axes, crosstalks, show):
        im = ax.imshow(ct.reshape(grid, grid), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="none")
        ax.set_title(name, fontsize=11, fontweight="bold")
        if grid > 5:
            ax.set_xticks(np.arange(-0.5, grid, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        if grid <= 5:
            for i in range(grid):
                for j in range(grid):
                    ax.text(j, i, f"{ct.reshape(grid, grid)[i, j]:.2f}",
                            ha="center", va="center", fontsize=6)

    fig.subplots_adjust(left=0.02, right=0.82)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Crosstalk por neurona")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


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
