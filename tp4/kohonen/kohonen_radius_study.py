import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kohonen.som import SOM
from kohonen.kohonen_europe import load_data


plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#2C3E50'
plt.rcParams['axes.labelcolor'] = '#2C3E50'
plt.rcParams['xtick.color'] = '#2C3E50'
plt.rcParams['ytick.color'] = '#2C3E50'

OUTPUT_DIR = "output/kohonen_radius_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RADII = [1.0, 3.0, 5.0]
COLORS = {1.0: "#F1C40F", 3.0: "#2ECC71", 5.0: "#3498DB"}
EPOCHS = 1000
SMOOTH_WINDOW = 50


def _smooth(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean. Returns (x_indices, smoothed_values)."""
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    x = np.arange(window - 1, len(values))
    return x, smoothed


def plot_te_convergence(X):
    print("Running Experiment: Topological Error vs. epochs for different initial radii...")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor("white")

    for r in RADII:
        print(f"  Training SOM with R₀ = {r}")
        som = SOM(
            grid_rows=5,
            grid_cols=5,
            input_dim=X.shape[1],
            lr=0.5,
            lr_decay="exponential",
            radius=r,
            radius_decay="exponential",
            neighborhood_fn="gaussian",
            epochs=EPOCHS,
            seed=42,
        )
        _, te_history = som.train(X, record_te=True)
        x_s, te_s = _smooth(te_history, SMOOTH_WINDOW)
        ax.plot(x_s, te_s, label=f"R = {r:.1f}", color=COLORS[r], linewidth=2.0, alpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')

    ax.set_title("Error Topológico en función de las Épocas", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Épocas", fontsize=11, labelpad=8)
    ax.set_ylabel("Error Topológico (TE)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.5, color="#BDC3C7")
    ax.legend(
        title="Radio Inicial (R₀)",
        frameon=True, facecolor="white", edgecolor="#E2E8F0",
        fontsize=9.5, title_fontsize=10,
    )

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "te_convergence.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {out_path}")


def plot_te_lr_convergence(X):
    print("Running Experiment: Topological Error vs. epochs for different learning rates...")

    learning_rates = [0.3, 0.5, 0.7]
    colors = {0.3: "#27AE60", 0.5: "#E67E22", 0.7: "#C0392B"}
    fixed_radius = 3.0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor("white")

    for lr in learning_rates:
        print(f"  Training SOM with LR = {lr}")
        som = SOM(
            grid_rows=5,
            grid_cols=5,
            input_dim=X.shape[1],
            lr=lr,
            lr_decay="exponential",
            radius=fixed_radius,
            radius_decay="exponential",
            neighborhood_fn="gaussian",
            epochs=EPOCHS,
            seed=42,
        )
        _, te_history = som.train(X, record_te=True)
        x_s, te_s = _smooth(te_history, SMOOTH_WINDOW)
        ax.plot(x_s, te_s, label=f"LR = {lr:.1f}", color=colors[lr], linewidth=2.0, alpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')

    ax.set_title(f"Error Topológico en función de las Épocas (R₀ = {fixed_radius})", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Épocas", fontsize=11, labelpad=8)
    ax.set_ylabel("Error Topológico (TE)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.5, color="#BDC3C7")
    ax.legend(
        title="Tasa de Aprendizaje (LR)",
        frameon=True, facecolor="white", edgecolor="#E2E8F0",
        fontsize=9.5, title_fontsize=10,
    )

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "te_lr_convergence.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {out_path}")


def main():
    _, X, _ = load_data("data/europe.csv")
    plot_te_convergence(X)
    plot_te_lr_convergence(X)


if __name__ == "__main__":
    main()
