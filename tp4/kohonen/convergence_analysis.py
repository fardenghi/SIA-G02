import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kohonen.som import SOM
from kohonen.kohonen_europe import load_data

# Visual styling for minimalist, professional, presentation-ready plots
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#2C3E50'
plt.rcParams['axes.labelcolor'] = '#2C3E50'
plt.rcParams['xtick.color'] = '#2C3E50'
plt.rcParams['ytick.color'] = '#2C3E50'

# Output directory
OUTPUT_DIR = "output/convergence"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_radio_convergence(X: np.ndarray):
    print("Running Experiment: Neighborhood Radius (R) convergence...")
    radii = [0.5, 1.5, 3.0, 5.0]
    colors = {0.5: "#E74C3C", 1.5: "#F1C40F", 3.0: "#2ECC71", 5.0: "#3498DB"}
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor("white")
    
    for r in radii:
        print(f"  Training SOM with Radius = {r}")
        som = SOM(
            grid_rows=5,
            grid_cols=5,
            input_dim=X.shape[1],
            lr=0.5,
            lr_decay="exponential",
            radius=r,
            radius_decay="exponential",
            neighborhood_fn="gaussian",
            epochs=1000,
            seed=42
        )
        history = som.train(X)
        ax.plot(history, label=f"R = {r:.1f}", color=colors[r], linewidth=2.0, alpha=0.95)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    ax.set_title("Análisis de Convergencia", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Épocas", fontsize=11, labelpad=8)
    ax.set_ylabel("Distancia promedio a BMU", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.5, color="#BDC3C7")
    
    ax.legend(title="Radio de Vecindad (R)", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "convergencia_radio.png"), dpi=150)
    plt.close(fig)


def plot_lr_convergence(X: np.ndarray):
    print("Running Experiment: Learning Rate (LR) convergence...")
    learning_rates = [0.3, 0.5, 0.7]
    colors = {0.3: "#27AE60", 0.5: "#E67E22", 0.7: "#C0392B"}
    
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
            radius=3.0,
            radius_decay="exponential",
            neighborhood_fn="gaussian",
            epochs=1000,
            seed=42
        )
        history = som.train(X)
        ax.plot(history, label=f"LR = {lr:.1f}", color=colors[lr], linewidth=2.0, alpha=0.95)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    ax.set_title("Análisis de Convergencia", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Épocas", fontsize=11, labelpad=8)
    ax.set_ylabel("Distancia promedio a BMU", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.5, color="#BDC3C7")
    
    ax.legend(title="Tasa de Aprendizaje (LR)", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "convergencia_lr.png"), dpi=150)
    plt.close(fig)


def main():
    print("=========================================================")
    print("📊 INICIANDO ANÁLISIS DE CONVERGENCIA MINIMALISTA 📊")
    print("=========================================================")
    
    # Load standardized dataset
    _, X, _ = load_data("data/europe.csv")
    
    # Generate the two minimalist plots
    plot_radio_convergence(X)
    plot_lr_convergence(X)
    
    print("\n=========================================================")
    print("🎉 Gráficos de convergencia minimalistas creados con éxito.")
    print(f"📁 Guardados en: {OUTPUT_DIR}/")
    print("=========================================================")


if __name__ == "__main__":
    main()
