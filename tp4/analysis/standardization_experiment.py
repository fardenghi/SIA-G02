"""
2×2 Standardization experiment (Section 3 of the guide).

              | WITH STANDARDIZATION | WITHOUT STANDARDIZATION |
  PCA / Oja   | Correlation matrix   | Covariance matrix       |
  Kohonen     | Homogeneous distances| Magnitude hijacking     |

Outputs:
  pca_comparison.png     : loadings + country scores, scaled vs raw (4-panel)
  variance_comparison.png: scree plot comparison, shows dominance in raw PCA
  kohonen_comparison.png : side-by-side country maps (scaled vs raw)
  component_planes_comparison.png: component planes side-by-side (scaled vs raw)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kohonen.som import SOM


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    countries = df["Country"].tolist()
    feature_df = df.drop(columns=["Country"])
    feature_names = feature_df.columns.tolist()
    X_raw = feature_df.to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X_raw)
    return countries, X_raw, X_scaled, feature_names


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def run_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA()
    pca.fit(X)
    scores_pc1 = pca.transform(X)[:, 0]
    loading = pca.components_[0]
    return loading, scores_pc1, pca.explained_variance_ratio_


def run_som(X: np.ndarray, cfg: dict) -> tuple[SOM, np.ndarray]:
    som = SOM(
        grid_rows=cfg["grid_rows"], grid_cols=cfg["grid_cols"], input_dim=X.shape[1],
        lr=cfg["lr"], lr_decay=cfg["lr_decay"], radius=cfg["radius"],
        radius_decay=cfg["radius_decay"], neighborhood_fn=cfg["neighborhood_fn"],
        epochs=cfg["epochs"], seed=cfg["seed"],
    )
    som.train(X)
    coords = som.predict(X)
    return som, coords


def canonical_sign(loading: np.ndarray) -> np.ndarray:
    """Flip so the largest-magnitude component is positive (for consistent display)."""
    dominant = int(np.argmax(np.abs(loading)))
    return loading if loading[dominant] >= 0 else -loading


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pca_comparison(
    feature_names: list[str],
    loading_scaled: np.ndarray,
    loading_raw: np.ndarray,
    scores_scaled: np.ndarray,
    scores_raw: np.ndarray,
    countries: list[str],
    path: str,
) -> None:
    """4-panel: loadings and country scores for scaled vs raw."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(feature_names))
    width = 0.6

    # --- Panel A: Loadings (scaled) ---
    axes[0, 0].bar(x, loading_scaled, width, color="steelblue")
    axes[0, 0].axhline(0, color="black", linewidth=0.5)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    axes[0, 0].set_ylabel("Loading PC1")
    axes[0, 0].set_ylim(-0.7, 0.7)
    axes[0, 0].set_title("(A) Datos estandarizados — loadings PC1\n"
                          "(cada variable contribuye de forma balanceada)")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # --- Panel B: Loadings (raw) ---
    axes[0, 1].bar(x, loading_raw, width, color="darkorange")
    axes[0, 1].axhline(0, color="black", linewidth=0.5)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    axes[0, 1].set_ylabel("Loading PC1")
    axes[0, 1].set_title("(B) Datos crudos — loadings PC1\n"
                          "(dominados por la variable de mayor varianza absoluta)")
    dominant_idx = int(np.argmax(np.abs(loading_raw)))
    axes[0, 1].annotate(
        f"↑ {feature_names[dominant_idx]}\ndomina",
        xy=(dominant_idx, loading_raw[dominant_idx]),
        xytext=(dominant_idx + 0.5, loading_raw[dominant_idx] * 0.7),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=8, color="red",
    )
    axes[0, 1].grid(axis="y", alpha=0.3)

    # --- Panel C: Country scores (scaled) ---
    order_s = np.argsort(scores_scaled)
    colors_s = ["steelblue" if s < 0 else "indianred" for s in scores_scaled[order_s]]
    axes[1, 0].barh([countries[i] for i in order_s], scores_scaled[order_s], color=colors_s)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_title("(C) Ranking países — datos estandarizados")
    axes[1, 0].set_xlabel("Score PC1")
    axes[1, 0].tick_params(axis="y", labelsize=7)

    # --- Panel D: Country scores (raw) ---
    order_r = np.argsort(scores_raw)
    colors_r = ["steelblue" if s < 0 else "indianred" for s in scores_raw[order_r]]
    axes[1, 1].barh([countries[i] for i in order_r], scores_raw[order_r], color=colors_r)
    axes[1, 1].axvline(0, color="black", linewidth=0.5)
    axes[1, 1].set_title("(D) Ranking países — datos crudos\n"
                          "(ordenamiento controlado por escala, no por calidad de vida)")
    axes[1, 1].set_xlabel("Score PC1")
    axes[1, 1].tick_params(axis="y", labelsize=7)

    fig.suptitle("Módulo A vs B: Efecto de la estandarización sobre PCA/Oja",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variance_comparison(
    feature_names: list[str],
    evr_scaled: np.ndarray,
    evr_raw: np.ndarray,
    path: str,
) -> None:
    """Scree plot for standardized vs raw — shows how PC1 concentration differs."""
    n = len(feature_names)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x, evr_scaled * 100, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"PC{i+1}" for i in range(n)])
    axes[0].set_ylabel("Varianza explicada (%)")
    axes[0].set_title(f"Datos estandarizados\nPC1 explica {evr_scaled[0]*100:.1f}% — varianza distribuida")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, evr_raw * 100, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"PC{i+1}" for i in range(n)])
    axes[1].set_ylabel("Varianza explicada (%)")
    axes[1].set_title(f"Datos crudos\nPC1 explica {evr_raw[0]*100:.1f}% — casi toda la varianza concentrada")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Scree plot: efecto de la estandarización sobre la distribución de varianza",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_country_map(
    ax: plt.Axes,
    countries: list[str],
    coords: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    title: str,
    facecolor: str,
) -> None:
    assignments: dict[tuple, list[str]] = {}
    for country, (r, c) in zip(countries, coords):
        assignments.setdefault((int(r), int(c)), []).append(country)

    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_aspect("equal")
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.grid(True, color="gray", linewidth=0.4)

    for (r, c), country_list in assignments.items():
        y = grid_rows - 1 - r
        ax.text(c + 0.5, y + 0.5, "\n".join(country_list),
                ha="center", va="center", fontsize=6.2,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=facecolor, alpha=0.35))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")


def plot_kohonen_comparison(
    countries: list[str],
    coords_scaled: np.ndarray,
    coords_raw: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    path: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(grid_cols * 2.6 * 2, grid_rows * 2.4))
    _draw_country_map(ax1, countries, coords_scaled, grid_rows, grid_cols,
                      "(A) Datos estandarizados\nagrupamiento por estructura macroeconómica",
                      "steelblue")
    _draw_country_map(ax2, countries, coords_raw, grid_rows, grid_cols,
                      "(B) Datos crudos\n'secuestro' por magnitud: GDP y Área dominan la distancia euclidiana",
                      "darkorange")
    fig.suptitle("Módulo A vs B: Efecto de la estandarización sobre Kohonen",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_component_planes_comparison(
    som_scaled: SOM,
    som_raw: SOM,
    feature_names: list[str],
    path: str,
) -> None:
    """Side-by-side component planes: standardized (top row) vs raw (bottom row)."""
    n = len(feature_names)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 2 * 2.8))

    for i, name in enumerate(feature_names):
        for row, (som, label) in enumerate([(som_scaled, "Estándar"), (som_raw, "Crudo")]):
            plane = som.weights[:, :, i]
            im = axes[row, i].imshow(plane, cmap="RdBu_r", interpolation="nearest")
            plt.colorbar(im, ax=axes[row, i], fraction=0.046, pad=0.04)
            axes[row, i].set_title(f"{name}\n({label})", fontsize=8)
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    fig.suptitle(
        "Planos de Componentes: datos estandarizados (fila superior) vs crudos (fila inferior)\n"
        "Con datos crudos, el patrón de GDP/Área aplasta la estructura de las demás variables",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    feature_names: list[str],
    loading_scaled: np.ndarray,
    loading_raw: np.ndarray,
    evr_scaled: np.ndarray,
    evr_raw: np.ndarray,
) -> None:
    print("\n=== EXPERIMENTO ESTANDARIZACIÓN 2×2 ===")
    print(f"\nVarianza explicada por PC1:")
    print(f"  Estandarizado: {evr_scaled[0]*100:.1f}%")
    print(f"  Crudo:         {evr_raw[0]*100:.1f}%")

    print("\nLoadings PC1:")
    print(f"  {'Variable':<14} {'Estandarizado':>14} {'Crudo':>14}")
    print("  " + "-" * 44)
    for name, ls, lr in zip(feature_names, loading_scaled, loading_raw):
        print(f"  {name:<14} {ls:>14.4f} {lr:>14.4f}")

    dominant_idx = int(np.argmax(np.abs(loading_raw)))
    print(f"\n  Variable dominante en datos crudos: {feature_names[dominant_idx]}"
          f"  (loading={loading_raw[dominant_idx]:.4f})")
    negligible = [name for name, lr in zip(feature_names, loading_raw) if abs(lr) < 0.05]
    if negligible:
        print(f"  Variables prácticamente ignoradas:  {', '.join(negligible)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento estandarización 2×2")
    parser.add_argument("--kohonen-config", default="configs/kohonen_europe.json")
    parser.add_argument("--data", default="data/europe.csv")
    parser.add_argument("--output-dir", default="output/standardization")
    args = parser.parse_args()

    with open(args.kohonen_config) as f:
        som_cfg = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    countries, X_raw, X_scaled, feature_names = load_data(args.data)

    # PCA on both — flip sign so the dominant loading is always positive
    loading_scaled, scores_scaled, evr_scaled = run_pca(X_scaled)
    if loading_scaled[np.argmax(np.abs(loading_scaled))] < 0:
        loading_scaled, scores_scaled = -loading_scaled, -scores_scaled

    loading_raw, scores_raw, evr_raw = run_pca(X_raw)
    if loading_raw[np.argmax(np.abs(loading_raw))] < 0:
        loading_raw, scores_raw = -loading_raw, -scores_raw

    # SOM on both (same config)
    print("Entrenando SOM con datos estandarizados...")
    som_scaled, coords_scaled = run_som(X_scaled, som_cfg)
    print("Entrenando SOM con datos crudos...")
    som_raw, coords_raw = run_som(X_raw, som_cfg)

    print_summary(feature_names, loading_scaled, loading_raw, evr_scaled, evr_raw)

    out = args.output_dir
    print("\nGenerando gráficos...")
    plot_pca_comparison(feature_names, loading_scaled, loading_raw,
                        scores_scaled, scores_raw, countries,
                        os.path.join(out, "pca_comparison.png"))
    plot_variance_comparison(feature_names, evr_scaled, evr_raw,
                             os.path.join(out, "variance_comparison.png"))
    plot_kohonen_comparison(countries, coords_scaled, coords_raw,
                            som_cfg["grid_rows"], som_cfg["grid_cols"],
                            os.path.join(out, "kohonen_comparison.png"))
    plot_component_planes_comparison(som_scaled, som_raw, feature_names,
                                     os.path.join(out, "component_planes_comparison.png"))

    print(f"\nGráficos guardados en {out}/")


if __name__ == "__main__":
    main()
