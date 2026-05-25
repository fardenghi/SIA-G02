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
import matplotlib.cm as cm
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

def plot_pca_loadings_comparison(
    feature_names: list[str],
    loading_scaled: np.ndarray,
    loading_raw: np.ndarray,
    path: str,
) -> None:
    """2-panel: PC1 loadings for scaled vs raw."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(feature_names))
    width = 0.6

    axes[0].bar(x, loading_scaled, width, color="steelblue")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Loading PC1")
    axes[0].set_ylim(-0.7, 0.7)
    axes[0].set_title("Datos estandarizados — Loadings PC1")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, loading_raw, width, color="darkorange")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Loading PC1")
    axes[1].set_title("Datos crudos — Loadings PC1")
    dominant_idx = int(np.argmax(np.abs(loading_raw)))
    axes[1].annotate(
        f"↑ {feature_names[dominant_idx]}\ndomina",
        xy=(dominant_idx, loading_raw[dominant_idx]),
        xytext=(dominant_idx + 0.5, loading_raw[dominant_idx] * 0.7),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=8, color="red",
    )
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Efecto de la estandarización sobre PCA/Oja — Loadings",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca_scores_comparison(
    scores_scaled: np.ndarray,
    scores_raw: np.ndarray,
    countries: list[str],
    path: str,
) -> None:
    """2-panel: PC1 country scores for scaled vs raw."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    order_s = np.argsort(scores_scaled)
    colors_s = ["steelblue" if s < 0 else "indianred" for s in scores_scaled[order_s]]
    axes[0].barh([countries[i] for i in order_s], scores_scaled[order_s], color=colors_s)
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_title("Ranking países — datos estandarizados")
    axes[0].set_xlabel("Score PC1")
    axes[0].tick_params(axis="y", labelsize=7)

    order_r = np.argsort(scores_raw)
    colors_r = ["steelblue" if s < 0 else "indianred" for s in scores_raw[order_r]]
    axes[1].barh([countries[i] for i in order_r], scores_raw[order_r], color=colors_r)
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].set_title("Ranking países — datos crudos")
    axes[1].set_xlabel("Score PC1")
    axes[1].tick_params(axis="y", labelsize=7)

    fig.suptitle("Efecto de la estandarización sobre PCA/Oja — Rankings",
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


def plot_kohonen_comparison(
    countries: list[str],
    coords_scaled: np.ndarray,
    coords_raw: np.ndarray,
    scores_scaled: np.ndarray,
    scores_raw: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    path: str,
) -> None:
    """Side-by-side gradient maps: scaled vs raw."""
    def build_cells(coords, scores):
        cell_scores: dict[tuple, list[float]] = {}
        cell_ctries: dict[tuple, list[str]]  = {}
        for country, coord, score in zip(countries, coords, scores):
            key = (int(coord[0]), int(coord[1]))
            cell_scores.setdefault(key, []).append(float(score))
            cell_ctries.setdefault(key, []).append(country)
        return cell_scores, cell_ctries

    cs_s, cc_s = build_cells(coords_scaled, scores_scaled)
    cs_r, cc_r = build_cells(coords_raw,    scores_raw)

    fig, axes = plt.subplots(1, 2, figsize=(grid_cols * 2.6 * 2, grid_rows * 2.4))

    for ax, cell_scores, cell_ctries, title in [
        (axes[0], cs_s, cc_s, "Datos estandarizados"),
        (axes[1], cs_r, cc_r, "Datos crudos"),
    ]:
        avg = {k: float(np.mean(v)) for k, v in cell_scores.items()}
        norm = plt.Normalize(vmin=min(avg.values()), vmax=max(avg.values()))
        cmap = cm.RdBu_r

        ax.set_xlim(0, grid_cols)
        ax.set_ylim(0, grid_rows)
        ax.set_aspect("equal")
        for r in range(grid_rows):
            for c in range(grid_cols):
                y = r
                if (r, c) in cell_ctries:
                    ax.add_patch(plt.Rectangle((c, y), 1, 1,
                                               facecolor=cmap(norm(avg[(r, c)])),
                                               edgecolor="gray", linewidth=0.7))
                    ax.text(c + 0.5, y + 0.5, "\n".join(cell_ctries[(r, c)]),
                            ha="center", va="center", fontsize=11)
                else:
                    ax.add_patch(plt.Rectangle((c, y), 1, 1,
                                               facecolor="#e8e8e8", edgecolor="gray", linewidth=0.5))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Score PC1 promedio", fontsize=11)
        ax.set_xticks(np.arange(grid_cols) + 0.5)
        ax.set_xticklabels(range(grid_cols), fontsize=12)
        ax.set_yticks(np.arange(grid_rows) + 0.5)
        ax.set_yticklabels(range(grid_rows), fontsize=12)
        ax.set_title(title, fontsize=14)

    fig.suptitle("Efecto de la estandarización sobre Kohonen", fontsize=12, weight="bold")
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
    plot_pca_loadings_comparison(feature_names, loading_scaled, loading_raw,
                                 os.path.join(out, "pca_loadings_comparison.png"))
    plot_pca_scores_comparison(scores_scaled, scores_raw, countries,
                               os.path.join(out, "pca_scores_comparison.png"))
    plot_variance_comparison(feature_names, evr_scaled, evr_raw,
                             os.path.join(out, "variance_comparison.png"))
    plot_kohonen_comparison(countries, coords_scaled, coords_raw,
                            scores_scaled, scores_raw,
                            som_cfg["grid_rows"], som_cfg["grid_cols"],
                            os.path.join(out, "kohonen_comparison.png"))

    print(f"\nGráficos guardados en {out}/")


if __name__ == "__main__":
    main()
