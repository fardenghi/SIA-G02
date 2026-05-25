"""
Cross-validation experiment: PCA vs Kohonen (Sections 1.1, 1.2, 2.1)

Runs both models on standardized europe.csv and produces:
  - 1.1_pc1_ranking.png         : bar chart ranking all countries by PC1 score
  - 1.1_kohonen_extremes.png    : Kohonen map highlighting top/bottom PC1 countries
  - 1.2_component_planes.png    : SOM component planes ordered by PCA loading
  - 1.2_plane_correlations.png  : correlation between planes vs PCA loading structure
  - 2.1_pc1_gradient_map.png    : Kohonen grid colored by PC1 score per cell
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kohonen.som import SOM


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(path)
    countries = df["Country"].tolist()
    feature_df = df.drop(columns=["Country"])
    feature_names = feature_df.columns.tolist()
    X = StandardScaler().fit_transform(feature_df.to_numpy(dtype=float))
    return countries, X, feature_names


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_som(X: np.ndarray, cfg: dict) -> SOM:
    som = SOM(
        grid_rows=cfg["grid_rows"], grid_cols=cfg["grid_cols"], input_dim=X.shape[1],
        lr=cfg["lr"], lr_decay=cfg["lr_decay"], radius=cfg["radius"],
        radius_decay=cfg["radius_decay"], neighborhood_fn=cfg["neighborhood_fn"],
        epochs=cfg["epochs"], seed=cfg["seed"],
    )
    som.train(X)
    return som


def get_pc1(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X).ravel()
    loading = pca.components_[0]
    return loading, scores


# ---------------------------------------------------------------------------
# Section 1.1: Extreme countries
# ---------------------------------------------------------------------------

def analyze_extremes(
    countries: list[str],
    som_coords: np.ndarray,
    pc1_scores: np.ndarray,
    n_extreme: int = 3,
) -> None:
    order = np.argsort(pc1_scores)
    bottom = [(countries[i], som_coords[i], pc1_scores[i]) for i in order[:n_extreme]]
    top = [(countries[i], som_coords[i], pc1_scores[i]) for i in order[-n_extreme:]]

    print("\n=== SECCIÓN 1.1 — Correspondencia de extremos topológicos ===")
    print(f"\nTop {n_extreme} países (PC1 más alto):")
    for name, coord, score in reversed(top):
        print(f"  {name:<22} score={score:+.3f}  pos=({coord[0]},{coord[1]})")

    print(f"\nBottom {n_extreme} países (PC1 más bajo):")
    for name, coord, score in bottom:
        print(f"  {name:<22} score={score:+.3f}  pos=({coord[0]},{coord[1]})")

    print("\nDistancias en grilla (top ↔ bottom):")
    for t_name, t_coord, _ in top:
        for b_name, b_coord, _ in bottom:
            d = float(np.linalg.norm(np.array(t_coord) - np.array(b_coord)))
            print(f"  {t_name} ↔ {b_name}: {d:.2f}")


# ---------------------------------------------------------------------------
# Section 2.1: Middle countries
# ---------------------------------------------------------------------------

def analyze_middle(
    countries: list[str],
    som_coords: np.ndarray,
    pc1_scores: np.ndarray,
    threshold: float = 0.5,
) -> None:
    middle = [
        (countries[i], som_coords[i], pc1_scores[i])
        for i in range(len(countries)) if abs(pc1_scores[i]) < threshold
    ]
    print(f"\n=== SECCIÓN 2.1 — Países 'promedio' (|PC1| < {threshold}) ===")
    if not middle:
        print(f"  (ningún país dentro del umbral; considera aumentar el threshold)")
        return
    for name, coord, score in sorted(middle, key=lambda x: x[2]):
        print(f"  {name:<22} score={score:+.3f}  pos=({coord[0]},{coord[1]})")

    coords_only = [c for _, c, _ in middle]
    if len(coords_only) > 1:
        dists = []
        for i in range(len(coords_only)):
            for j in range(i + 1, len(coords_only)):
                d = float(np.linalg.norm(np.array(coords_only[i]) - np.array(coords_only[j])))
                dists.append(d)
        print(f"\n  Distancia media entre países promedio en la grilla: {np.mean(dists):.2f}")
        print(f"  Distancia máxima:  {max(dists):.2f}  (Kohonen los separa si son distintos)")


# ---------------------------------------------------------------------------
# Plots — Section 1.1
# ---------------------------------------------------------------------------

_RED   = "#B2182B"
_BLUE  = "#2166AC"
_RED_L = "#FDDBC7"
_BLUE_L = "#D1E5F0"
_GRAY  = "#CCCCCC"
_GRAY_L = "#F5F5F5"


def plot_pc1_ranking(
    countries: list[str],
    pc1_scores: np.ndarray,
    n_top: int,
    n_bottom: int,
    path: str,
) -> None:
    """Horizontal bar chart of all countries sorted by PC1 score.
    Top-n_top highlighted in red, bottom-n_bottom in blue."""
    order = np.argsort(pc1_scores)
    sorted_countries = [countries[i] for i in order]
    sorted_scores    = pc1_scores[order]
    n = len(countries)

    colors = [
        _BLUE  if i < n_bottom else
        _RED   if i >= n - n_top else
        "#AAAAAA"
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(sorted_countries, sorted_scores, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Score sobre PC1")
    ax.set_title("PCA — Ranking de países por score PC1\n"
                 "(datos estandarizados, sklearn)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    for i, score in enumerate(sorted_scores):
        if i < n_bottom or i >= n - n_top:
            offset = 0.07 if score >= 0 else -0.07
            ax.text(score + offset, i, f"{score:+.2f}",
                    va="center", ha="left" if score >= 0 else "right", fontsize=8)

    legend = [
        mpatches.Patch(facecolor=_RED,  label=f"Top {n_top} — PC1 alto (mayor desarrollo)"),
        mpatches.Patch(facecolor=_BLUE, label=f"Bottom {n_bottom} — PC1 bajo (menor desarrollo)"),
        mpatches.Patch(facecolor="#AAAAAA", label="Resto"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kohonen_extremes(
    countries: list[str],
    som_coords: np.ndarray,
    pc1_scores: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    n_top: int,
    n_bottom: int,
    path: str,
) -> None:
    """Kohonen grid with top-n_top and bottom-n_bottom PC1 countries highlighted.
    Same color scheme as plot_pc1_ranking so both plots read as a pair."""
    order = np.argsort(pc1_scores)
    bottom_set = {countries[i] for i in order[:n_bottom]}
    top_set    = {countries[i] for i in order[-n_top:]}

    assignments: dict[tuple, list[str]] = {}
    for country, (r, c) in zip(countries, som_coords):
        assignments.setdefault((int(r), int(c)), []).append(country)

    fig, ax = plt.subplots(figsize=(grid_cols * 2.4, grid_rows * 2.4))
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_aspect("equal")

    for r in range(grid_rows):
        for c in range(grid_cols):
            y = r
            cell = assignments.get((r, c), [])
            has_top    = any(ct in top_set    for ct in cell)
            has_bottom = any(ct in bottom_set for ct in cell)

            if has_top:
                face, edge, lw = _RED_L,  _RED,  2.5
            elif has_bottom:
                face, edge, lw = _BLUE_L, _BLUE, 2.5
            else:
                face, edge, lw = _GRAY_L, _GRAY, 0.8

            ax.add_patch(plt.Rectangle((c, y), 1, 1,
                                       facecolor=face, edgecolor=edge, linewidth=lw))

            if cell:
                lines = []
                for ct in cell:
                    if ct in top_set:
                        lines.append(f"▲ {ct}")
                    elif ct in bottom_set:
                        lines.append(f"▼ {ct}")
                    else:
                        lines.append(ct)
                ax.text(c + 0.5, y + 0.5, "\n".join(lines),
                        ha="center", va="center", fontsize=15)

    ax.set_xticks(np.arange(grid_cols) + 0.5)
    ax.set_xticklabels(range(grid_cols), fontsize=14)
    ax.set_yticks(np.arange(grid_rows) + 0.5)
    ax.set_yticklabels(range(grid_rows), fontsize=14)
    ax.set_title(
        "Kohonen — posición de los extremos de PC1\n"
        f"▲ Top {n_top} (fondo rojo)  |  ▼ Bottom {n_bottom} (fondo azul)",
        fontsize=16,
    )
    legend = [
        mpatches.Patch(facecolor=_RED_L,  edgecolor=_RED,  linewidth=2,
                       label=f"Top {n_top} PC1"),
        mpatches.Patch(facecolor=_BLUE_L, edgecolor=_BLUE, linewidth=2,
                       label=f"Bottom {n_bottom} PC1"),
        mpatches.Patch(facecolor=_GRAY_L, edgecolor=_GRAY,
                       label="Resto"),
    ]
    ax.legend(handles=legend, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — Section 2.1
# ---------------------------------------------------------------------------

def plot_pc1_gradient_map(
    countries: list[str],
    som_coords: np.ndarray,
    pc1_scores: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    path: str,
) -> None:
    """Kohonen grid colored by average PC1 score per cell.
    The gradient reveals the intermediate 'gray' band where PCA-similar
    countries may occupy distant positions in the topology."""
    cell_scores: dict[tuple, list[float]] = {}
    cell_countries: dict[tuple, list[str]] = {}
    for country, coord, score in zip(countries, som_coords, pc1_scores):
        key = (int(coord[0]), int(coord[1]))
        cell_scores.setdefault(key, []).append(float(score))
        cell_countries.setdefault(key, []).append(country)

    avg_score = {k: float(np.mean(v)) for k, v in cell_scores.items()}
    vmin, vmax = min(avg_score.values()), max(avg_score.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdBu_r

    fig, ax = plt.subplots(figsize=(grid_cols * 2.4, grid_rows * 2.4))
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_aspect("equal")

    for r in range(grid_rows):
        for c in range(grid_cols):
            y = r
            if (r, c) in avg_score:
                color = cmap(norm(avg_score[(r, c)]))
                ax.add_patch(plt.Rectangle((c, y), 1, 1,
                                           facecolor=color, edgecolor="gray", linewidth=0.7))
                ax.text(c + 0.5, y + 0.5, "\n".join(cell_countries[(r, c)]),
                        ha="center", va="center", fontsize=15)
            else:
                ax.add_patch(plt.Rectangle((c, y), 1, 1,
                                           facecolor="#e8e8e8", edgecolor="gray", linewidth=0.5))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Score PC1 promedio", fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Score PC1 promedio", fontsize=15)
    ax.set_xticks(np.arange(grid_cols) + 0.5)
    ax.set_xticklabels(range(grid_cols), fontsize=14)
    ax.set_yticks(np.arange(grid_rows) + 0.5)
    ax.set_yticklabels(range(grid_rows), fontsize=14)
    ax.set_title("Sección 2.1 — Mapa de Kohonen coloreado por score PC1", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — Section 1.2
# ---------------------------------------------------------------------------

def plot_component_planes_ranked(
    som: SOM,
    feature_names: list[str],
    loadings: np.ndarray,
    path: str,
) -> None:
    """Kohonen component planes sorted by PCA loading value (high → low).
    Variables with the same loading sign should show the same spatial pattern."""
    order = np.argsort(-loadings)
    n = len(feature_names)
    gr, gc = som.grid_rows, som.grid_cols
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.2))
    axes = np.array(axes).ravel()

    for plot_idx, feat_idx in enumerate(order):
        ax = axes[plot_idx]
        plane = som.weights[:, :, feat_idx]
        im = ax.imshow(plane, cmap="RdBu_r", interpolation="nearest", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Peso de la neurona")

        loading_val = float(loadings[feat_idx])
        sign = "+" if loading_val >= 0 else ""
        title_color = _RED if loading_val > 0 else _BLUE

        ax.set_title(
            f"{feature_names[feat_idx]}\n"
            f"Loading PC1 (PCA): {sign}{loading_val:.3f}",
            fontsize=9, color=title_color, weight="bold",
        )
        # label axes so it's obvious these are the Kohonen grid coordinates
        ax.set_xticks(range(gc))
        ax.set_yticks(range(gr))
        ax.set_xticklabels(range(gc), fontsize=6)
        ax.set_yticklabels(range(gr), fontsize=6)
        ax.set_xlabel("col Kohonen", fontsize=6)
        ax.set_ylabel("fila Kohonen", fontsize=6)

    for j in range(len(order), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Sección 1.2 — Planos de Componentes (Red de Kohonen)\n"
        "Peso aprendido por neurona para cada variable · ordenados por loading PC1 (PCA)\n"
        "Títulos rojos = loading positivo · títulos azules = loading negativo",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_plane_correlations(
    som: SOM,
    feature_names: list[str],
    loadings: np.ndarray,
    path: str,
) -> None:
    """
    Left: Pearson correlation between each pair of SOM component planes.
    Right: Expected sign pattern from PCA loadings (outer product).
    If both matrices agree, Kohonen's topology reflects PCA's linear structure.
    """
    planes = [som.weights[:, :, i].ravel() for i in range(len(feature_names))]
    plane_corr = np.corrcoef(planes)

    loading_outer = np.outer(loadings, loadings)
    lo_norm = loading_outer / max(np.max(np.abs(loading_outer)), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mat, title in zip(
        axes,
        [plane_corr, lo_norm],
        [
            "Correlación entre planos\nde componentes Kohonen",
            "Correlación esperada\nsegún loadings PCA (prod. exterior)",
        ],
    ):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        n = len(feature_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(mat[i, j]) > 0.7 else "black")

    fig.suptitle(
        "Sección 1.2 — Coherencia en la correlación de variables\n"
        "(si ambas matrices coinciden en signo, Kohonen refleja la misma estructura lineal que PCA)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Comparativa PCA vs Kohonen")
    parser.add_argument("--kohonen-config", default="configs/kohonen_europe.json")
    parser.add_argument("--output-dir", default="output/compare")
    parser.add_argument("--middle-threshold", type=float, default=0.5,
                        help="Umbral |PC1| para clasificar países como 'promedio'")
    args = parser.parse_args()

    with open(args.kohonen_config) as f:
        som_cfg = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    countries, X, feature_names = load_data(som_cfg["data"])

    print("Entrenando SOM...")
    som = train_som(X, som_cfg)
    som_coords = som.predict(X)

    loading, pc1_scores = get_pc1(X)

    analyze_extremes(countries, som_coords, pc1_scores)
    analyze_middle(countries, som_coords, pc1_scores, threshold=args.middle_threshold)

    n_top, n_bottom = 5, 4
    out = args.output_dir
    print("\nGenerando gráficos...")

    # Section 1.1
    plot_pc1_ranking(countries, pc1_scores, n_top, n_bottom,
                     os.path.join(out, "1.1_pc1_ranking.png"))
    plot_kohonen_extremes(countries, som_coords, pc1_scores,
                          som_cfg["grid_rows"], som_cfg["grid_cols"], n_top, n_bottom,
                          os.path.join(out, "1.1_kohonen_extremes.png"))

    # Section 1.2
    plot_component_planes_ranked(som, feature_names, loading,
                                 os.path.join(out, "1.2_component_planes.png"))
    plot_plane_correlations(som, feature_names, loading,
                            os.path.join(out, "1.2_plane_correlations.png"))

    # Section 2.1
    plot_pc1_gradient_map(countries, som_coords, pc1_scores,
                          som_cfg["grid_rows"], som_cfg["grid_cols"],
                          os.path.join(out, "2.1_pc1_gradient_map.png"))

    print(f"\nGráficos guardados en {out}/")


if __name__ == "__main__":
    main()
