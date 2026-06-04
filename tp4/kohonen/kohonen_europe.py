import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kohonen.som import SOM


def load_data(path: str) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(path)
    countries = df["Country"].tolist()
    feature_df = df.drop(columns=["Country"])
    feature_names = feature_df.columns.tolist()
    X_scaled = StandardScaler().fit_transform(feature_df.to_numpy(dtype=float))
    return countries, X_scaled, feature_names


def run(cfg: dict, X: np.ndarray) -> tuple[SOM, np.ndarray]:
    som = SOM(
        grid_rows=cfg["grid_rows"],
        grid_cols=cfg["grid_cols"],
        input_dim=X.shape[1],
        lr=cfg["lr"],
        lr_decay=cfg["lr_decay"],
        radius=cfg["radius"],
        radius_decay=cfg["radius_decay"],
        neighborhood_fn=cfg["neighborhood_fn"],
        epochs=cfg["epochs"],
        seed=cfg["seed"],
    )
    som.train(X)
    coords = som.predict(X)
    return som, coords


def build_assignments(countries: list[str], coords: np.ndarray) -> dict[tuple, list[str]]:
    assignments: dict[tuple, list[str]] = {}
    for country, (r, c) in zip(countries, coords):
        key = (int(r), int(c))
        assignments.setdefault(key, []).append(country)
    return assignments


_COUNTRY_GROUPS = [
    {
        "label": "Desarrolladas",
        "countries": {"Luxembourg", "Ireland", "Iceland"},
        "color": "#FFBFBF",  # pastel red
    },
    {
        "label": "Centroeuropa (Desarrollada)",
        "countries": {"Czech Republic", "Slovenia", "Austria", "Denmark"},
        "color": "#FFD9B3",  # pastel orange
    },
    {
        "label": "Centroeuropa (Transición)",
        "countries": {"Belgium", "Hungary", "Lithuania", "Slovakia", "Latvia"},
        "color": "#FFF5B3",  # pastel yellow
    },
    {
        "label": "Occidente/Nórdica Próspera",
        "countries": {"Netherlands", "Switzerland", "Finland", "Norway", "Germany", "Sweden", "Italy"},
        "color": "#B3D4F5",  # pastel blue
    },
    {
        "label": "Este Emergente",
        "countries": {"Ukraine", "Bulgaria", "Estonia", "Poland"},
        "color": "#C2EBC2",  # pastel green
    },
]
_DEFAULT_CELL_COLOR = "#E8E8E8"


def _cell_color(countries_in_cell: list[str]) -> str:
    counts: dict[int, int] = {}
    for country in countries_in_cell:
        for i, g in enumerate(_COUNTRY_GROUPS):
            if country in g["countries"]:
                counts[i] = counts.get(i, 0) + 1
                break
    if not counts:
        return _DEFAULT_CELL_COLOR
    # tie-break: prefer the group with lowest index (order in _COUNTRY_GROUPS)
    return _COUNTRY_GROUPS[min(counts, key=lambda i: (-counts[i], i))]["color"]


def plot_country_map(
    assignments: dict[tuple, list[str]],
    grid_rows: int,
    grid_cols: int,
    path: str,
) -> None:
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(grid_cols * 2, grid_rows * 2))
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.grid(True, color="gray", linewidth=0.5, zorder=1)
    ax.set_xticklabels([str(c) for c in range(grid_cols)])
    ax.set_yticklabels([str(r) for r in range(grid_rows)])
    ax.tick_params(axis="both", labelsize=14)

    for (r, c), countries in assignments.items():
        color = _cell_color(countries)
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="none", zorder=0))
        ax.text(
            c + 0.5, r + 0.5, "\n".join(countries),
            ha="center", va="center", fontsize=15, zorder=2,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none", alpha=0.85),
        )

    ax.set_title("Mapa de países — Red de Kohonen", fontsize=18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_u_matrix(som: SOM, path: str) -> None:
    u = som.u_matrix()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(u, cmap="bone_r", interpolation="nearest", origin="lower")
    plt.colorbar(im, ax=ax, label="Distancia promedio")
    ax.set_title("U-Matrix — distancias entre neuronas vecinas")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    for r in range(som.grid_rows):
        for c in range(som.grid_cols):
            ax.text(c, r, f"{u[r, c]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if u[r, c] > u.mean() else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hit_map(
    assignments: dict[tuple, list[str]],
    grid_rows: int,
    grid_cols: int,
    n_total: int,
    path: str,
) -> None:
    counts = np.zeros((grid_rows, grid_cols), dtype=int)
    for (r, c), countries in assignments.items():
        counts[r, c] = len(countries)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(counts, cmap="YlOrRd", interpolation="nearest", origin="lower")
    plt.colorbar(im, ax=ax, label="Cantidad de países")
    ax.set_title(f"Hit map — países por neurona (total: {n_total})")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    for r in range(grid_rows):
        for c in range(grid_cols):
            ax.text(c, r, str(counts[r, c]), ha="center", va="center", fontsize=9,
                    color="white" if counts[r, c] > counts.mean() else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_component_planes(som: SOM, feature_names: list[str], path: str) -> None:
    n = len(feature_names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8))
    axes = np.array(axes).ravel()
    for i, name in enumerate(feature_names):
        plane = som.weights[:, :, i]
        im = axes[i].imshow(plane, cmap="RdBu_r", interpolation="nearest", origin="lower")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].set_title(name, fontsize=9)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Planos de Componentes — Red de Kohonen", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_assignments(assignments: dict[tuple, list[str]], grid_rows: int, grid_cols: int) -> None:
    print("\nAsignación de países por neurona:")
    print(f"{'Neurona':<12} {'Países'}")
    print("-" * 50)
    for r in range(grid_rows):
        for c in range(grid_cols):
            key = (r, c)
            countries = assignments.get(key, [])
            if countries:
                print(f"  ({r},{c})      {', '.join(countries)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Red de Kohonen — Ejercicio Europa")
    parser.add_argument("--config", default="configs/kohonen_europe.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    countries, X, feature_names = load_data(cfg["data"])
    som, coords = run(cfg, X)
    assignments = build_assignments(countries, coords)
    print_assignments(assignments, cfg["grid_rows"], cfg["grid_cols"])

    print("Conteo de países por neurona:")
    for r in range(cfg["grid_rows"]):
        row_str = ""
        for c in range(cfg["grid_cols"]):
            n = len(assignments.get((r, c), []))
            row_str += f"{n:3}"
        print(row_str)

    out = cfg["output_dir"]
    plot_country_map(assignments, cfg["grid_rows"], cfg["grid_cols"],
                     os.path.join(out, "country_map.png"))
    plot_u_matrix(som, os.path.join(out, "u_matrix.png"))
    plot_hit_map(assignments, cfg["grid_rows"], cfg["grid_cols"],
                 len(countries), os.path.join(out, "hit_map.png"))
    plot_component_planes(som, feature_names, os.path.join(out, "component_planes.png"))

    print(f"\nGráficos guardados en {out}/")


if __name__ == "__main__":
    main()
