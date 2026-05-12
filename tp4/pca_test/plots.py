from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pca_test.pca import run_pca

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "europe.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def plot_variance(pca, output_dir: Path) -> None:
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    labels = [f"PC{i+1}" for i in range(len(explained))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, explained * 100, color="steelblue", alpha=0.8, label="Varianza explicada")
    ax.plot(labels, cumulative * 100, color="tomato", marker="o", linewidth=2, label="Varianza acumulada")

    for i, (v, c) in enumerate(zip(explained, cumulative)):
        ax.text(i, v * 100 + 0.8, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i, c * 100 + 0.8, f"{c*100:.1f}%", ha="center", va="bottom", fontsize=8, color="tomato")

    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_title("Varianza explicada por componente (PCA)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "variance_explained.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_pc1_vs_pc2(components: np.ndarray, countries: list[str], pca, feature_names: list[str], output_dir: Path) -> None:
    var = pca.explained_variance_ratio_
    loadings = pca.components_[:2].T  # shape (n_features, 2)

    # scale arrows to span ~same range as scores
    scale = np.max(np.abs(components[:, :2])) / np.max(np.abs(loadings))

    fig, ax = plt.subplots(figsize=(11, 8))

    ax.scatter(components[:, 0], components[:, 1], color="steelblue", zorder=3)
    for i, country in enumerate(countries):
        ax.annotate(country, (components[i, 0], components[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    for i, feature in enumerate(feature_names):
        dx, dy = loadings[i, 0] * scale, loadings[i, 1] * scale
        ax.arrow(0, 0, dx, dy, head_width=0.12, head_length=0.08,
                 fc="tomato", ec="tomato", linewidth=1.5, zorder=4)
        ax.text(dx * 1.1, dy * 1.1, feature, color="tomato", fontsize=9, fontweight="bold", ha="center")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% varianza)")
    ax.set_title("Biplot PC1 vs PC2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "pc1_vs_pc2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_pc1_ranking(components: np.ndarray, countries: list[str], pca, output_dir: Path) -> None:
    pc1 = components[:, 0]
    order = np.argsort(pc1)
    sorted_countries = [countries[i] for i in order]
    sorted_values = pc1[order]
    colors = ["tomato" if v < 0 else "steelblue" for v in sorted_values]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(sorted_countries, sorted_values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    ax.set_title("Ranking de países por PC1\n(desarrollo económico y calidad de vida)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "pc1_ranking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    pca, components, countries = run_pca(df)

    feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_variance(pca, OUTPUT_DIR)
    plot_pc1_vs_pc2(components, countries, pca, feature_names, OUTPUT_DIR)
    plot_pc1_ranking(components, countries, pca, OUTPUT_DIR)


if __name__ == "__main__":
    main()
