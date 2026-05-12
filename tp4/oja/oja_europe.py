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

from oja import OjaNetwork


def load_data(path: str) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(path)
    countries = df["Country"].tolist()
    feature_df = df.drop(columns=["Country"])
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    return countries, X_scaled, feature_names


def run(cfg: dict, X: np.ndarray) -> OjaNetwork:
    net = OjaNetwork(
        input_dim=X.shape[1],
        lr=cfg["lr"],
        epochs=cfg["epochs"],
        seed=cfg["seed"],
    )
    net.train(X)
    return net


def compute_sklearn_pc1(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X).ravel()
    loading = pca.components_[0]
    explained = float(pca.explained_variance_ratio_[0])
    return loading, scores, explained


def align_sign(oja_w: np.ndarray, sk_w: np.ndarray) -> np.ndarray:
    return oja_w if np.dot(oja_w, sk_w) >= 0 else -oja_w


def plot_loadings(
    feature_names: list[str],
    oja_w: np.ndarray,
    sk_w: np.ndarray,
    path: str,
) -> None:
    x = np.arange(len(feature_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, oja_w, width, label="Oja", color="steelblue")
    ax.bar(x + width / 2, sk_w, width, label="sklearn PCA", color="darkorange")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_ylabel("Peso (componente normalizada)")
    ax.set_title("Loadings de PC1 — Oja vs sklearn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_country_scores(
    countries: list[str],
    scores: np.ndarray,
    path: str,
) -> None:
    order = np.argsort(scores)
    sorted_countries = [countries[i] for i in order]
    sorted_scores = scores[order]
    colors = ["steelblue" if s < 0 else "indianred" for s in sorted_scores]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(sorted_countries, sorted_scores, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Score sobre PC1 (Oja)")
    ax.set_title("Países proyectados sobre la primer componente — Oja")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(
    history: list[np.ndarray],
    sk_w: np.ndarray,
    path: str,
) -> None:
    history = [h / np.linalg.norm(h) for h in history]
    history = [h if np.dot(h, sk_w) >= 0 else -h for h in history]
    diffs = [float(np.linalg.norm(h - sk_w)) for h in history]
    cosines = [float(np.dot(h, sk_w)) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(diffs, color="steelblue")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("‖w_oja − w_sklearn‖")
    ax1.set_title("Distancia al autovector de sklearn")
    ax1.grid(True, alpha=0.3)

    ax2.plot(cosines, color="darkgreen")
    ax2.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("cos(w_oja, w_sklearn)")
    ax2.set_title("Similitud coseno con sklearn")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_comparison(
    feature_names: list[str],
    oja_w: np.ndarray,
    sk_w: np.ndarray,
    oja_scores: np.ndarray,
    sk_scores: np.ndarray,
    explained: float,
) -> None:
    cos = float(np.dot(oja_w, sk_w))
    max_abs = float(np.max(np.abs(oja_w - sk_w)))
    corr = float(np.corrcoef(oja_scores, sk_scores)[0, 1])

    print("\nLoadings PC1 (alineados en signo):")
    print(f"{'Feature':<14} {'Oja':>10} {'sklearn':>10} {'|diff|':>10}")
    print("-" * 48)
    for name, o, s in zip(feature_names, oja_w, sk_w):
        print(f"{name:<14} {o:>10.4f} {s:>10.4f} {abs(o - s):>10.4f}")

    print("\nMétricas de comparación:")
    print(f"  cos(oja, sklearn)      = {cos:.6f}")
    print(f"  max |oja - sklearn|    = {max_abs:.6f}")
    print(f"  corr(oja_scores, sk_scores) = {corr:.6f}")
    print(f"  varianza explicada PC1 (sklearn) = {explained * 100:.2f}%")


def print_ranking(countries: list[str], scores: np.ndarray) -> None:
    order = np.argsort(-scores)
    print("\nRanking de países sobre PC1 (Oja):")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:>2}. {countries[idx]:<20} {scores[idx]:>+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Regla de Oja — Ejercicio Europa")
    parser.add_argument("--config", default="../configs/oja_europe.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    countries, X, feature_names = load_data(cfg["data"])
    net = run(cfg, X)
    oja_component = net.component()

    sk_w, sk_scores, explained = compute_sklearn_pc1(X)

    sign = 1.0 if np.dot(oja_component, sk_w) >= 0 else -1.0
    oja_w = sign * oja_component
    oja_scores = X @ oja_w

    print_comparison(feature_names, oja_w, sk_w, oja_scores, sk_scores, explained)
    print_ranking(countries, oja_scores)

    out = cfg["output_dir"]
    plot_loadings(feature_names, oja_w, sk_w, os.path.join(out, "loadings.png"))
    plot_country_scores(countries, oja_scores, os.path.join(out, "country_scores.png"))
    plot_convergence(net.history, sk_w, os.path.join(out, "convergence.png"))

    print(f"\nGráficos guardados en {out}/")


if __name__ == "__main__":
    main()
