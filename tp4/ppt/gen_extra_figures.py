"""Genera figuras extra para la presentación TP4 (scatter Oja vs sklearn)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from oja.oja import OjaNetwork


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(HERE, "..", "data", "europe.csv"))
    countries = df["Country"].tolist()
    feat_df = df.drop(columns=["Country"])
    X = StandardScaler().fit_transform(feat_df.to_numpy(dtype=float))
    return countries, X, feat_df.columns.tolist()


def main():
    countries, X, _ = load_data()

    with open(os.path.join(HERE, "..", "configs", "oja_europe.json")) as f:
        cfg = json.load(f)

    net = OjaNetwork(input_dim=X.shape[1], lr=cfg["lr"], epochs=cfg["epochs"], seed=cfg["seed"])
    net.train(X)
    oja_w = net.component()

    pca = PCA(n_components=1)
    sk_scores = pca.fit_transform(X).ravel()
    sk_w = pca.components_[0]

    if np.dot(oja_w, sk_w) < 0:
        oja_w = -oja_w
    oja_scores = X @ oja_w

    cos = float(np.dot(oja_w, sk_w))
    corr = float(np.corrcoef(oja_scores, sk_scores)[0, 1])
    explained = float(pca.explained_variance_ratio_[0])

    # Scatter Oja vs sklearn
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    lims = [min(sk_scores.min(), oja_scores.min()) - 0.3,
            max(sk_scores.max(), oja_scores.max()) + 0.3]
    ax.plot(lims, lims, color="#5eead4", lw=1.2, linestyle="--", alpha=0.7, label="y = x")
    ax.scatter(sk_scores, oja_scores, s=70, color="#c084fc", edgecolor="white", linewidth=0.5, alpha=0.9)
    for c, sx, sy in zip(countries, sk_scores, oja_scores):
        ax.annotate(c, (sx, sy), fontsize=6, color="#cbd5e1", xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Score PCA (sklearn)", color="#e5e7eb", fontsize=11)
    ax.set_ylabel("Score Oja", color="#e5e7eb", fontsize=11)
    ax.set_title(f"Oja vs sklearn — corr={corr:.4f}, cos(w)={cos:.4f}",
                 color="#f9fafb", fontsize=12, pad=12)
    ax.tick_params(colors="#9ca3af")
    for sp in ax.spines.values():
        sp.set_color("#374151")
    ax.grid(True, alpha=0.15, color="#6b7280")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oja_vs_sklearn_scatter.png"), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Resumen numérico para usar en slide
    summary = {
        "cos_oja_sklearn": cos,
        "corr_scores": corr,
        "explained_pc1": explained,
        "oja_loading": oja_w.tolist(),
        "sk_loading": sk_w.tolist(),
    }
    with open(os.path.join(OUT, "oja_vs_sklearn_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"OK — scatter en {OUT}/oja_vs_sklearn_scatter.png")
    print(f"cos={cos:.4f} corr={corr:.4f} explained={explained:.4f}")


if __name__ == "__main__":
    main()
