import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--n-components", type=int, default=1)
args = parser.parse_args()

df = pd.read_csv("data/europe.csv")

countries = df["Country"]
X = df.drop(columns=["Country"])
features = X.columns.tolist()

X_scaled = StandardScaler().fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)

pca = PCA(n_components=args.n_components)
pca.fit(X_scaled)
scores = pca.transform(X_scaled)

# --- stdout ---
col_w = 12
header = f"{'':15}" + "".join(f"{'PC' + str(i + 1):>{col_w}}" for i in range(args.n_components))
print("Autovectores:")
print(header)
for j, feature in enumerate(features):
    row = "".join(f"{pca.components_[i][j]:>{col_w}.6f}" for i in range(args.n_components))
    print(f"{feature:<15}{row}")

print()
print("Varianza explicada:")
print(f"{'':15}" + "".join(f"{'PC' + str(i + 1):>{col_w}}" for i in range(args.n_components)))
print(f"{'Autovalor':<15}" + "".join(f"{pca.explained_variance_[i]:>{col_w}.4f}" for i in range(args.n_components)))
print(f"{'Varianza (%)':<15}" + "".join(f"{pca.explained_variance_ratio_[i] * 100:>{col_w}.2f}" for i in range(args.n_components)))
print()

print("Scores por país:")
header = f"  {'País':<20}" + "".join(f"  {'PC' + str(i + 1):>8}" for i in range(args.n_components))
print(header)
for country, row in sorted(zip(countries, scores), key=lambda x: x[1][0], reverse=True):
    scores_str = "".join(f"  {v:+8.4f}" for v in row)
    print(f"  {country:<20}{scores_str}")

# --- plots ---
out_dir = os.path.join("output", str(args.n_components))
os.makedirs(out_dir, exist_ok=True)

n_all = len(pca_full.explained_variance_)
pc_labels = [f"PC{i + 1}" for i in range(n_all)]

# 1. Scree plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(pc_labels, pca_full.explained_variance_, color="steelblue", alpha=0.7, label="Autovalor")
ax2 = ax.twinx()
ax2.plot(pc_labels, np.cumsum(pca_full.explained_variance_ratio_) * 100, marker="o", color="tomato", label="Varianza acumulada (%)")
ax.set_xlabel("Componente")
ax.set_ylabel("Autovalor")
ax2.set_ylabel("Varianza acumulada (%)")
ax2.set_ylim(0, 105)
ax.set_title("Scree plot")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "scree_plot.png"), dpi=150)
plt.close(fig)

# 2. Biplot (PC1 vs PC2)
if args.n_components >= 2:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(scores[:, 0], scores[:, 1], color="steelblue", zorder=3)
    for i, country in enumerate(countries):
        ax.annotate(country, (scores[i, 0], scores[i, 1]), fontsize=7, ha="left", va="bottom")
    scale = max(abs(scores[:, :2]).max(), 1)
    loading_scale = scale * 0.8
    for j, feature in enumerate(features):
        lx, ly = pca.components_[0][j] * loading_scale, pca.components_[1][j] * loading_scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5))
        ax.text(lx * 1.08, ly * 1.08, feature, color="tomato", fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Biplot PC1 – PC2")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "biplot.png"), dpi=150)
    plt.close(fig)

# 3. Heatmap de loadings
fig, ax = plt.subplots(figsize=(max(5, args.n_components * 1.2), 5))
data = pca.components_.T
im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(args.n_components))
ax.set_xticklabels([f"PC{i + 1}" for i in range(args.n_components)])
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
for i in range(len(features)):
    for j in range(args.n_components):
        ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(data[i, j]) > 0.5 else "black")
plt.colorbar(im, ax=ax, label="Loading")
ax.set_title("Heatmap de loadings")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "heatmap_loadings.png"), dpi=150)
plt.close(fig)

# 4. Bar chart de loadings por componente
fig, axes = plt.subplots(1, args.n_components, figsize=(4 * args.n_components, 5), squeeze=False)
for i in range(args.n_components):
    ax = axes[0][i]
    vals = pca.components_[i]
    colors = ["steelblue" if v >= 0 else "tomato" for v in vals]
    ax.barh(features, vals, color=colors)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1, 1)
    ax.set_title(f"PC{i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}%)")
    ax.set_xlabel("Loading")
fig.suptitle("Loadings por componente")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "bar_loadings.png"), dpi=150)
plt.close(fig)

print(f"\nGráficos guardados en {out_dir}/")
