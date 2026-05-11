import argparse
import pandas as pd
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

pca = PCA(n_components=args.n_components)
pca.fit(X_scaled)
scores = pca.transform(X_scaled)

for i in range(args.n_components):
    print(f"PC{i + 1}:")
    print(f"  Varianza explicada: {pca.explained_variance_ratio_[i] * 100:.2f}%")
    print(f"  Autovalor:          {pca.explained_variance_[i]:.4f}")
    print(f"  Loadings:")
    for feature, weight in sorted(zip(features, pca.components_[i]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feature:<15} {weight:+.4f}")
    print()

print("Scores por país:")
header = f"  {'País':<20}" + "".join(f"  {'PC' + str(i + 1):>8}" for i in range(args.n_components))
print(header)
for country, row in sorted(zip(countries, scores), key=lambda x: x[1][0], reverse=True):
    scores_str = "".join(f"  {v:+8.4f}" for v in row)
    print(f"  {country:<20}{scores_str}")
