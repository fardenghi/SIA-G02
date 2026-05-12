from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pca_test.standardize import standardize

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "europe.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def run_pca(df: pd.DataFrame) -> tuple[PCA, np.ndarray, list[str]]:
    standardized = standardize(df)
    numeric = standardized.select_dtypes(include=[np.number])
    countries = df["Country"].tolist()

    pca = PCA()
    components = pca.fit_transform(numeric.values)
    return pca, components, countries


def print_results(pca: PCA, feature_names: list[str]) -> None:
    explained = pca.explained_variance_ratio_
    print("Varianza explicada por componente:")
    for i, var in enumerate(explained):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%) | acumulada: {explained[:i+1].sum()*100:.2f}%")

    print("\nLoadings (pesos de cada variable por componente):")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(len(explained))],
    )
    print(loadings.round(4).to_string())


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    pca, components, countries = run_pca(df)
    print_results(pca, numeric_cols)

    scores = pd.DataFrame(
        components,
        index=countries,
        columns=[f"PC{i+1}" for i in range(components.shape[1])],
    )
    scores.to_csv(Path(__file__).resolve().parent / "pca_scores.csv")
    print(f"\nScores guardados en pca_test/pca_scores.csv")


if __name__ == "__main__":
    main()
