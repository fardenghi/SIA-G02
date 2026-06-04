from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pca_test.standardize import standardize

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "europe.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def boxplot(df: pd.DataFrame, title: str, output_path: Path) -> None:
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [numeric[col].values for col in numeric.columns],
        tick_labels=numeric.columns,
        patch_artist=True,
    )
    ax.set_title(title)
    ax.set_ylabel("Valor")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    boxplot(df, "Boxplot antes de estandarizar", OUTPUT_DIR / "boxplot_raw.png")
    boxplot(
        standardize(df),
        "Boxplot despues de estandarizar (z-score)",
        OUTPUT_DIR / "boxplot_standardized.png",
    )


if __name__ == "__main__":
    main()
