from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "europe.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "europe_standardized.csv"


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    standardized = (numeric - numeric.mean()) / numeric.std(ddof=0)
    non_numeric = df.select_dtypes(exclude=[np.number])
    return pd.concat([non_numeric, standardized], axis=1)[df.columns]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    standardized = standardize(df)
    standardized.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved standardized data to {OUTPUT_PATH}")
    print(standardized.head())


if __name__ == "__main__":
    main()
