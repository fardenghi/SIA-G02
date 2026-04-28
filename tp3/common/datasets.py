import ast
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DIGITS_DIR = DATA_DIR / "ej2_digits"
MORE_DIGITS_DIR = DATA_DIR / "ej3_more_digits"


def xor_dataset():
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([-1.0, 1.0, 1.0, -1.0])
    return X, y


def to_one_hot(y, n_classes, encoding="zero_one"):
    y = np.asarray(y, dtype=int)
    N = len(y)
    Y = np.zeros((N, n_classes), dtype=float)
    Y[np.arange(N), y] = 1.0
    if encoding == "signed":
        Y = 2.0 * Y - 1.0
    return Y


def load_digit_frame(path):
    """Load a digit CSV and deserialize the image column to numpy arrays."""
    df = pd.read_csv(path)
    df["image"] = df["image"].apply(
        lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
    )
    return df


def digit_image(row, size=(28, 28)):
    """Return one deserialized digit row as a 2-D image."""
    return row["image"].reshape(size)


def _load_digit_csv(path):
    df = load_digit_frame(path)
    X = np.stack(df["image"].values)
    y = df["label"].values.astype(int)
    return X, y


def load_digits(path=None):
    if path is None:
        path = DIGITS_DIR / "digits.csv"
    return _load_digit_csv(path)


def load_digits_test(path=None):
    if path is None:
        path = DIGITS_DIR / "digits_test.csv"
    return _load_digit_csv(path)


def load_more_digits(path=None):
    if path is None:
        path = MORE_DIGITS_DIR / "more_digits.csv"
    return _load_digit_csv(path)
