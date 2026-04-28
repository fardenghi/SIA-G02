import ast
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "data and documentation"


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


def _load_csv(path):
    df = pd.read_csv(path)
    df["image"] = df["image"].apply(
        lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
    )
    X = np.stack(df["image"].values)
    y = df["label"].values.astype(int)
    return X, y


def load_digits(path=None):
    if path is None:
        path = _DATA_DIR / "digits.csv"
    return _load_csv(path)


def load_digits_test(path=None):
    if path is None:
        path = _DATA_DIR / "digits_test.csv"
    return _load_csv(path)


def load_more_digits(path=None):
    if path is None:
        path = _DATA_DIR / "more_digits.csv"
    return _load_csv(path)
