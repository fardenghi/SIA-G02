"""Cuenta muestras por dígito en un dataset de dígitos.

Ejemplos:
    uv run python -m exercises.ej2_y_3_count_muestras
    uv run python -m exercises.ej2_y_3_count_muestras --dataset data/ej2_digits/digits_test.csv
    uv run python -m exercises.ej2_y_3_count_muestras --dataset data/ej3_more_digits/more_digits.csv
"""

import argparse
from pathlib import Path

import numpy as np

from common.datasets import DIGITS_DIR, load_digit_frame

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET = DIGITS_DIR / "digits.csv"
_N_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Cuenta muestras por dígito en un dataset.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET,
                        help="Path al archivo .csv del dataset.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset no encontrado: {args.dataset}")

    df = load_digit_frame(args.dataset)
    y = df["label"].values.astype(int)
    total = len(y)

    print(f"Dataset: {args.dataset}")
    print(f"Total muestras: {total}")
    print("─" * 42)
    print(f"{'Dígito':>7}  {'Muestras':>9}  {'% del total':>11}")
    print("─" * 42)

    for c in range(_N_CLASSES):
        count = int((y == c).sum())
        pct = count / total * 100 if total > 0 else 0.0
        warning = "  ← SIN MUESTRAS" if count == 0 else ""
        print(f"{c:>7}  {count:>9}  {pct:>10.1f}%{warning}")

    print("─" * 42)


if __name__ == "__main__":
    main()
