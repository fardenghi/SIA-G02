"""Inspect one digit sample as a 28x28 image.

Examples:
    uv run python -m exercises.ej2_digits.inspect_sample
    uv run python -m exercises.ej2_digits.inspect_sample --dataset train --index 10
    uv run python -m exercises.ej2_digits.inspect_sample --dataset more --index 20 --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from common.datasets import DIGITS_DIR, MORE_DIGITS_DIR, digit_image, load_digit_frame

_ROOT = Path(__file__).resolve().parents[2]
_DATASETS = {
    "train": DIGITS_DIR / "digits.csv",
    "test": DIGITS_DIR / "digits_test.csv",
    "more": MORE_DIGITS_DIR / "more_digits.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render one digit CSV sample.")
    parser.add_argument(
        "--dataset",
        choices=sorted(_DATASETS),
        default="test",
        help="Dataset to inspect: train, test, or more.",
    )
    parser.add_argument("--index", type=int, default=0, help="Row index to render.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG destination. Defaults to outputs/ej2_digits/samples/.",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot window.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_digit_frame(_DATASETS[args.dataset])
    if args.index < 0 or args.index >= len(df):
        raise SystemExit(f"index must be between 0 and {len(df) - 1}")

    row = df.iloc[args.index]
    label = int(row["label"])
    image = digit_image(row)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{args.dataset}[{args.index}] label={label}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    output = args.output
    if output is None:
        output = _ROOT / "outputs" / "ej2_digits" / "samples" / f"{args.dataset}_{args.index}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Sample saved to {output}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
