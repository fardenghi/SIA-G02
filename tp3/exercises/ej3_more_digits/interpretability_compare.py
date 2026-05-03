"""Comparativa de atribución entre dos modelos sobre el mismo dígito.

Busca un ejemplo donde el modelo A predice correctamente y el B falla
(o donde ambos aciertan pero con saliency muy diferente), y dibuja una
única figura con dos filas (una por modelo) × 3 columnas.

Uso:
    uv run python -m exercises.ej3_more_digits.interpretability_compare \\
        outputs/ej3_more_digits/models/arch_wide.npz \\
        outputs/ej3_more_digits/models/baseline_pure.npz
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test
from common.mlp import MLP

from exercises.ej3_more_digits.interpretability import (
    IMG_SIZE, occlusion_map, saliency_map, softmax_prob,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "ej3_more_digits" / "interpretability"


def find_disagreement(X, y, mlp_a, mlp_b, target_class=None):
    """Busca un idx donde A acierta y B falla.

    Si target_class está dado, restringe a esa clase real. Devuelve el idx
    con mayor confianza combinada (A en lo correcto, B en lo equivocado)
    para que el contraste sea visible.
    """
    pa = softmax_prob(mlp_a, X)
    pb = softmax_prob(mlp_b, X)
    pred_a = pa.argmax(axis=1)
    pred_b = pb.argmax(axis=1)

    mask = (pred_a == y) & (pred_b != y)
    if target_class is not None:
        mask = mask & (y == target_class)
    if not mask.any():
        return None

    pool = np.where(mask)[0]
    score = pa[pool, y[pool]] * pb[pool, pred_b[pool]]
    return int(pool[np.argmax(score)])


def render_compare(x, y_true, mlp_a, mlp_b, names, patch, stride, out_path):
    img = x.reshape(IMG_SIZE, IMG_SIZE)
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2))

    for row, (mlp, name) in enumerate([(mlp_a, names[0]), (mlp_b, names[1])]):
        probs = softmax_prob(mlp, x.reshape(1, -1))[0]
        pred = int(probs.argmax())
        target = pred

        sal = saliency_map(mlp, x, target)
        sal_abs = np.abs(sal).reshape(IMG_SIZE, IMG_SIZE)
        if sal_abs.max() > 0:
            sal_abs /= sal_abs.max()

        occ = occlusion_map(mlp, x, target, patch=patch, stride=stride)
        peak = np.abs(occ).max()
        occ_norm = occ / peak if peak > 0 else occ

        ax_img, ax_sal, ax_occ = axes[row]
        ax_img.imshow(img, cmap="gray")
        correct = "OK" if pred == y_true else "FAIL"
        ax_img.set_title(
            f"{name}\ny={y_true}  pred={pred} ({probs[pred]:.2f})  [{correct}]",
            fontsize=9,
        )
        ax_img.axis("off")

        ax_sal.imshow(img, cmap="gray", alpha=0.5)
        ax_sal.imshow(sal_abs, cmap="hot", alpha=0.6)
        ax_sal.set_title(f"Saliency wrt clase {pred}", fontsize=9)
        ax_sal.axis("off")

        ax_occ.imshow(img, cmap="gray", alpha=0.5)
        ax_occ.imshow(occ_norm, cmap="seismic", vmin=-1, vmax=1, alpha=0.6)
        ax_occ.set_title(f"Occlusion ({patch}x{patch})", fontsize=9)
        ax_occ.axis("off")

    fig.suptitle(f"Comparativa de atribución — y_true = {y_true}",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_a", type=str)
    p.add_argument("model_b", type=str)
    p.add_argument("--target-class", type=int, default=None,
                   help="restringir a una clase real específica (0-9)")
    p.add_argument("--idx", type=int, default=None,
                   help="usar este idx específico del test set (override)")
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()

    pa = Path(args.model_a)
    pb = Path(args.model_b)
    for mp in (pa, pb):
        if not mp.exists():
            raise SystemExit(f"No existe {mp}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlp_a = MLP.load(str(pa))
    mlp_b = MLP.load(str(pb))
    X, y = load_digits_test()

    if args.idx is not None:
        idx = args.idx
    else:
        idx = find_disagreement(X, y, mlp_a, mlp_b,
                                target_class=args.target_class)
        if idx is None:
            raise SystemExit("No se encontró ejemplo con A correcto y B equivocado.")

    out_path = out_dir / f"compare_{pa.stem}_vs_{pb.stem}_idx{idx}.png"
    render_compare(
        X[idx], int(y[idx]), mlp_a, mlp_b, (pa.stem, pb.stem),
        args.patch, args.stride, out_path,
    )

    print(f"OK | A: {pa.name}  B: {pb.name}")
    print(f"OK | idx elegido: {idx}  (y_true={y[idx]})")
    print(f"OK | figura: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
