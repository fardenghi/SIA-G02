"""Métodos de atribución para interpretar el MLP de ej3.

Implementa dos técnicas:
  1. Saliency map (vanilla gradient): |∂score_clase / ∂input|
  2. Occlusion sensitivity: cuánto cae la confianza al tapar una región

Genera, para cada dígito 0-9, una figura con tres paneles:
    [imagen original] [saliency overlay] [occlusion heatmap]

Uso:
    uv run python -m exercises.ej3_more_digits.interpretability \\
        outputs/ej3_more_digits/models/arch_wide.npz

Opciones:
    --patch N        tamaño del cuadrado de oclusión (default 4)
    --stride N       paso de oclusión (default 2)
    --out PATH       directorio de salida (default outputs/.../interpretability)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_digits_test
from common.mlp import MLP

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "ej3_more_digits" / "interpretability"
IMG_SIZE = 28


def saliency_map(mlp: MLP, x: np.ndarray, target_class: int) -> np.ndarray:
    """Gradiente del logit (pre-softmax) de target_class respecto al input.

    Returns un array (784,) con ∂z_out[target] / ∂X.
    """
    X = x.reshape(1, -1)
    mlp.forward(X)

    # delta inicial = ∂score/∂z_out = one-hot(target_class)
    delta = np.zeros_like(mlp.layers[-1].z)
    delta[0, target_class] = 1.0

    for l in reversed(range(len(mlp.layers))):
        layer = mlp.layers[l]
        a_prev = X if l == 0 else mlp.layers[l - 1].a
        _, _, delta_a = layer.backward(delta, a_prev)
        if l > 0:
            prev = mlp.layers[l - 1]
            delta = delta_a * prev.activation_prime(prev.z)
        else:
            grad_input = delta_a  # shape (1, 784)

    return grad_input.ravel()


def softmax_prob(mlp: MLP, X: np.ndarray) -> np.ndarray:
    """Probabilidades softmax (la salida del MLP ya viene con softmax)."""
    return mlp.forward(X)


def occlusion_map(mlp: MLP, x: np.ndarray, target_class: int,
                  patch: int = 4, stride: int = 2) -> np.ndarray:
    """Mapa de sensibilidad por oclusión.

    Para cada posición (i, j), tapa un cuadrado patch×patch con 0 y mide
    la caída en P(target_class). El mapa devuelto tiene el tamaño de la imagen,
    promediando aportes de cada patch que cubre el píxel.
    """
    img = x.reshape(IMG_SIZE, IMG_SIZE)
    base_prob = softmax_prob(mlp, x.reshape(1, -1))[0, target_class]

    drop = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)
    counts = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)

    positions = []
    occluded = []
    for i in range(0, IMG_SIZE - patch + 1, stride):
        for j in range(0, IMG_SIZE - patch + 1, stride):
            occ = img.copy()
            occ[i:i + patch, j:j + patch] = 0.0
            occluded.append(occ.ravel())
            positions.append((i, j))

    occluded = np.stack(occluded, axis=0)
    probs = softmax_prob(mlp, occluded)[:, target_class]
    drops = base_prob - probs  # positivo => tapar bajó la confianza => esa zona era importante

    for (i, j), d in zip(positions, drops):
        drop[i:i + patch, j:j + patch] += d
        counts[i:i + patch, j:j + patch] += 1.0

    counts[counts == 0] = 1.0
    return drop / counts


def pick_examples(X: np.ndarray, y: np.ndarray, mlp: MLP) -> dict[int, int]:
    """Para cada clase 0-9 elige una imagen bien clasificada con alta confianza."""
    probs = softmax_prob(mlp, X)
    preds = probs.argmax(axis=1)
    chosen = {}
    for c in range(10):
        mask = (y == c) & (preds == c)
        if not mask.any():
            mask = (y == c)
        idx_pool = np.where(mask)[0]
        confidences = probs[idx_pool, c]
        chosen[c] = int(idx_pool[np.argmax(confidences)])
    return chosen


def pick_wrong_examples(X: np.ndarray, y: np.ndarray, mlp: MLP) -> dict[int, int]:
    """Elige hasta una imagen mal clasificada por clase real, con la mayor
    confianza en la predicción equivocada (errores 'seguros' del modelo).

    Devuelve dict {clase_real -> idx}. Clases sin errores quedan fuera.
    """
    probs = softmax_prob(mlp, X)
    preds = probs.argmax(axis=1)
    chosen = {}
    for c in range(10):
        mask = (y == c) & (preds != c)
        if not mask.any():
            continue
        idx_pool = np.where(mask)[0]
        wrong_conf = probs[idx_pool, preds[idx_pool]]
        chosen[c] = int(idx_pool[np.argmax(wrong_conf)])
    return chosen


def render_grid(X, y, mlp, indices, patch, stride, out_path, model_name=""):
    fig, axes = plt.subplots(len(indices), 3, figsize=(7, 2.2 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]

    for row, (cls, idx) in enumerate(indices.items()):
        x = X[idx]
        true_y = int(y[idx])
        probs = softmax_prob(mlp, x.reshape(1, -1))[0]
        pred = int(probs.argmax())
        target = pred  # explicamos lo que el modelo predice

        sal = saliency_map(mlp, x, target)
        sal_abs = np.abs(sal).reshape(IMG_SIZE, IMG_SIZE)
        if sal_abs.max() > 0:
            sal_abs /= sal_abs.max()

        occ = occlusion_map(mlp, x, target, patch=patch, stride=stride)
        occ_norm = occ.copy()
        peak = np.abs(occ_norm).max()
        if peak > 0:
            occ_norm /= peak

        img = x.reshape(IMG_SIZE, IMG_SIZE)

        ax_img, ax_sal, ax_occ = axes[row]
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"y={true_y}  pred={pred} ({probs[pred]:.2f})", fontsize=9)
        ax_img.axis("off")

        ax_sal.imshow(img, cmap="gray", alpha=0.5)
        ax_sal.imshow(sal_abs, cmap="hot", alpha=0.6)
        ax_sal.set_title("Saliency |∂logit/∂x|", fontsize=9)
        ax_sal.axis("off")

        ax_occ.imshow(img, cmap="gray", alpha=0.5)
        im = ax_occ.imshow(occ_norm, cmap="seismic", vmin=-1, vmax=1, alpha=0.6)
        ax_occ.set_title(f"Occlusion ({patch}x{patch})", fontsize=9)
        ax_occ.axis("off")

    fig.suptitle(f"Atribución — {model_name}", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="ruta al .npz del modelo")
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--wrong", action="store_true",
                   help="usa ejemplos mal clasificados en vez de los mejores")
    p.add_argument("--classes", type=str, default=None,
                   help="lista de clases separadas por coma (ej. '3,6')")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"No existe {model_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlp = MLP.load(str(model_path))
    X, y = load_digits_test()

    if args.wrong:
        chosen = pick_wrong_examples(X, y, mlp)
        suffix = "_wrong"
        title = f"{model_path.stem} (errores)"
    else:
        chosen = pick_examples(X, y, mlp)
        suffix = "_attribution"
        title = model_path.stem

    if args.classes is not None:
        wanted = [int(c) for c in args.classes.split(",") if c.strip()]
        chosen = {c: chosen[c] for c in wanted if c in chosen}
        suffix = suffix + "_" + "-".join(str(c) for c in wanted)

    if not chosen:
        raise SystemExit("No se encontraron ejemplos para los criterios pedidos.")

    out_path = out_dir / f"{model_path.stem}{suffix}.png"
    render_grid(X, y, mlp, chosen, args.patch, args.stride, out_path,
                model_name=title)

    print(f"OK | modelo: {model_path.name}")
    print(f"OK | figura: {out_path.relative_to(ROOT)}")
    print(f"OK | dígitos elegidos (clase -> idx): {chosen}")


if __name__ == "__main__":
    main()
