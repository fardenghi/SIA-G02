"""Loader de datasets IDX (MNIST / Fashion-MNIST) para los experimentos de control del VAE.

Hipótesis: los emojis son demasiado complejos / pocos para la capacidad de la red. MNIST y
Fashion-MNIST son benchmarks canónicos — 28×28 gris (misma dimensión que los emojis, cero
cambios al pipeline) pero con decenas de miles de muestras. Si la MISMA red genera dígitos /
prendas decentes, el techo de los emojis era del dataset (manifold ralo), no de la arquitectura.
Fashion-MNIST es drop-in (mismo formato IDX) pero visualmente más rico: siluetas de ropa.

Descarga perezosa de los IDX públicos a `data/<kind>/` (una sola vez). Devuelve X en [0,1].
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

# kind -> mirror base. Mismos nombres de archivo IDX en ambos.
_MIRRORS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist",
    "fashion": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
}
_FILES = {
    "images": "train-images-idx3-ubyte.gz",
    "labels": "train-labels-idx1-ubyte.gz",
}
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _download(kind: str, name: str) -> Path:
    data_dir = _DATA_ROOT / kind
    data_dir.mkdir(parents=True, exist_ok=True)
    dst = data_dir / name
    if not dst.exists():
        urllib.request.urlretrieve(f"{_MIRRORS[kind]}/{name}", dst)
    return dst


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols).astype(np.float64) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).copy()


def load_mnist(n: int | None = None, digits: list[int] | None = None,
               seed: int = 0, kind: str = "mnist",
               size: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Carga MNIST o Fashion-MNIST como (X, labels). X en [0,1], shape (N, size²), gris.

    `kind` ∈ {"mnist", "fashion"}. `n` submuestrea (para que el VAE numpy entrene en tiempo
    razonable). `digits` filtra a un subconjunto de clases (0-9; en fashion son tipos de prenda,
    p.ej. [0,1,8]=remera/pantalón/cartera). Barajado determinista.

    `size` (opcional) redimensiona desde el nativo 28×28 con PIL BILINEAR; `None` o 28 lo deja
    intacto. OJO: subir de 28 es interpolar (no agrega detalle real), solo da más píxeles.
    """
    X = _read_idx_images(_download(kind, _FILES["images"]))
    y = _read_idx_labels(_download(kind, _FILES["labels"]))
    if digits is not None:
        mask = np.isin(y, digits)
        X, y = X[mask], y[mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    if n is not None:
        perm = perm[:n]
    X, y = X[perm], y[perm]
    if size is not None and size != 28:
        from PIL import Image
        imgs = X.reshape(-1, 28, 28)
        X = np.stack([
            np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize(
                (size, size), Image.BILINEAR), dtype=np.float64).ravel() / 255.0
            for im in imgs
        ])
    return X, [str(int(d)) for d in y]
