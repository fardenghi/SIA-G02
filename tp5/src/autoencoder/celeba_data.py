"""Loader de CelebA (caras alineadas) para el experimento de generación del VAE.

Cierra el arco del TP: empezamos queriendo generar caras (emojis) y fallamos por densidad del
dataset (110 glifos). CelebA tiene ~163k caras **alineadas y centradas** — densas y de calidad
genuina (fotos), pero estructuralmente regulares (todas las caras en la misma posición), así que
la complejidad queda domada para un latente moderado.

No se baja el dataset entero: se pagina el datasets-server de HuggingFace, se descarga un
subconjunto en paralelo, se hace center-crop a la cara + resize a `size`×`size` en gris, y se
**cachea a `.npy`** (las corridas siguientes son instantáneas). Devuelve X en [0,1].

Esperable: caras **borrosas pero coherentes** (el clásico "VAE face blur" de la pérdida-píxel),
no nítidas. Lo importante es que la densidad evita el grano que teníamos con emojis.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

_API = "https://datasets-server.huggingface.co/rows"
_DATASET = "tpremoli/CelebA-attrs"
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "celeba"
# Crop centrado en la cara del alineado 178×218 -> 128×128 (saca fondo/pelo de los bordes).
_CROP = (25, 45, 153, 173)  # (left, upper, right, lower)


def _get(url: str, tries: int = 6) -> bytes:
    """GET con reintentos y backoff exponencial ante 429 (rate limit) / errores transitorios."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(2 ** i)                                  # 1,2,4,8,16 s
        except urllib.error.URLError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def _fetch_urls(n: int) -> list[str]:
    """Pagina el datasets-server (máx 100 filas/llamada) y junta n URLs de imagen."""
    urls: list[str] = []
    offset = 0
    while len(urls) < n:
        length = min(100, n - len(urls))
        q = urllib.parse.urlencode({"dataset": _DATASET, "config": "default",
                                    "split": "train", "offset": offset, "length": length})
        rows = json.loads(_get(f"{_API}?{q}"))["rows"]
        if not rows:
            break
        urls.extend(row["row"]["image"]["src"] for row in rows)
        offset += len(rows)
    return urls[:n]


def _load_one(url: str, size: int) -> np.ndarray:
    img = Image.open(io.BytesIO(_get(url))).convert("L")        # gris
    img = img.crop(_CROP).resize((size, size), Image.BILINEAR)  # cara centrada -> size×size
    # Invertir (1-x): las fotos son claro-sobre-oscuro; con cmap='Greys' (0=blanco) eso saldría
    # en negativo. Invirtiendo matcheamos la convención "tinta" de MNIST/emojis (piel clara=0).
    return 1.0 - np.asarray(img, dtype=np.float64).ravel() / 255.0


def load_celeba(n: int = 3000, size: int = 64, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Carga n caras de CelebA como (X, labels). X en [0,1], shape (N, size*size), gris.

    Cachea a `data/celeba/celeba_n{n}_s{size}.npy`. Descarga en paralelo (tolera fallos sueltos).
    `seed` no baraja (toma las primeras n filas, determinista); está por simetría con los otros
    loaders.
    """
    cache = _DATA_DIR / f"celeba_n{n}_s{size}.npy"
    if cache.exists():
        X = np.load(cache)
        return X, [str(i) for i in range(X.shape[0])]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    urls = _fetch_urls(n)
    out: list[np.ndarray | None] = [None] * len(urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_load_one, u, size): i for i, u in enumerate(urls)}
        for fut in concurrent.futures.as_completed(futs):
            try:
                out[futs[fut]] = fut.result()
            except Exception:
                out[futs[fut]] = None                            # imagen suelta que falló: se omite
    X = np.stack([x for x in out if x is not None])
    np.save(cache, X)
    return X, [str(i) for i in range(X.shape[0])]
