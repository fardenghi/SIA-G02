"""Loader de emojis-cara MULTI-VENDOR: muchos estilos de los mismos conceptos.

Idea (analogía de MNIST): un solo font de emojis da ~110 caras homogéneas (tope del glifo).
Pero los mismos ~130 conceptos de cara dibujados por **varios vendors** (Apple, Google, Facebook,
Twitter, Twemoji, OpenMoji color/negro, Noto) dan ~700-1000 imágenes de cara **homogéneas pero
genuinamente diversas en estilo** — igual que MNIST son los mismos 10 dígitos con miles de
caligrafías. Sirve para testear si el techo de los emojis era **cantidad de datos** (densidad del
manifold) y no la complejidad por imagen.

Descarga PNGs por codepoint desde jsdelivr (CDN), los pasa a la MISMA convención que
`render_emoji` (tinta=alto sobre fondo 0, recorte al bounding-box y encuadre cuadrado) y cachea a
`data/emoji_multi/`. Devuelve X en [0,1].
"""

from __future__ import annotations

import concurrent.futures
import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "emoji_multi"
_INK_THRESHOLD = 8

# Rangos de caras (homogéneos): smileys + caras suplementarias. Mismos conceptos que FACES_RANGES.
_FACE_RANGES = [(0x1F600, 0x1F64A), (0x1F910, 0x1F917), (0x1F920, 0x1F92F),
                (0x1F970, 0x1F97A), (0x1FAE0, 0x1FAE8)]


def _face_codepoints() -> list[int]:
    return [cp for a, b in _FACE_RANGES for cp in range(a, b + 1)]


# vendor -> función(codepoint)->URL. Todos direccionables por codepoint en jsdelivr.
def _vendor_urls(cp: int) -> dict[str, str]:
    lo, up = f"{cp:x}", f"{cp:X}"
    ds = "https://cdn.jsdelivr.net/npm/emoji-datasource-{v}/img/{v}/64/" + lo + ".png"
    return {
        "apple": ds.format(v="apple"),
        "google": ds.format(v="google"),
        "facebook": ds.format(v="facebook"),
        "twitter": ds.format(v="twitter"),
        "twemoji": f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/72x72/{lo}.png",
        "openmoji_color": f"https://cdn.jsdelivr.net/gh/hfg-gmuend/openmoji@latest/color/72x72/{up}.png",
        "openmoji_black": f"https://cdn.jsdelivr.net/gh/hfg-gmuend/openmoji@latest/black/72x72/{up}.png",
        "noto": f"https://cdn.jsdelivr.net/gh/googlefonts/noto-emoji@main/png/72/emoji_u{lo}.png",
    }


def _get(url: str, tries: int = 5) -> bytes | None:
    """GET con backoff. Devuelve None ante 404 (el vendor no tiene ese emoji)."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == tries - 1:
                return None
            time.sleep(2 ** i)
        except urllib.error.URLError:
            if i == tries - 1:
                return None
            time.sleep(2 ** i)
    return None


def _to_gray_vec(png: bytes, size: int) -> np.ndarray | None:
    """PNG (con alpha) -> vector gris [0,1] en convención tinta (=alto), igual que render_emoji."""
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))            # fondo blanco
    L = np.asarray(Image.alpha_composite(bg, im).convert("L"), dtype=np.float64)
    ink = 255.0 - L                                                  # tinta=alto, fondo=0
    ys, xs = np.where(ink > _INK_THRESHOLD)
    if xs.size == 0:
        return None
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float64)                # encuadre cuadrado sobre 0
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    small = Image.fromarray(square.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return (np.asarray(small, dtype=np.float64) / 255.0).reshape(-1)


def load_multi_emojis(size: int = 28, seed: int = 0,
                      max_workers: int = 12) -> tuple[np.ndarray, list[str]]:
    """Carga caras de emoji de 8 vendors como (X, labels). X en [0,1], (N, size*size), gris.

    Cachea a `data/emoji_multi/emoji_multi_s{size}.npy`. Tolera 404 (vendor sin ese emoji).
    `labels` = "U+XXXX@vendor". `seed` solo baraja el orden final.
    """
    cache = _DATA_DIR / f"emoji_multi_s{size}.npy"
    if cache.exists():
        X = np.load(cache)
        return X, [str(i) for i in range(X.shape[0])]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(cp, vendor, url) for cp in _face_codepoints()
            for vendor, url in _vendor_urls(cp).items()]

    def fetch(job):
        cp, vendor, url = job
        png = _get(url)
        if png is None:
            return None
        try:
            vec = _to_gray_vec(png, size)
        except Exception:
            return None
        return None if vec is None else (vec, f"U+{cp:04X}@{vendor}")

    rows: list[tuple[np.ndarray, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(fetch, jobs):
            if res is not None:
                rows.append(res)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    X = np.stack([rows[i][0] for i in perm])
    labels = [rows[i][1] for i in perm]
    np.save(cache, X)
    return X, labels
