"""Loader de texturas de bloques de Minecraft para el VAE.

Descarga el cliente de Minecraft (jar público de Mojang), extrae los PNGs de
`assets/minecraft/textures/block/`, los convierte a escala de grises y los cachea en
`data/minecraft/`. Las texturas son nativas 16×16; se redimensionan al `size` pedido.

Descarga perezosa: solo la primera vez (~30-50 MB el jar). Las siguientes corridas
leen el caché .npy instantáneamente.

    from autoencoder.minecraft_data import load_minecraft
    X, labels = load_minecraft(size=16)              # ~300 bloques modernos, 16×16 gris
    X, labels = load_minecraft(size=16, classic=True) # texturas pre-1.14 (Programmer Art)
    X, labels = load_minecraft(size=28, color=True)   # bloques modernos RGB
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

_MANIFEST = "https://launchermeta.mojang.com/mc/game/version_manifest_v2.json"
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "minecraft"

# Patrones que identifican texturas que NO son bloques sólidos
_BLOCK_EXCLUDE = [
    "_stairs", "_slab", "_fence", "_fence_gate", "_gate",
    "_pane", "_trapdoor", "_door", "_wall",
    "_plant", "_bush", "_leaves", "_sapling", "_fern", "_grass",
    "_torch", "_candle", "_lantern", "_chain", "_bars",
    "water", "lava",
    "_snow", "_vine", "_kelp", "_seagrass", "_coral",
]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


_CLASSIC_VERSION = "1.12.2"  # última versión con las texturas originales (pre-texture-update 1.14)


def _get_jar_url(version_id: str | None = None) -> str:
    """Devuelve la URL del client.jar. Sin version_id usa el último release."""
    manifest = json.loads(_get(_MANIFEST))
    target = version_id or manifest["latest"]["release"]
    for v in manifest["versions"]:
        if v["id"] == target:
            version_json = json.loads(_get(v["url"]))
            return version_json["downloads"]["client"]["url"]
    raise RuntimeError(f"No se encontró la versión '{target}' en el manifest de Mojang")


def _load_png(data: bytes, size: int, color: bool = False) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        # Ignorar texturas animadas (más altas que anchas) y no cuadradas
        if img.height != img.width:
            return None
        # Componer sobre fondo blanco para que el alpha quede bien
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        if color:
            out = bg.convert("RGB").resize((size, size), Image.BILINEAR)
            return np.asarray(out, dtype=np.float64).ravel() / 255.0  # shape: size*size*3
        else:
            out = bg.convert("L").resize((size, size), Image.BILINEAR)
            return np.asarray(out, dtype=np.float64).ravel() / 255.0  # shape: size*size
    except Exception:
        return None


def _extract_block_textures(zf: zipfile.ZipFile, size: int, color: bool,
                             blocks_only: bool) -> tuple[list, list]:
    """Extrae texturas de bloques de un zipfile abierto. Devuelve (images, names)."""
    # Moderno (1.13+): textures/block/   Clásico (<=1.12): textures/blocks/
    for prefix in ("assets/minecraft/textures/block/", "assets/minecraft/textures/blocks/"):
        block_files = [f for f in zf.namelist()
                       if f.startswith(prefix) and f.endswith(".png")]
        if block_files:
            break
    images, names = [], []
    for path in sorted(block_files):
        name = Path(path).stem
        if blocks_only and any(kw in name for kw in _BLOCK_EXCLUDE):
            continue
        arr = _load_png(zf.read(path), size, color=color)
        if arr is not None:
            images.append(arr)
            names.append(name)
    return images, names


def load_minecraft(size: int = 16, color: bool = False, n: int | None = None,
                   seed: int = 0, blocks_only: bool = False,
                   classic: bool = False) -> tuple[np.ndarray, list[str]]:
    """Carga texturas de bloques de Minecraft como (X, labels). X en [0,1].

    shape (N, size*size) en grises o (N, size*size*3) en color.
    `n` submuestrea aleatoriamente (None = todos los bloques).
    `blocks_only` filtra escaleras, vallas, puertas, plantas, etc.
    `classic` descarga el jar de Minecraft 1.12.2 (texturas pre-texture-update 1.14).
    Cachea a `data/minecraft/minecraft_s{size}_{mode}[_classic][_blocksonly].npy`.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    mode = "rgb" if color else "gray"
    suffix = ("_classic" if classic else "") + ("_blocksonly" if blocks_only else "")
    cache = _DATA_DIR / f"minecraft_s{size}_{mode}{suffix}.npy"
    labels_cache = _DATA_DIR / f"minecraft_s{size}_labels{suffix}.txt"

    if cache.exists() and labels_cache.exists():
        X = np.load(cache)
        labels = labels_cache.read_text().splitlines()
        if n is not None and n < X.shape[0]:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=n, replace=False)
            X, labels = X[idx], [labels[i] for i in idx]
        return X, labels

    # Descargar jar(s) si no existen
    jar_path = _DATA_DIR / ("client_classic.jar" if classic else "client.jar")
    if not jar_path.exists():
        label = f"Minecraft {_CLASSIC_VERSION} (texturas clásicas)" if classic else "Minecraft (último release)"
        print(f"Descargando cliente de {label} (puede tardar unos minutos)...")
        version_id = _CLASSIC_VERSION if classic else None
        jar_url = _get_jar_url(version_id)
        urllib.request.urlretrieve(jar_url, jar_path)
        print("Descarga completa.")

    with zipfile.ZipFile(jar_path) as zf:
        images, names = _extract_block_textures(zf, size, color, blocks_only)

    X = np.stack(images)
    np.save(cache, X)
    labels_cache.write_text("\n".join(names))
    channels = "RGB" if color else "gris"
    era = " clásicas (Programmer Art)" if classic else ""
    bo_str = " (solo bloques sólidos)" if blocks_only else ""
    print(f"Cargados {X.shape[0]} bloques de Minecraft{era} ({size}×{size} {channels}){bo_str}.")

    if n is not None and n < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=n, replace=False)
        X, names = X[idx], [names[i] for i in idx]

    return X, names
