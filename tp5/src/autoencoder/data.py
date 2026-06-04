"""Carga del dataset `font.h` y utilidades asociadas.

Cada carácter en `font.h` son 7 enteros de 5 bits (`0x00`-`0x1f`). Se desempaqueta
cada fila a sus 5 bits del bit 4 (MSB) al bit 0 (LSB) y se aplanan las 7 filas
fila-por-fila, produciendo una matriz binaria `X` de forma `(32, 35)`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

GLYPH_ROWS = 7
GLYPH_COLS = 5
GLYPH_SIZE = GLYPH_ROWS * GLYPH_COLS  # 35

# Etiquetas de los 32 caracteres tal como aparecen en font.h (0x60..0x7f).
CHAR_LABELS = [
    "`", "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "{", "|", "}", "~", "DEL",
]


def _parse_rows(font_path: str | Path) -> np.ndarray:
    """Extrae los 32 patrones de 7 enteros desde el archivo `font.h`.

    Devuelve un arreglo de forma `(32, 7)` con los enteros de 5 bits.
    """
    text = Path(font_path).read_text()
    patterns: list[list[int]] = []
    # Cada fila del font es una secuencia de 7 enteros hex entre llaves: { 0x.., ... }
    for match in re.finditer(r"\{([^{}]*0x[^{}]*)\}", text):
        body = match.group(1)
        ints = [int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]+", body)]
        if len(ints) == GLYPH_ROWS:
            patterns.append(ints)
    arr = np.array(patterns, dtype=np.int64)
    if arr.shape != (32, GLYPH_ROWS):
        raise ValueError(
            f"Se esperaban 32 patrones de {GLYPH_ROWS} enteros, se obtuvo {arr.shape}"
        )
    return arr


def unpack_bits(value: int) -> np.ndarray:
    """Desempaqueta un entero de 5 bits del bit 4 (MSB) al bit 0 (LSB).

    Ejemplo: `0x1f` -> `[1, 1, 1, 1, 1]`; `0x10` -> `[1, 0, 0, 0, 0]`.
    """
    return np.array([(value >> bit) & 1 for bit in range(GLYPH_COLS - 1, -1, -1)],
                    dtype=np.float64)


def load_font(font_path: str | Path) -> np.ndarray:
    """Carga `font.h` y devuelve la matriz binaria `X` de forma `(32, 35)`."""
    rows = _parse_rows(font_path)
    X = np.zeros((rows.shape[0], GLYPH_SIZE), dtype=np.float64)
    for i, pattern in enumerate(rows):
        flat = np.concatenate([unpack_bits(int(v)) for v in pattern])
        X[i] = flat
    return X


def select_subset(X: np.ndarray, subset: list[int] | None) -> np.ndarray:
    """Selecciona un subconjunto de patrones por índice; `None` devuelve todos."""
    if subset is None:
        return X
    return X[np.asarray(subset, dtype=int)]


def labels_for_subset(subset: list[int] | None) -> list[str]:
    """Devuelve las etiquetas de carácter correspondientes al subset."""
    if subset is None:
        return list(CHAR_LABELS)
    return [CHAR_LABELS[i] for i in subset]


def to_grid(vector: np.ndarray) -> np.ndarray:
    """Convierte un vector de 35 valores a una grilla `(7, 5)` para visualizar."""
    vector = np.asarray(vector).reshape(-1)
    if vector.size != GLYPH_SIZE:
        raise ValueError(f"Se esperaban {GLYPH_SIZE} valores, se obtuvo {vector.size}")
    return vector.reshape(GLYPH_ROWS, GLYPH_COLS)


def add_noise(
    X: np.ndarray,
    noise_type: str = "salt_pepper",
    level: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Corrompe entradas binarias invirtiendo píxeles con probabilidad `level`.

    - `salt_pepper`: cada píxel se fuerza a 0 o 1 al azar con probabilidad `level`.
    - `bit_flip`: cada píxel se invierte (`1-x`) con probabilidad `level`.

    Con `level == 0` devuelve una copia idéntica a `X`.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X, dtype=np.float64)
    if level <= 0:
        return X.copy()

    mask = rng.random(X.shape) < level
    out = X.copy()
    if noise_type == "salt_pepper":
        salt = rng.random(X.shape) < 0.5
        out[mask] = salt[mask].astype(np.float64)
    elif noise_type == "bit_flip":
        out[mask] = 1.0 - out[mask]
    else:
        raise ValueError(f"noise_type desconocido: {noise_type!r}")
    return out
