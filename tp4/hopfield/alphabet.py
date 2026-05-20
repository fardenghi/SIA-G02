"""Patrones 5x5 del abecedario completo (A-Z) en {+1, -1}.

Cada letra se define como una cadena de 25 caracteres usando "*" para +1 y "." para -1,
distribuidos en 5 filas de 5 columnas (top-down, left-to-right).
"""
from __future__ import annotations

import numpy as np


_RAW: dict[str, str] = {
    "A": (
        ".***."
        "*...*"
        "*****"
        "*...*"
        "*...*"
    ),
    "B": (
        "****."
        "*...*"
        "****."
        "*...*"
        "****."
    ),
    "C": (
        ".****"
        "*...."
        "*...."
        "*...."
        ".****"
    ),
    "D": (
        "****."
        "*...*"
        "*...*"
        "*...*"
        "****."
    ),
    "E": (
        "*****"
        "*...."
        "****."
        "*...."
        "*****"
    ),
    "F": (
        "*****"
        "*...."
        "****."
        "*...."
        "*...."
    ),
    "G": (
        ".****"
        "*...."
        "*..**"
        "*...*"
        ".***."
    ),
    "H": (
        "*...*"
        "*...*"
        "*****"
        "*...*"
        "*...*"
    ),
    "I": (
        "*****"
        "..*.."
        "..*.."
        "..*.."
        "*****"
    ),
    "J": (
        "*****"
        "...*."
        "...*."
        "*..*."
        ".**.."
    ),
    "K": (
        "*...*"
        "*..*."
        "***.."
        "*..*."
        "*...*"
    ),
    "L": (
        "*...."
        "*...."
        "*...."
        "*...."
        "*****"
    ),
    "M": (
        "*...*"
        "**.**"
        "*.*.*"
        "*...*"
        "*...*"
    ),
    "N": (
        "*...*"
        "**..*"
        "*.*.*"
        "*..**"
        "*...*"
    ),
    "O": (
        ".***."
        "*...*"
        "*...*"
        "*...*"
        ".***."
    ),
    "P": (
        "****."
        "*...*"
        "****."
        "*...."
        "*...."
    ),
    "Q": (
        ".***."
        "*...*"
        "*.*.*"
        "*..*."
        ".**.*"
    ),
    "R": (
        "****."
        "*...*"
        "****."
        "*..*."
        "*...*"
    ),
    "S": (
        ".****"
        "*...."
        ".***."
        "....*"
        "****."
    ),
    "T": (
        "*****"
        "..*.."
        "..*.."
        "..*.."
        "..*.."
    ),
    "U": (
        "*...*"
        "*...*"
        "*...*"
        "*...*"
        ".***."
    ),
    "V": (
        "*...*"
        "*...*"
        "*...*"
        ".*.*."
        "..*.."
    ),
    "W": (
        "*...*"
        "*...*"
        "*.*.*"
        "**.**"
        "*...*"
    ),
    "X": (
        "*...*"
        ".*.*."
        "..*.."
        ".*.*."
        "*...*"
    ),
    "Y": (
        "*...*"
        ".*.*."
        "..*.."
        "..*.."
        "..*.."
    ),
    "Z": (
        "*****"
        "...*."
        "..*.."
        ".*..."
        "*****"
    ),
}


def _to_matrix(raw: str) -> np.ndarray:
    if len(raw) != 25:
        raise ValueError(f"Patrón inválido (largo {len(raw)}, se esperaban 25)")
    bits = [1 if ch == "*" else -1 for ch in raw]
    return np.array(bits, dtype=np.int8).reshape(5, 5)


ALPHABET: dict[str, np.ndarray] = {ch: _to_matrix(raw) for ch, raw in _RAW.items()}
LETTERS: list[str] = list(ALPHABET.keys())


def get_letter(ch: str) -> np.ndarray:
    ch = ch.upper()
    if ch not in ALPHABET:
        raise KeyError(f"Letra '{ch}' no está en el abecedario")
    return ALPHABET[ch].copy()


def letter_vector(ch: str) -> np.ndarray:
    return get_letter(ch).flatten().astype(np.int8)


def letters_in_range(start: str, end: str) -> list[str]:
    start, end = start.upper(), end.upper()
    if start not in ALPHABET or end not in ALPHABET:
        raise KeyError(f"Rango inválido: {start}-{end}")
    i, j = LETTERS.index(start), LETTERS.index(end)
    if i > j:
        i, j = j, i
    return LETTERS[i : j + 1]


def render_ascii(matrix: np.ndarray) -> str:
    rows = []
    for row in matrix:
        rows.append(" ".join("*" if v == 1 else " " for v in row))
    return "\n".join(rows)
