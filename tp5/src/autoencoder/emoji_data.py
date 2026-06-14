"""Dataset de emojis para el VAE (Ej2a).

Rasteriza un set curado de emojis desde `NotoColorEmoji.ttf` (Pillow) a imágenes en
escala de grises de `size×size` en `[0,1]` (tinta=1, fondo=0), aplanadas a vectores de
`size·size`. El glifo se centra por bounding-box para un encuadre consistente. La salida es
determinista (depende solo de fuente y tamaño).

NotoColorEmoji es una fuente de bitmaps a color con un único "strike" a 109 px; al pasar a
grises se conserva la forma/sombras y se pierde el color, suficiente para que el VAE 2D
organice los glifos en el latente y genere muestras nuevas (Ej2c).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
# Único tamaño de bitmap disponible en NotoColorEmoji.
NOTO_STRIKE = 109
# Lienzo de rasterizado previo al recorte/redimensionado (holgado para el strike de 109px).
_CANVAS = 160
# Umbral de "tinta" (sobre 255) para el bounding-box del glifo.
_INK_THRESHOLD = 8

# Set curado de emojis visualmente distintos (glifos simples, sin secuencias ZWJ/skin-tone).
EMOJIS: list[tuple[str, str]] = [
    ("😀", "grin"), ("😃", "smile"), ("😄", "laugh"), ("😁", "beam"),
    ("😆", "squint"), ("😅", "sweat"), ("😂", "joy"), ("🙂", "slight"),
    ("😉", "wink"), ("😊", "blush"), ("😍", "heart_eyes"), ("😎", "cool"),
    ("🤔", "think"), ("😴", "sleep"), ("😭", "cry"), ("😡", "rage"),
    ("🐱", "cat"), ("🐶", "dog"), ("🐭", "mouse"), ("🐰", "rabbit"),
    ("🦊", "fox"), ("🐻", "bear"), ("🐼", "panda"), ("🐯", "tiger"),
    ("🍎", "apple"), ("🍊", "orange"), ("🍋", "lemon"), ("🍓", "strawberry"),
    ("🍕", "pizza"), ("⭐", "star"), ("🌙", "moon"), ("❤", "heart"),
]


def _load_font(font_path: str | Path = DEFAULT_FONT, strike: int = NOTO_STRIKE):
    """Carga la fuente; error claro si no existe."""
    from PIL import ImageFont

    if not Path(font_path).exists():
        raise FileNotFoundError(
            f"No se encontró la fuente de emojis: {font_path}. Instalá "
            "'fonts-noto-color-emoji' o pasá font_path a otra fuente.")
    try:
        return ImageFont.truetype(str(font_path), size=strike)
    except OSError:
        # Fuentes monocromas escalables: cualquier tamaño sirve.
        return ImageFont.truetype(str(font_path), size=64)


def render_emoji(ch: str, size: int = 28, font_path: str | Path = DEFAULT_FONT,
                 font=None) -> np.ndarray:
    """Rasteriza un emoji a un vector de `size·size` en `[0,1]` (tinta=1, fondo=0)."""
    from PIL import Image, ImageDraw

    font = font or _load_font(font_path)
    img = Image.new("RGBA", (_CANVAS, _CANVAS), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # anchor='mm': el glifo se dibuja centrado en el punto medio del lienzo.
    draw.text((_CANVAS // 2, _CANVAS // 2), ch, font=font,
              embedded_color=True, anchor="mm")

    # Luminancia -> tinta (alto donde es oscuro sobre fondo blanco).
    ink = 255 - np.asarray(img.convert("L"), dtype=np.float64)

    # Recorte por bounding-box de la tinta y encuadre cuadrado centrado.
    ys, xs = np.where(ink > _INK_THRESHOLD)
    if xs.size:
        crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        crop = ink
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float64)
    square[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop

    small = Image.fromarray(square.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return (np.asarray(small, dtype=np.float64) / 255.0).reshape(-1)


def load_emojis(size: int = 28, subset: list[int] | None = None,
                font_path: str | Path = DEFAULT_FONT) -> tuple[np.ndarray, list[str]]:
    """Carga el dataset de emojis. Devuelve `(X, labels)` con `X` de forma `(N, size·size)`."""
    font = _load_font(font_path)
    items = EMOJIS if subset is None else [EMOJIS[i] for i in subset]
    X = np.stack([render_emoji(ch, size, font=font) for ch, _ in items])
    labels = [label for _, label in items]
    return X, labels


def augment_dataset(
    X: np.ndarray,
    labels: list[str],
    size: int = 28,
    n_aug: int = 4,
    rng: np.random.Generator | None = None,
    max_rot: float = 15.0,
    max_shift: float = 2.0,
    max_zoom: float = 0.12,
) -> tuple[np.ndarray, list[str]]:
    """Expande el dataset con copias levemente rotadas/trasladadas/escaladas (opcional).

    Enriquece el latente para suavizar la generación de un set chico de emojis. El original
    queda incluido como primer bloque. Determinista dado `rng`.
    """
    from scipy.ndimage import rotate, shift

    rng = rng or np.random.default_rng()
    grids = X.reshape(-1, size, size)
    out_X = [X]
    out_labels = list(labels)
    for _ in range(n_aug):
        block = []
        for g in grids:
            ang = float(rng.uniform(-max_rot, max_rot))
            zoom = 1.0 + float(rng.uniform(-max_zoom, max_zoom))
            r = rotate(g, ang, reshape=False, order=1, mode="constant", cval=0.0)
            dy, dx = rng.uniform(-max_shift, max_shift, size=2)
            r = shift(r * zoom, (dy, dx), order=1, mode="constant", cval=0.0)
            block.append(np.clip(r, 0.0, 1.0).reshape(-1))
        out_X.append(np.stack(block))
        out_labels += list(labels)
    return np.vstack(out_X), out_labels
