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
                 font=None, color: bool = False) -> np.ndarray:
    """Rasteriza un emoji a un vector en `[0,1]`.

    Grises (`color=False`): `(size·size,)` con tinta=1, fondo=0. Color (`color=True`): RGB
    aplanado en orden canal-mayor `(3·size·size,)` con los colores reales sobre fondo blanco
    (=1), compatible con `ConvVAE(in_channels=3)` (que reshapea a `(N, 3, size, size)`).
    """
    from PIL import Image, ImageDraw

    font = font or _load_font(font_path)
    img = Image.new("RGBA", (_CANVAS, _CANVAS), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # anchor='mm': el glifo se dibuja centrado en el punto medio del lienzo.
    draw.text((_CANVAS // 2, _CANVAS // 2), ch, font=font,
              embedded_color=True, anchor="mm")

    # Bounding-box de la tinta (por luminancia), compartido por ambos modos.
    ink = 255 - np.asarray(img.convert("L"), dtype=np.float64)
    ys, xs = np.where(ink > _INK_THRESHOLD)
    if xs.size:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    else:
        y0, y1, x0, x1 = 0, ink.shape[0], 0, ink.shape[1]

    if color:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64)  # (C, C, 3), fondo blanco=255
        crop = rgb[y0:y1, x0:x1, :]
        h, w = crop.shape[:2]
        side = max(h, w)
        square = np.full((side, side, 3), 255.0)  # encuadre cuadrado sobre fondo blanco
        oy, ox = (side - h) // 2, (side - w) // 2
        square[oy:oy + h, ox:ox + w, :] = crop
        small = Image.fromarray(square.astype(np.uint8)).resize((size, size), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float64) / 255.0  # (size, size, 3)
        return arr.transpose(2, 0, 1).reshape(-1)  # canal-mayor: (3·size·size,)

    # Grises: recorte de la tinta sobre fondo negro (=0).
    crop = ink[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float64)
    square[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop
    small = Image.fromarray(square.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return (np.asarray(small, dtype=np.float64) / 255.0).reshape(-1)


def load_emojis(size: int = 28, subset: list[int] | None = None,
                font_path: str | Path = DEFAULT_FONT,
                color: bool = False) -> tuple[np.ndarray, list[str]]:
    """Carga el dataset de emojis. `X` de forma `(N, size·size)` o, con `color=True`,
    `(N, 3·size·size)` (RGB canal-mayor)."""
    font = _load_font(font_path)
    items = EMOJIS if subset is None else [EMOJIS[i] for i in subset]
    X = np.stack([render_emoji(ch, size, font=font, color=color) for ch, _ in items])
    labels = [label for _, label in items]
    return X, labels


# Rangos Unicode con la mayoría de los emojis pictográficos de NotoColorEmoji.
_EMOJI_RANGES = [
    (0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF), (0x2600, 0x26FF), (0x2700, 0x27BF), (0x1FA70, 0x1FAFF),
]


def load_many_emojis(size: int = 28, max_n: int | None = None, color: bool = False,
                     font_path: str | Path = DEFAULT_FONT) -> tuple[np.ndarray, list[str]]:
    """Carga muchos glifos de emoji **distintos** rasterizando los rangos Unicode soportados.

    Filtra codepoints no soportados (glifo vacío) y duplicados (mismo rasterizado). Para el
    experimento de "más datos": permite pasar de 32 a ~1300 imágenes distintas. `max_n` corta
    la cantidad. Etiquetas = codepoint en hex. La salida es determinista (orden de los rangos).
    """
    font = _load_font(font_path)
    X: list[np.ndarray] = []
    labels: list[str] = []
    seen: set[int] = set()
    for a, b in _EMOJI_RANGES:
        for cp in range(a, b + 1):
            ch = chr(cp)
            try:
                gray = render_emoji(ch, size, font=font)  # gris: filtra/deduplica
            except Exception:
                continue
            if gray.mean() <= 0.01:  # glifo vacío (codepoint no soportado)
                continue
            key = hash(np.round(gray, 3).tobytes())
            if key in seen:
                continue
            seen.add(key)
            X.append(render_emoji(ch, size, font=font, color=True) if color else gray)
            labels.append(f"U+{cp:04X}")
            if max_n is not None and len(X) >= max_n:
                return np.stack(X), labels
    return np.stack(X), labels


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
