"""Capas espaciales desde cero para el encoder/decoder convolucional del VAE (Ej2, It.2).

Interfaz uniforme idéntica a la de `layers.Dense`: cada capa expone `forward(x) -> y` y
`backward(dout) -> dx`; las capas con parámetros acumulan `dW`/`db` y exponen `W`/`b`/`n_params`
para el pack/unpack plano. Así `ConvVAE` itera capas genéricas (Conv/Upsample/Flatten/Reshape/
Dense) con el mismo bucle que `VAE`.

Toda la matemática nueva (Conv2D vía im2col/col2im, Upsample2D nearest) va con gradient-check
numérico en `tests/test_conv.py`.

Convención de tensores espaciales: `(N, C, H, W)` (batch, canales, alto, ancho).
"""

from __future__ import annotations

import numpy as np

from .layers import get_activation, init_weights


def conv_out_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
    """Tamaño espacial de salida de una convolución `(H + 2p - k)//s + 1`."""
    return (in_size + 2 * padding - kernel) // stride + 1


class Conv2D:
    """Convolución 2D con activación, forward/backward analítico vía im2col/col2im.

    Pesos `W` de forma `(out_ch, in_ch, k, k)` y sesgo `b` de forma `(out_ch,)`. La activación
    va horneada en la capa (como `Dense`): el backward aplica primero su derivada y luego
    propaga el gradiente convolucional.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
                 padding: int = 0, activation: str = "linear", init: str = "he_normal",
                 rng: np.random.Generator | None = None):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel
        self.stride = stride
        self.padding = padding
        self.activation_name = activation
        self._fwd, self._grad = get_activation(activation)
        rng = rng or np.random.default_rng()
        # init_weights espera (n_in, n_out) con fan_in = in_ch·k·k (correcto para He/Xavier).
        w_flat = init_weights(in_ch * kernel * kernel, out_ch, init, rng)  # (Cin·k·k, O)
        self.W = w_flat.T.reshape(out_ch, in_ch, kernel, kernel)
        self.b = np.zeros(out_ch)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache: dict | None = None

    def _im2col(self, x_pad: np.ndarray, hp: int, wp: int) -> np.ndarray:
        """Despliega los parches de `x_pad` a una matriz `(N·hp·wp, C·k·k)`."""
        n, c = x_pad.shape[0], x_pad.shape[1]
        k, s = self.k, self.stride
        cols = np.empty((n, c, k, k, hp, wp), dtype=x_pad.dtype)
        for a in range(k):
            a_end = a + s * hp
            for bcol in range(k):
                b_end = bcol + s * wp
                cols[:, :, a, bcol, :, :] = x_pad[:, :, a:a_end:s, bcol:b_end:s]
        return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * hp * wp, c * k * k)

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        k, s, p = self.k, self.stride, self.padding
        hp, wp = conv_out_size(h, k, s, p), conv_out_size(w, k, s, p)
        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        cols_mat = self._im2col(x_pad, hp, wp)            # (N·hp·wp, C·k·k)
        w_row = self.W.reshape(self.out_ch, -1)           # (O, C·k·k)
        z_mat = cols_mat @ w_row.T + self.b               # (N·hp·wp, O)
        z = z_mat.reshape(n, hp, wp, self.out_ch).transpose(0, 3, 1, 2)  # (N, O, hp, wp)
        a_out = self._fwd(z)
        self._cache = {"x_shape": x.shape, "x_pad_shape": x_pad.shape,
                       "cols_mat": cols_mat, "z": z, "a": a_out, "hp": hp, "wp": wp}
        return a_out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        n, ch, h, w = c["x_shape"]
        k, s, p = self.k, self.stride, self.padding
        hp, wp = c["hp"], c["wp"]
        # Derivada de la activación (horneada), luego backward convolucional.
        dz = dout * self._grad(c["z"], c["a"])                       # (N, O, hp, wp)
        dz_mat = dz.transpose(0, 2, 3, 1).reshape(n * hp * wp, self.out_ch)
        # dW (desde im2col cacheado), db (suma sobre N, hp, wp).
        self.dW = (c["cols_mat"].T @ dz_mat).T.reshape(self.W.shape)
        self.db = dz_mat.sum(axis=0)
        # dx vía col2im: scatter-add de los gradientes de cada parche (parches solapados suman).
        w_row = self.W.reshape(self.out_ch, -1)                      # (O, C·k·k)
        dcols = (dz_mat @ w_row).reshape(n, hp, wp, ch, k, k).transpose(0, 3, 4, 5, 1, 2)
        dx_pad = np.zeros(c["x_pad_shape"], dtype=dout.dtype)
        for a in range(k):
            a_end = a + s * hp
            for bcol in range(k):
                b_end = bcol + s * wp
                dx_pad[:, :, a:a_end:s, bcol:b_end:s] += dcols[:, :, a, bcol, :, :]
        return dx_pad[:, :, p:p + h, p:p + w]

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size


class Upsample2D:
    """Upsampling nearest por un factor entero: repite cada píxel `scale×scale`.

    Recomendado para el decoder (en vez de conv transpuesta): el backward es un sum-pool
    trivial y evita los artefactos de tablero de ajedrez. El "upsampling aprendible" se logra
    con `Upsample2D` seguido de `Conv2D`.
    """

    n_params = 0

    def __init__(self, scale: int = 2):
        self.scale = scale
        self._in_shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._in_shape = x.shape
        s = self.scale
        return x.repeat(s, axis=2).repeat(s, axis=3)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, h, w = self._in_shape
        s = self.scale
        # Suma los gradientes de cada bloque scale×scale (transpuesta del nearest-upsample).
        return dout.reshape(n, c, h, s, w, s).sum(axis=(3, 5))


class Flatten:
    """`(N, C, H, W) -> (N, C·H·W)`; el backward restaura la forma espacial cacheada."""

    n_params = 0

    def __init__(self):
        self._in_shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._in_shape)


class Reshape:
    """`(N, ...) -> (N, *shape)`; inverso de `Flatten` para el glue Dense→conv del decoder."""

    n_params = 0

    def __init__(self, shape):
        self.shape = tuple(shape)
        self._in_shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._in_shape = x.shape
        return x.reshape(x.shape[0], *self.shape)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._in_shape)
