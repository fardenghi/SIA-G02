"""VAE convolucional (Ej2, Iteración 2): encoder/decoder espaciales con cabezas variacionales.

**No toca `vae.py`.** Reutiliza la matemática variacional ya validada de `VAE`
(`reparameterize`, `kl_divergence`, el ELBO canónico `recon·input_dim + β·KL`), las `losses`,
`Adam` y el patrón pack/unpack. Lo nuevo es solo el *plumbing* de capas espaciales
(`conv.Conv2D`/`Upsample2D`/`Flatten`/`Reshape`) y su orquestación.

Expone la **misma interfaz que `VAE`** (`encode`/`decode`/`forward`/`elbo`/`backward`/
`sample_prior`/`get_params`/`set_params`/`get_grads`/`latent_dim`), así que `train_vae` y todas
las visualizaciones/diagnósticos funcionan sin cambios. La diferencia con `VAE`: el encoder y
el decoder son listas de capas genéricas, y la entrada/salida se reshapea entre vector plano
`(N, D)` (que esperan el dataset y las `losses`) y mapa espacial `(N, C, H, W)`.

Arquitectura (entrada grises `size×size`, `n = len(conv_channels)` downsamples stride-2):

    Encoder:  (N,1,S,S) → [Conv(s2) → act]×n → Flatten → Dense→act → μ/logσ² (Dense→L)
    Decoder:  z(N,L) → Dense→act → Reshape → [Upsample×2 → Conv(s1) → act]×n → Flatten → (N,D)

El último Conv del decoder usa `output_activation` (sigmoid). El bookkeeping espacial se
verifica en construcción: `S → S/2 → … → S/2ⁿ` (encoder) y el espejo `×2` (decoder) debe
recuperar `S`.
"""

from __future__ import annotations

import numpy as np

from .conv import Conv2D, Flatten, Reshape, Upsample2D, conv_out_size
from .layers import Dense
from .losses import get_loss
from .vae import VAE


class ConvVAE:
    def __init__(
        self,
        size: int,
        latent_dim: int = 2,
        conv_channels: list[int] | None = None,
        dense_hidden: int = 64,
        in_channels: int = 1,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        init: str = "he_normal",
        loss: str = "bce",
        seed: int | None = None,
    ):
        if latent_dim < 1:
            raise ValueError("latent_dim debe ser >= 1")
        conv_channels = list(conv_channels) if conv_channels is not None else [16, 32]
        if len(conv_channels) < 1 or not all(isinstance(c, int) and c > 0 for c in conv_channels):
            raise ValueError("conv_channels debe ser lista no vacía de enteros > 0")

        self.size = size
        self.in_channels = in_channels
        self.input_dim = in_channels * size * size  # dim del vector plano (= columnas de X)
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels
        self.dense_hidden = dense_hidden
        self.activation = activation
        self.output_activation = output_activation
        self.init = init
        self.loss = loss
        self._loss_value, self._loss_grad = get_loss(loss)

        # Bookkeeping espacial: alto/ancho tras cada conv stride-2 (k3, p1).
        n_down = len(conv_channels)
        feat = size
        for _ in range(n_down):
            feat = conv_out_size(feat, 3, 2, 1)
        self.feat_h = feat
        self.feat_ch = conv_channels[-1]
        if self.feat_h * (2 ** n_down) != size:
            raise ValueError(
                f"size={size} no es compatible con {n_down} downsamples: el decoder "
                f"(upsample ×2 por capa) produce {self.feat_h * (2 ** n_down)}≠{size}. "
                "Usá un size tal que size/2ⁿ·2ⁿ == size (p. ej. 28 u 8 con 2 capas).")
        flat_dim = self.feat_ch * self.feat_h * self.feat_h

        rng = np.random.default_rng(seed)

        # --- Encoder: convs stride-2 → flatten → dense ---
        ch_seq = [in_channels] + conv_channels  # p. ej. [1, 16, 32]
        self.encoder_layers: list = []
        for cin, cout in zip(ch_seq[:-1], ch_seq[1:]):
            self.encoder_layers.append(Conv2D(cin, cout, 3, 2, 1, activation, init, rng))
        self.encoder_layers.append(Flatten())
        self.encoder_layers.append(Dense(flat_dim, dense_hidden, activation, init, rng))

        # --- Cabezas variacionales (lineales, idénticas a VAE) ---
        self.mu_head = Dense(dense_hidden, latent_dim, "linear", init, rng)
        self.logvar_head = Dense(dense_hidden, latent_dim, "linear", init, rng)

        # --- Decoder (espejo): dense → reshape → [upsample → conv]×n → flatten ---
        self.decoder_layers: list = []
        self.decoder_layers.append(Dense(latent_dim, flat_dim, activation, init, rng))
        self.decoder_layers.append(Reshape((self.feat_ch, self.feat_h, self.feat_h)))
        dec_ch = list(reversed(ch_seq))  # p. ej. [32, 16, 1]
        dec_pairs = list(zip(dec_ch[:-1], dec_ch[1:]))
        for idx, (cin, cout) in enumerate(dec_pairs):
            is_last = idx == len(dec_pairs) - 1
            act = output_activation if is_last else activation
            self.decoder_layers.append(Upsample2D(2))
            self.decoder_layers.append(Conv2D(cin, cout, 3, 1, 1, act, init, rng))
        self.decoder_layers.append(Flatten())  # (N, C, S, S) → (N, D)

        self._cache: dict | None = None

    # ---- propiedades / pack-unpack ------------------------------------- #
    @property
    def layers(self) -> list:
        return self.encoder_layers + [self.mu_head, self.logvar_head] + self.decoder_layers

    def _param_layers(self) -> list:
        # Solo capas con pesos (Conv2D, Dense); Flatten/Reshape/Upsample no tienen.
        return [layer for layer in self.layers if hasattr(layer, "W")]

    @property
    def n_params(self) -> int:
        return sum(layer.n_params for layer in self.layers)

    def get_params(self) -> np.ndarray:
        chunks = []
        for layer in self._param_layers():
            chunks.append(layer.W.reshape(-1))
            chunks.append(layer.b.reshape(-1))
        return np.concatenate(chunks)

    def set_params(self, vector: np.ndarray) -> None:
        offset = 0
        for layer in self._param_layers():
            w_size = layer.W.size
            layer.W = vector[offset:offset + w_size].reshape(layer.W.shape)
            offset += w_size
            b_size = layer.b.size
            layer.b = vector[offset:offset + b_size].reshape(layer.b.shape)
            offset += b_size
        if offset != vector.size:
            raise ValueError(
                f"Tamaño de vector incorrecto: {vector.size}, se usaron {offset}")

    def get_grads(self) -> np.ndarray:
        chunks = []
        for layer in self._param_layers():
            chunks.append(layer.dW.reshape(-1))
            chunks.append(layer.db.reshape(-1))
        return np.concatenate(chunks)

    # ---- forward -------------------------------------------------------- #
    def _to_spatial(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], self.in_channels, self.size, self.size)

    def encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self._to_spatial(X)
        for layer in self.encoder_layers:
            a = layer.forward(a)
        return self.mu_head.forward(a), self.logvar_head.forward(a)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        a = Z
        for layer in self.decoder_layers:
            a = layer.forward(a)
        return a  # (N, D), plano para las losses

    # Reutiliza la matemática variacional validada del VAE.
    reparameterize = staticmethod(VAE.reparameterize)
    kl_divergence = staticmethod(VAE.kl_divergence)

    def forward(self, X, eps=None, rng=None):
        mu, logvar = self.encode(X)
        if eps is None:
            rng = rng or np.random.default_rng()
            eps = rng.standard_normal(mu.shape)
        z = self.reparameterize(mu, logvar, eps)
        x_hat = self.decode(z)
        self._cache = {"X": X, "x_hat": x_hat, "mu": mu, "logvar": logvar,
                       "z": z, "eps": eps}
        return x_hat, mu, logvar, z

    def elbo(self, beta: float = 1.0) -> tuple[float, float, float]:
        if self._cache is None:
            raise RuntimeError("Llamá a forward(X) antes de elbo().")
        c = self._cache
        recon = self.input_dim * self._loss_value(c["x_hat"], c["X"])
        kl = self.kl_divergence(c["mu"], c["logvar"])
        return recon + beta * kl, recon, kl

    # ---- backward ------------------------------------------------------- #
    def backward(self, beta: float = 1.0) -> None:
        if self._cache is None:
            raise RuntimeError("Llamá a forward(X) antes de backward().")
        c = self._cache
        X, x_hat = c["X"], c["x_hat"]
        mu, logvar, eps = c["mu"], c["logvar"], c["eps"]
        n = X.shape[0]

        # Reconstrucción: dL/dx_hat → backprop por el decoder (genérico) → dL_recon/dz.
        grad = self.input_dim * self._loss_grad(x_hat, X)
        for layer in reversed(self.decoder_layers):
            grad = layer.backward(grad)
        dz = grad  # (N, L)

        # Reparametrización + KL (idéntico a VAE.backward).
        std = np.exp(0.5 * logvar)
        dmu = dz + beta * (mu / n)
        dlogvar = dz * 0.5 * std * eps + beta * (0.5 * (np.exp(logvar) - 1.0) / n)

        # Cabezas (suman sobre el mismo último oculto) → backprop por el encoder.
        dh = self.mu_head.backward(dmu) + self.logvar_head.backward(dlogvar)
        grad = dh
        for layer in reversed(self.encoder_layers):
            grad = layer.backward(grad)

    # ---- generación ----------------------------------------------------- #
    def sample_prior(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.standard_normal((n, self.latent_dim))

    def generate(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return self.decode(self.sample_prior(n, rng))
