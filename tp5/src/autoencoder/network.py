"""Autoencoder MLP construido por espejo desde `encoder_layers`.

`encoder_layers = [35, 25, 15, 8, 2]` define el encoder (capas densas 35→25→15→8→2)
y el decoder se construye como su espejo automático (2→8→15→25→35). Encoder y decoder
quedan como listas separables, lo que habilita `encode`/`decode` explícitos y la futura
extensión a VAE.

Convención de backprop: la pérdida entrega `da = dL/d(salida activada)` y la red lo
propaga capa por capa multiplicando por la derivada de cada activación.
"""

from __future__ import annotations

import numpy as np

from .layers import Dense


class Autoencoder:
    def __init__(
        self,
        encoder_layers: list[int],
        activation: str = "tanh",
        output_activation: str = "sigmoid",
        init: str = "xavier_normal",
        seed: int | None = None,
        latent_activation: str | None = None,
    ):
        if len(encoder_layers) < 2:
            raise ValueError("encoder_layers debe tener al menos entrada y latente")
        self.encoder_sizes = list(encoder_layers)
        self.decoder_sizes = list(reversed(encoder_layers))
        self.activation = activation
        self.output_activation = output_activation
        # Activación de la capa latente (cuello). Por defecto sigue a la de ocultas
        # (retrocompatibilidad); el default teórico para un AE es 'linear' (cuello sin
        # no-linealidad: representación sin acotar, conexión con PCA, y la cabeza de
        # media en VAE también es lineal).
        self.latent_activation = latent_activation or activation
        self.init = init
        self.input_dim = encoder_layers[0]
        self.latent_dim = encoder_layers[-1]

        rng = np.random.default_rng(seed)

        # Encoder: ocultas con `activation`; la capa latente (última) con
        # `latent_activation`. El orden de creación de capas no cambia, así que con
        # el default los pesos iniciales son idénticos a los de antes.
        self.encoder: list[Dense] = []
        encoder_pairs = list(zip(self.encoder_sizes[:-1], self.encoder_sizes[1:]))
        for idx, (n_in, n_out) in enumerate(encoder_pairs):
            is_latent = idx == len(encoder_pairs) - 1
            act = self.latent_activation if is_latent else activation
            self.encoder.append(Dense(n_in, n_out, act, init, rng))

        # Decoder: ocultas con `activation`, la última capa con `output_activation`.
        self.decoder: list[Dense] = []
        decoder_pairs = list(zip(self.decoder_sizes[:-1], self.decoder_sizes[1:]))
        for idx, (n_in, n_out) in enumerate(decoder_pairs):
            is_last = idx == len(decoder_pairs) - 1
            act = output_activation if is_last else activation
            self.decoder.append(Dense(n_in, n_out, act, init, rng))

    # ---- propiedades ---------------------------------------------------- #
    @property
    def layers(self) -> list[Dense]:
        return self.encoder + self.decoder

    @property
    def n_params(self) -> int:
        return sum(layer.n_params for layer in self.layers)

    # ---- forward -------------------------------------------------------- #
    def encode(self, X: np.ndarray) -> np.ndarray:
        a = X
        for layer in self.encoder:
            a = layer.forward(a)
        return a

    def decode(self, Z: np.ndarray) -> np.ndarray:
        a = Z
        for layer in self.decoder:
            a = layer.forward(a)
        return a

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(X))

    # ---- backward ------------------------------------------------------- #
    def backward(self, da: np.ndarray) -> None:
        """Propaga `da = dL/d(salida)` por toda la red, llenando dW/db de cada capa."""
        grad = da
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ---- pack / unpack de pesos ---------------------------------------- #
    def get_params(self) -> np.ndarray:
        """Empaqueta todos los pesos y sesgos en un único vector plano."""
        chunks = []
        for layer in self.layers:
            chunks.append(layer.W.reshape(-1))
            chunks.append(layer.b.reshape(-1))
        return np.concatenate(chunks)

    def set_params(self, vector: np.ndarray) -> None:
        """Desempaqueta un vector plano de pesos de vuelta a las capas."""
        offset = 0
        for layer in self.layers:
            w_size = layer.W.size
            layer.W = vector[offset:offset + w_size].reshape(layer.W.shape)
            offset += w_size
            b_size = layer.b.size
            layer.b = vector[offset:offset + b_size].reshape(layer.b.shape)
            offset += b_size
        if offset != vector.size:
            raise ValueError(
                f"Tamaño de vector incorrecto: {vector.size}, se usaron {offset}"
            )

    def get_grads(self) -> np.ndarray:
        """Empaqueta los gradientes acumulados en un vector plano (orden de get_params)."""
        chunks = []
        for layer in self.layers:
            chunks.append(layer.dW.reshape(-1))
            chunks.append(layer.db.reshape(-1))
        return np.concatenate(chunks)
