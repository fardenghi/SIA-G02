"""Autoencoder Variacional (VAE) construido por composición sobre las piezas del Ej1.

Reutiliza la capa densa (`layers.Dense`), las pérdidas de reconstrucción (`losses`,
`bce`/`mse`) y el patrón de pack/unpack de pesos del autoencoder determinista. La
matemática NUEVA y propia del VAE es:

  - dos cabezas lineales que producen `μ` y `logσ²` a partir del último oculto del encoder;
  - el truco de reparametrización `z = μ + e^{logσ²/2}·ε`, con `ε ~ N(0, I)`, que hace
    diferenciable el muestreo;
  - la divergencia `KL(N(μ,σ) ‖ N(0, I))` y su gradiente analítico;
  - el backward que combina el gradiente de reconstrucción (que llega a `z` por el decoder)
    con el de la KL (que llega directo a las cabezas), respetando la reparametrización.

La pérdida total es el ELBO negativo `recon + β·KL` en convención **canónica** ("nats por
muestra"): la reconstrucción se suma sobre los `D` píxeles y la KL sobre las dims latentes,
ambas promediadas por el batch. Así los dos términos quedan en la misma escala y `β=1` es el
VAE estándar (`β>1` regulariza más, *β-VAE*). Como `losses` divide por `y_pred.size` (= N·D,
media por elemento), se **reescala la reconstrucción por `input_dim` (D)** para recuperar la
suma por píxel; la KL se promedia por `N` (suma sobre las dims latentes).

El backward queda verificado por gradient-check numérico (ver `tests/test_vae.py`).
"""

from __future__ import annotations

import numpy as np

from .layers import Dense
from .losses import get_loss


class VAE:
    def __init__(
        self,
        encoder_layers: list[int],
        latent_dim: int = 2,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        init: str = "xavier_normal",
        loss: str = "bce",
        seed: int | None = None,
        decoder_layers: list[int] | None = None,
    ):
        """Construye el VAE.

        `encoder_layers` es el CUERPO del encoder incluyendo la entrada, p. ej.
        `[784, 256, 64]` (entrada 784, dos ocultas 256 y 64). Las cabezas `μ`/`logσ²`
        mapean el último oculto (64) a `latent_dim`. El decoder, por defecto, es el espejo
        `[latent_dim, 64, 256, 784]`; se puede pasar `decoder_layers` explícito (de
        `latent_dim` a la dim de entrada) para un decoder asimétrico.
        """
        if len(encoder_layers) < 1:
            raise ValueError("encoder_layers debe incluir al menos la dim de entrada")
        if latent_dim < 1:
            raise ValueError("latent_dim debe ser >= 1")
        self.encoder_sizes = list(encoder_layers)
        self.input_dim = encoder_layers[0]
        self.latent_dim = latent_dim
        self.last_hidden = encoder_layers[-1]
        self.activation = activation
        self.output_activation = output_activation
        self.init = init
        self.loss = loss
        self._loss_value, self._loss_grad = get_loss(loss)

        self.decoder_sizes = (list(decoder_layers) if decoder_layers is not None
                              else [latent_dim] + list(reversed(encoder_layers)))
        if self.decoder_sizes[0] != latent_dim or self.decoder_sizes[-1] != self.input_dim:
            raise ValueError(
                "decoder_layers debe ir de la dim latente "
                f"({latent_dim}) a la dim de entrada ({self.input_dim}); "
                f"se recibió {self.decoder_sizes}")

        rng = np.random.default_rng(seed)

        # Cuerpo del encoder: capas ocultas con `activation`. Si solo hay entrada
        # (encoder_layers de largo 1), el cuerpo queda vacío y las cabezas leen la entrada.
        self.encoder_body: list[Dense] = []
        for n_in, n_out in zip(self.encoder_sizes[:-1], self.encoder_sizes[1:]):
            self.encoder_body.append(Dense(n_in, n_out, activation, init, rng))

        # Cabezas variacionales: lineales (μ sin acotar; logσ² puede ser negativo).
        self.mu_head = Dense(self.last_hidden, latent_dim, "linear", init, rng)
        self.logvar_head = Dense(self.last_hidden, latent_dim, "linear", init, rng)

        # Decoder: ocultas con `activation`, última capa con `output_activation`.
        self.decoder: list[Dense] = []
        decoder_pairs = list(zip(self.decoder_sizes[:-1], self.decoder_sizes[1:]))
        for idx, (n_in, n_out) in enumerate(decoder_pairs):
            is_last = idx == len(decoder_pairs) - 1
            act = output_activation if is_last else activation
            self.decoder.append(Dense(n_in, n_out, act, init, rng))

        # Cache del último forward (para el backward y el cálculo del ELBO).
        self._cache: dict | None = None

    # ---- propiedades ---------------------------------------------------- #
    @property
    def layers(self) -> list[Dense]:
        return self.encoder_body + [self.mu_head, self.logvar_head] + self.decoder

    @property
    def n_params(self) -> int:
        return sum(layer.n_params for layer in self.layers)

    # ---- forward -------------------------------------------------------- #
    def encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve `(μ, logσ²)` para el batch `X`."""
        a = X
        for layer in self.encoder_body:
            a = layer.forward(a)
        mu = self.mu_head.forward(a)
        logvar = self.logvar_head.forward(a)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """`z = μ + e^{logσ²/2}·ε`. Con `ε` fijo el muestreo es determinista."""
        return mu + np.exp(0.5 * logvar) * eps

    def decode(self, Z: np.ndarray) -> np.ndarray:
        a = Z
        for layer in self.decoder:
            a = layer.forward(a)
        return a

    def forward(
        self,
        X: np.ndarray,
        eps: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward completo con muestreo. Devuelve `(x_hat, μ, logσ², z)` y cachea todo.

        Si `eps` se provee se usa tal cual (determinista, para gradient-check); si no, se
        muestrea `ε ~ N(0, I)` con `rng` (o un rng nuevo).
        """
        mu, logvar = self.encode(X)
        if eps is None:
            rng = rng or np.random.default_rng()
            eps = rng.standard_normal(mu.shape)
        z = self.reparameterize(mu, logvar, eps)
        x_hat = self.decode(z)
        self._cache = {"X": X, "x_hat": x_hat, "mu": mu, "logvar": logvar,
                       "z": z, "eps": eps}
        return x_hat, mu, logvar, z

    # ---- KL y ELBO ------------------------------------------------------ #
    @staticmethod
    def kl_divergence(mu: np.ndarray, logvar: np.ndarray) -> float:
        """`KL(N(μ,σ) ‖ N(0,I))` promediada por muestra (sumada sobre dims latentes).

        `KL = -0.5 · Σ_j (1 + logσ²_j − μ_j² − e^{logσ²_j})`, promediada sobre el batch.
        Es 0 cuando `μ = 0` y `logσ² = 0` (posterior == prior).
        """
        n = mu.shape[0]
        return float(-0.5 * np.sum(1.0 + logvar - mu**2 - np.exp(logvar)) / n)

    def elbo(self, beta: float = 1.0) -> tuple[float, float, float]:
        """Pérdida total del último forward: `(total, recon, kl)` con `total = recon+β·kl`."""
        if self._cache is None:
            raise RuntimeError("Llamá a forward(X) antes de elbo().")
        c = self._cache
        # recon en "nats por muestra" (suma sobre los D píxeles, vía ×input_dim): misma
        # escala que la KL (suma sobre dims latentes) -> β=1 es el ELBO canónico.
        recon = self.input_dim * self._loss_value(c["x_hat"], c["X"])
        kl = self.kl_divergence(c["mu"], c["logvar"])
        return recon + beta * kl, recon, kl

    # ---- backward ------------------------------------------------------- #
    def backward(self, beta: float = 1.0) -> None:
        """Llena dW/db de todas las capas para `∇(recon + β·KL)` del último forward.

        El gradiente de reconstrucción entra por la salida y baja por el decoder hasta `z`;
        de `z` se reparte a `μ` (factor 1) y `logσ²` (factor `0.5·e^{logσ²/2}·ε`). A eso se
        le suma el gradiente de la KL en las cabezas, y se propaga por el cuerpo del encoder.
        """
        if self._cache is None:
            raise RuntimeError("Llamá a forward(X) antes de backward().")
        c = self._cache
        X, x_hat = c["X"], c["x_hat"]
        mu, logvar, eps = c["mu"], c["logvar"], c["eps"]
        n = X.shape[0]

        # Reconstrucción: dL/dx_hat -> backprop por el decoder -> dL_recon/dz.
        # ×input_dim para que la recon esté en la misma escala que la KL (ver elbo()).
        grad = self.input_dim * self._loss_grad(x_hat, X)
        for layer in reversed(self.decoder):
            grad = layer.backward(grad)
        dz = grad  # dL_recon/dz

        # Reparametrización: z = μ + e^{logσ²/2}·ε.
        std = np.exp(0.5 * logvar)
        dmu_recon = dz
        dlogvar_recon = dz * 0.5 * std * eps

        # KL (normalizada por N, consistente con kl_divergence):
        #   dKL/dμ = μ/N ;  dKL/dlogσ² = 0.5·(e^{logσ²} − 1)/N
        dmu_kl = mu / n
        dlogvar_kl = 0.5 * (np.exp(logvar) - 1.0) / n

        dmu = dmu_recon + beta * dmu_kl
        dlogvar = dlogvar_recon + beta * dlogvar_kl

        # Backprop por las cabezas (ambas leen el mismo último oculto -> se suman).
        dx = self.mu_head.backward(dmu) + self.logvar_head.backward(dlogvar)

        # Backprop por el cuerpo del encoder.
        for layer in reversed(self.encoder_body):
            dx = layer.backward(dx)

    # ---- generación ----------------------------------------------------- #
    def sample_prior(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Muestrea `n` códigos latentes del prior `N(0, I)`."""
        rng = rng or np.random.default_rng()
        return rng.standard_normal((n, self.latent_dim))

    def generate(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Genera `n` muestras nuevas: `z ~ N(0,I)` -> decode (Ej2c)."""
        return self.decode(self.sample_prior(n, rng))

    # ---- pack / unpack de pesos ---------------------------------------- #
    def get_params(self) -> np.ndarray:
        chunks = []
        for layer in self.layers:
            chunks.append(layer.W.reshape(-1))
            chunks.append(layer.b.reshape(-1))
        return np.concatenate(chunks)

    def set_params(self, vector: np.ndarray) -> None:
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
                f"Tamaño de vector incorrecto: {vector.size}, se usaron {offset}")

    def get_grads(self) -> np.ndarray:
        chunks = []
        for layer in self.layers:
            chunks.append(layer.dW.reshape(-1))
            chunks.append(layer.db.reshape(-1))
        return np.concatenate(chunks)
