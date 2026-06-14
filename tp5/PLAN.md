# PLAN — TP5 Ejercicio 2: Autoencoder Variacional (VAE)

Plan de ejecución para completar el **Ejercicio 2** de la consigna (`docs/SIATP5.pdf`):
extender el autoencoder del Ejercicio 1 a un **VAE generativo** sobre un dataset nuevo.

El Ejercicio 1 (AE básico latente 2D con ≤1 píxel de error, estudio de
arquitecturas/optimización 1a2, scatter latente 1a3, letra nueva 1a4, y denoising AE 1b)
está **completo y validado** (43 tests verdes) — no se modifica.

---

## Decisiones de diseño (cerradas)

- **Dataset (2a):** emojis reales rasterizados con **Pillow + NotoColorEmoji.ttf**
  (`size=109`, `embedded_color=True`) → **28×28 en escala de grises `[0,1]`** (tinta=1,
  fondo=0). Set curado de ~16–32 emojis visualmente distintos. Verificado con spike
  (render OK, 26 ms/10 emojis, rango `[0,1]`).
- **Latente:** **2D** (habilita scatter de medias y manifold generativo, conecta con Ej1).
- **Esquema variacional (2b):** encoder → cabezas lineales `μ` y `logσ²`; truco de
  reparametrización `z = μ + e^{logσ²/2}·ε`, `ε ~ N(0,I)`; pérdida **ELBO =
  reconstrucción + β·KL** con `KL(N(μ,σ)‖N(0,I))`. `β` configurable con **warmup** opcional
  (anti posterior-collapse).
- **Generación (2c):** muestreo del prior `z ~ N(0,I)` → decode; manifold 2D; interpolación.
- **Augmentation (opcional):** rotaciones/traslaciones/escala leves por emoji para enriquecer
  el latente y suavizar la generación. Palanca opcional, activable por config; no bloquea.

## Política de reutilización (acordada)

El VAE **reutiliza** lo ya validado del Ej1 — no se reimplementa lo que funciona:
- `layers.Dense` (forward/backward, activaciones, `init_weights`).
- `optim.Adam`.
- `losses` (`bce`/`mse` value+grad) para el término de reconstrucción.
- El patrón de `get_params`/`set_params`/`get_grads`.

"Desde cero" aplica solo a la **matemática nueva del VAE** (reparametrización, KL y su
backward), que se escribe y se **verifica con gradient-check numérico**.

## Reglas por fase (todas)

1. **No modificar** el código existente del Ej1. Si aparece un bug en ese código,
   **consultar antes** de tocarlo.
2. Código nuevo en **módulos nuevos**. Cambios en `pyproject.toml` (dep Pillow, script
   CLI) y `README.md` son **aditivos** y están permitidos.
3. Al cerrar la fase: **agregar tests** que cubran lo implementado y correr la **suite
   completa** (`uv run pytest -q`) — debe quedar verde.
4. **Commit** con mensaje conciso, **sin co-author**.

---

## Fase 1 — Núcleo VAE (matemática desde cero + gradient-check)

**Objetivo:** clase `VAE` correcta, verificada por gradient-check antes de construir encima.

**Entregables** · `src/autoencoder/vae.py`
- Clase `VAE(encoder_layers, latent_dim=2, activation, output_activation, init, seed)`:
  - Encoder body: lista de `Dense` (ocultas con `activation`).
  - Dos cabezas **lineales**: `μ` y `logσ²` (`Dense` con activación `linear`).
  - Decoder: `Dense` (espejo o explícito), última capa con `output_activation`.
  - `encode(X) -> (mu, logvar)`.
  - `reparameterize(mu, logvar, eps) -> z = mu + exp(0.5·logvar)·eps`.
  - `decode(z) -> x_hat`.
  - `forward(X, rng|eps) -> (x_hat, mu, logvar, z)` con cache para backward.
  - `kl_divergence(mu, logvar)`: value y grads `dKL/dμ`, `dKL/dlogvar`
    (`KL = -0.5·Σ(1 + logσ² − μ² − e^{logσ²})`).
  - `elbo(x_hat, X, mu, logvar, beta)`: recon (`losses`) + `β·KL`.
  - `backward(...)`: recon→decoder→`z`→(μ,logvar) vía reparametrización
    (`dz/dμ=1`, `dz/dlogvar=0.5·e^{0.5·logvar}·ε`) + grads de KL; combina en las cabezas y
    propaga por el encoder body reusando `Dense.backward`.
  - `get_params`/`set_params`/`get_grads`.

**Tests** · `tests/test_vae.py`
- Shapes de `encode`/`reparameterize`/`decode`/`forward`.
- `reparameterize` determinista con `eps` fijo; `z≈μ` cuando `logvar→−∞`.
- `kl_divergence`: valor vs fórmula manual; `KL=0` en `μ=0, logvar=0`; signo de grads.
- **Gradient-check numérico del ELBO completo** (recon + β·KL) con `ε` **fijo**
  (determinismo), diff relativa `< 1e-5`, para `bce` y `mse`.
- ELBO decrece con `Adam` sobre datos sintéticos.

**Cierre:** suite completa verde → commit.

---

## Fase 2 — Dataset de emojis (2a)

**Objetivo:** dataset reproducible de emojis 28×28 en grises.

**Entregables**
- `pyproject.toml`: agregar `pillow` a `dependencies` (aditivo).
- `src/autoencoder/emoji_data.py`:
  - Constante con la lista curada de emojis (+ etiquetas).
  - `render_emoji(ch, size=28, font_path=...) -> np.ndarray (size·size,)` en `[0,1]`.
  - `load_emojis(size=28, subset=None) -> (X, labels)`.
  - (Opcional) `augment(X, ...)`: rotaciones/traslaciones/escala leves.
  - Error claro si falta la fuente.

**Tests** · `tests/test_emoji_data.py`
- Shape `(N, size·size)`, rango `[0,1]`, determinismo (dos renders idénticos).
- Labels alineados con filas; `subset` correcto.
- `pytest.skip` elegante si la fuente no existe (CI sin fuente no rompe).

**Cierre:** suite completa verde → commit.

---

## Fase 3 — Entrenamiento, config y CLI del VAE (2b)

**Objetivo:** entrenar el VAE end-to-end por archivo de config, sin tocar código del Ej1.

**Entregables**
- `src/autoencoder/vae_train.py`:
  - `train_vae(vae, X, epochs, lr, beta, beta_warmup, seed, tracker, ...) -> metrics`:
    full-batch con `optim.Adam`, muestreo de `ε` por época, registro de
    `recon_loss`/`kl`/`elbo`. KL **warmup** opcional (β: 0→target en N épocas).
  - Tracker de métricas del VAE (nuevo; no toca `MetricsTracker`).
- `src/autoencoder/vae_config.py`: dataclasses + `load_vae_config` con validación propia
  (independiente de `config.py`): secciones `data` (size, subset, augment), `architecture`
  (encoder_layers, latent_dim, activation, output_activation, init), `training`
  (epochs, lr, beta, beta_warmup, seed), `output`.
- `src/autoencoder/vae_cli.py`: `run(config)` + `main()`.
- `pyproject.toml`: script `autoencoder-vae = "autoencoder.vae_cli:main"` (aditivo).
- `configs/vae/*.json`: al menos una config base ejecutable.

**Tests** · `tests/test_vae_train.py`, `tests/test_vae_config.py`
- `train_vae` baja el ELBO; `KL > 0`; shapes de métricas.
- Schedule de `beta_warmup` correcto.
- Config válido carga; inválido levanta `ConfigError` (claves/valores fuera de dominio).

**Cierre:** suite completa verde → commit.

---

## Fase 4 — Generación y visualización (2c)

**Objetivo:** generar muestras nuevas y visualizar el VAE; documentar.

**Entregables**
- `src/autoencoder/vae_viz.py`:
  - `sample_prior(rng, n) -> z` y `generate(vae, z) -> x_hat`.
  - `plot_latent_means(mu, labels)`: scatter 2D de las medias.
  - `plot_latent_manifold(vae, n, span)`: grid de `z` en `[-span, span]²` decodificado
    (mosaico — la viz generativa clásica del VAE 2D).
  - `plot_samples(vae, rng, n)`: muestras nuevas desde el prior (**2c**).
  - `plot_interpolation(vae, x0, x1, steps)`: interpolación en el latente.
  - `plot_reconstruction_gray(X, X_hat, labels)`: entrada vs reconstrucción (grises).
- Integración en `vae_cli`: generar todas las figuras + guardar muestras.

**Tests** · `tests/test_vae_viz.py`
- Smoke: cada figura se genera sin error y guarda archivo.
- `sample_prior` determinista con seed; shapes de muestreo/generación.

**Documentación** · `README.md` (sección nueva, aditiva)
- Ejercicio 2 (VAE): dataset de emojis (2a), esquema variacional reparam+KL (2b),
  generación de muestras nuevas (2c), cómo correr `autoencoder-vae`.

**Cierre:** suite completa verde → commit.

---

## Gate de correctitud

El **gradient-check del ELBO en la Fase 1** es la verificación dura de que la matemática
del VAE (reparametrización + KL + backward) es correcta antes de construir entrenamiento,
config y visualización encima. Sin ese check verde, no se avanza a la Fase 2.

## Estado de validación inicial

- Baseline Ej1: `uv run pytest -q` → **43 passed**.
- Spike de rasterizado de emojis (Pillow + NotoColorEmoji → 28×28 grises): OK.
- Entorno: Pillow 12.2 disponible; `NotoColorEmoji.ttf` presente.
