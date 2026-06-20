# ROADMAP — Iteraciones del VAE (TP5 Ej2)

> **Estado: Iteraciones 1 y 2 COMPLETADAS.** Conclusiones empíricas en
> [RESULTADOS.md](RESULTADOS.md). Resumen: el cuello latente no era el limitante del MLP
> (Iter. 1); la `ConvVAE` quedó implementada y validada pero, para recon BCE de este dataset
> chico, el MLP reconstruye mejor (Iter. 2). Lo de abajo es el plan original.


El Ejercicio 2 (VAE) está **completo y funcionando**: dataset de emojis (28×28 grises),
esquema variacional (reparametrización + KL, ELBO canónico con `β=1`), generación desde el
prior, visualizaciones y diagnósticos del encoder. Suite: **99 tests verdes**.

Este documento traza las **dos próximas iteraciones de mejora**, en orden de prioridad:

1. **Agrandar el espacio latente** (barato, ataca directo la borrosidad, es diagnóstico).
2. **Introducir convoluciones** (caro, mejora las features; recién vale la pena si (1)
   muestra que el cuello latente ya no es el limitante).

> **Por qué este orden:** la borrosidad observada es un límite de información (784 px → 2
> floats). Subir el latente ensancha ese cuello con un cambio de config; la convolución
> mejora la *calidad de las features* pero no ensancha el cuello. Además, (1) es diagnóstico:
> si subir el latente mejora mucho, el cuello era el problema; si apenas cambia, entonces el
> limitante son las features y (2) pasa a justificarse.

## Reglas por iteración (igual que en el Ej2)

1. **No modificar** el código validado del Ej1 ni el núcleo del VAE (`vae.py`) salvo bug; en
   ese caso, consultar antes. Preferir **módulos nuevos** y reutilizar lo que ya funciona
   (`Dense`, `Adam`, `losses`, `VAE.reparameterize`, `VAE.kl_divergence`).
2. Al cerrar cada fase: **agregar tests** que cubran lo nuevo (incluido **gradient-check**
   para toda matemática nueva) y correr la **suite completa** (`uv run pytest -q`).
3. **Commit** conciso, sin co-author.

---

# Iteración 1 — Agrandar el espacio latente

**Objetivo:** explorar `latent_dim ∈ {4, 8, 16, 32}`, medir la mejora de reconstrucción y
observar el **pruning automático** de dimensiones (algunas dims colapsan a KL≈0; el VAE usa
solo las que necesita).

**Punto de partida (ya soportado):** `VAE` ya acepta cualquier `latent_dim` (las cabezas
`μ`/`logσ²` mapean a esa dim y el decoder es el espejo). `vae_config` ya valida `latent_dim
>= 1`. **Entrenar con latente 8/16 NO requiere tocar el núcleo** — solo configs. Lo que falta
es soporte de **visualización para latente > 2** y el barrido comparativo.

### Fase 1.1 — Visualización para latente > 2 (proyección PCA)

Las figuras 2D-específicas (`manifold.png`, `aggregate_posterior.png`, scatter de medias)
solo tienen sentido en 2D. Para `latent_dim > 2`:

- **Nuevo helper** `project_pca(mu, k=2) -> (proj, var_ratio)` en `vae_metrics_viz.py`
  (numpy: centrar, `np.linalg.svd`, proyectar a los `k` componentes principales). "Desde
  cero", coherente con el resto.
- En `vae_cli.run()`: si `latent_dim > 2`, proyectar `μ` con PCA a 2D antes de
  `plot_latent_means` (título "(proyección PCA, var explicada XX%)").
- `manifold.png` y `aggregate_posterior.png`: **omitir** para `latent_dim > 2` (ya están
  guardados con `if vae.latent_dim == 2`). Opcional avanzado: recorrer el plano de los 2
  primeros PCs decodificando (manifold proyectado) — dejar para después.
- **`kl_per_dim.png` se vuelve la figura estrella:** mejorar `plot_kl_per_dim` para (a)
  ordenar las dims por KL descendente, (b) dibujar una línea de umbral de "unidad activa"
  (p. ej. `0.1` nats) y (c) anotar el conteo de dims activas. Agregar helper
  `active_units(vae, X, threshold=0.1) -> int`.

**Tests** (`tests/test_vae_metrics_viz.py`): `project_pca` (shape `(N,2)`, componentes
ordenados por varianza decreciente, reconstrucción aproximada con todos los componentes);
`active_units` (cuenta correcta sobre un caso sintético).

### Fase 1.2 — Configs y barrido comparativo

- **Configs nuevos:** `configs/vae/latent8.json`, `configs/vae/latent16.json` (copiar `base`
  cambiando `architecture.latent_dim`). Mantener `base.json` en `latent_dim: 2` como la
  corrida "didáctica" (con manifold).
- **Script** `scripts/vae_latent_sweep.py` (análogo a `vae_beta_sweep.py`): entrena
  `latent_dim ∈ {2,4,8,16,32}`, y grafica **recon_det vs latent_dim** y **nº de unidades
  activas vs latent_dim**. Nuevo plot `plot_latent_sweep(df, path)` en `vae_metrics_viz.py`
  (dos ejes). Esto responde "¿cuántas dimensiones efectivas piden los emojis?".
- **Tests:** smoke de `plot_latent_sweep`.

### Decisiones / notas

- **β con más dims:** al haber más dimensiones, la KL total puede crecer; mantener `β=1`
  canónico y observar el pruning en `kl_per_dim`. Si la reconstrucción no mejora lo esperado,
  recorrer un mini-barrido de β para el nuevo `latent_dim` (el `recon` baja al subir dims;
  buscar el punto donde varias dims quedan activas sin disparar la KL).
- **Arquitectura:** el encoder termina en 64 ocultas, así que `64 → 16/32` entra sin cambios.
  Para `latent_dim` muy grande (≥64) habría que ensanchar esa última oculta.
- **Hipótesis a validar:** esperá que la reconstrucción mejore claramente de 2→8 y empiece a
  saturar hacia 16–32, con un número de unidades activas que se estabiliza (= dimensionalidad
  intrínseca aproximada del dataset de emojis).

### Done de la Iteración 1

Reconstrucción medida para latente 2/8/16/32, figura del barrido, `kl_per_dim` mostrando
pruning, suite verde, commits por fase. **Conclusión esperada:** si el salto de calidad es
grande, el cuello era el limitante (la conv dará menos); si es chico, seguir con la Iteración 2.

---

# Iteración 2 — Encoder/decoder convolucional

**Objetivo:** reemplazar el MLP del encoder/decoder por capas **convolucionales desde cero**
(numpy, backward analítico, gradient-check), reutilizando toda la matemática variacional ya
validada. Mejora esperada: features con sesgo espacial (bordes/trazos), invariancia a
traslación (potencia el augment) y eficiencia de parámetros → reconstrucciones más nítidas.

**Estrategia de reutilización (clave):** **no tocar `vae.py`**. La `ConvVAE` reutiliza
`VAE.reparameterize` y `VAE.kl_divergence` (son estáticas y puras), las `losses`, `Adam` y el
patrón pack/unpack. Lo nuevo es solo el *plumbing* de capas espaciales y su orquestación.

> Definir una **interfaz uniforme de capa**: `forward(x) -> y` y `backward(dout) -> dx`
> (acumulando `dW`/`db`), igual que `Dense`. Así `ConvVAE` itera capas genéricas
> (Conv/Upsample/Flatten/Dense) con el mismo bucle que `VAE`, y el split variacional en las
> cabezas es idéntico al de `VAE.backward`.

### Fase 2.1 — Capas convolucionales desde cero (`src/autoencoder/conv.py`)

Implementar con backward analítico y **gradient-check individual** (entrada chica, diff rel
< 1e-5) cada una:

- **`Conv2D(in_ch, out_ch, kernel, stride, padding, init, rng)`**
  - `forward(x)`: `x` de forma `(N, C, H, W)` → `(N, O, H', W')`. Implementar vía **im2col**
    (desplegar parches a una matriz y hacer un `@` con `W` reshape `(O, C·k·k)`).
  - `backward(dout)`: `dW` (desde im2col cacheado), `db` (suma sobre N,H',W'), `dx` (vía
    **col2im**). Pesos `W` shape `(O, C, k, k)`, `b` shape `(O,)`.
  - `n_params`, y `W`/`b` planos para pack/unpack (mismo patrón que `Dense`).
- **`Upsample2D(scale=2)`** (nearest): `forward` repite cada píxel `scale×scale`; `backward`
  suma los gradientes de cada bloque (sum-pool). **Recomendado para el decoder en vez de
  conv transpuesta**: backward trivial y **evita los artefactos de tablero de ajedrez**.
  El "upsampling aprendible" se logra con `Upsample2D` seguido de `Conv2D`.
- **`Flatten`**: `(N, C, H, W) <-> (N, C·H·W)`; backward reescala la forma. Glue entre la
  parte conv y las `Dense` (cabezas/decoder).

**Tests** (`tests/test_conv.py`): shapes de forward para distintos stride/padding;
**gradient-check** de `Conv2D` (dW, db, dx), `Upsample2D` y `Flatten`; round-trip de
pack/unpack; un caso conocido a mano (p. ej. un filtro identidad).

### Fase 2.2 — `ConvVAE` (`src/autoencoder/conv_vae.py`)

Arquitectura sugerida (latente `L`, entrada grises 28×28 → reshape `(N,1,28,28)`):

```
Encoder:
  (N,1,28,28)
  Conv(1→16, k3, s2, p1)  -> (N,16,14,14) -> relu
  Conv(16→32, k3, s2, p1) -> (N,32, 7, 7) -> relu
  Flatten                 -> (N, 1568)
  Dense(1568→64)          -> relu
  -> mu_head/logvar_head (Dense 64→L)        [reusa la idea de VAE]

Decoder (espejo):
  z (N,L)
  Dense(L→1568) -> relu -> reshape (N,32,7,7)
  Upsample(×2)  -> (N,32,14,14) -> Conv(32→16,k3,s1,p1) -> relu
  Upsample(×2)  -> (N,16,28,28) -> Conv(16→ 1,k3,s1,p1) -> sigmoid
  Flatten -> (N,784)
```

- Reshape de entrada `(N,784)→(N,1,28,28)` y de salida `(N,1,28,28)→(N,784)` para encajar con
  el dataset y las `losses` (que trabajan sobre vectores planos).
- `forward/backward`: misma estructura que `VAE` (encoder genérico → cabezas → reparam →
  decoder genérico), reusando `VAE.reparameterize`, `VAE.kl_divergence`, y el ELBO canónico
  (`recon ×input_dim + β·KL`). `get_params`/`set_params`/`get_grads` iterando todas las capas.
- Verificar bookkeeping espacial: `28→14→7` (conv stride-2, k3, p1) y `7→14→28` (upsample×2).

**Tests** (`tests/test_conv_vae.py`): shapes encode/decode/forward; **gradient-check del ELBO
completo de la ConvVAE** (eps fijo, bce y mse, diff rel < 1e-5) — el gate de correctitud, igual
que con la MLP-VAE; ELBO decrece con Adam.

### Fase 2.3 — Config, CLI y comparación

- **Config:** extender `vae_config` con un bloque `architecture.kind: "mlp" | "conv"` (default
  `"mlp"`, retrocompatible) y, para `conv`, los canales/kernels. **Validación propia**, sin
  romper los configs MLP existentes.
- **CLI:** `conv_vae_cli.py` con script `autoencoder-vae-conv` (mantener `vae_cli` intacto),
  reutilizando todas las visualizaciones y diagnósticos ya existentes (operan sobre
  `encode`/`decode`/`sample_prior`, que la ConvVAE también expone).
- **Config nuevo:** `configs/vae/conv.json`.
- **Comparación:** correr MLP-VAE vs ConvVAE con mismo `latent_dim` y reportar recon_det, KL,
  nº de parámetros y nitidez visual (reconstruction/samples lado a lado).

**Tests:** carga de config `kind: "conv"` válido/inválido; smoke del CLI conv.

### Decisiones / notas

- **Costo:** Conv2D con im2col es O(memoria) en los parches; con 28×28 y batch chico es
  trivial. Full-batch numpy alcanza.
- **Por qué Upsample+Conv y no ConvTranspose:** mismo poder expresivo para este caso, backward
  mucho más simple y sin checkerboard. Si más adelante se quiere ConvTranspose, agregarla como
  capa extra (con su gradient-check).
- **Sinergia con la Iteración 1:** combinar conv con el `latent_dim` que la Iteración 1 haya
  mostrado como suficiente (no volver a 2 si 8–16 rinde mejor).

### Done de la Iteración 2

`Conv2D`/`Upsample2D`/`Flatten` con gradient-check, `ConvVAE` con gradient-check del ELBO,
config+CLI, comparación MLP vs Conv documentada, suite verde, commits por fase.

---

## Apéndice — mapa rápido del código actual

```
src/autoencoder/
  vae.py             # núcleo VAE: reparam, KL, ELBO canónico, backward (NO tocar)
  vae_train.py       # train_vae (Adam, β-warmup), VAEMetricsTracker
  vae_config.py      # load_vae_config (validación propia)
  vae_cli.py         # entrypoint autoencoder-vae (emojis -> train -> figuras)
  vae_viz.py         # manifold, samples, scatter de medias, interpolación, recon
  vae_metrics_viz.py # curvas, KL por dim, posterior stats, posterior vs prior, β-sweep
  emoji_data.py      # dataset emojis 28x28 grises (+ augment)
  layers.py losses.py optim.py network.py  # piezas del Ej1 reutilizables
configs/vae/         # base.json (latente 2)
scripts/             # vae_beta_sweep.py
tests/               # test_vae*.py, test_emoji_data.py, ...
```

Comandos:

```bash
uv run autoencoder-vae --config configs/vae/base.json   # corrida completa
uv run python scripts/vae_beta_sweep.py                 # ablación de β
uv run pytest -q                                        # suite completa
```

**Convenciones a respetar:** ELBO canónico (`β=1` = VAE estándar); toda matemática nueva va
con gradient-check; reutilizar `VAE.reparameterize`/`VAE.kl_divergence`/`Adam`/`losses`;
módulos nuevos en vez de tocar lo validado.
