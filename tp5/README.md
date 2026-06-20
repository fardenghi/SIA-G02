# TP5 — Autoencoders y VAE

Autoencoder MLP implementado **desde cero** con `numpy` (forward, backprop y gradientes
analíticos propios) para aprender los 32 caracteres de `font/font.h` (5×7 → 35 píxeles
binarios) a través de un espacio latente de **2 dimensiones**, reconstruyendo cada patrón
con **a lo sumo 1 píxel incorrecto**. Todo el experimento se controla por un archivo
JSON, sin tocar código.

## Instalación

Requiere [`uv`](https://docs.astral.sh/uv/). Las dependencias (`numpy`, `scipy`,
`pandas`, `matplotlib`) se instalan solas:

```bash
uv sync --extra dev   # incluye pytest
```

## Uso

```bash
uv run autoencoder --config configs/base_adam.json
```

Esto entrena la red (multi-restart), exporta métricas a CSV y genera las figuras. La
salida por consola informa el `max_pixel_error` del mejor restart y cuántos de los 32
patrones se reconstruyen exactamente.

### Configuraciones incluidas (`configs/`)

| Config            | Arquitectura            | Activación      | Init           | Optim  | Loss |
|-------------------|-------------------------|-----------------|----------------|--------|------|
| `base_adam`       | `35-25-15-8-2`          | tanh / sigmoid  | xavier_normal  | adam   | bce  |
| `base_lbfgs`      | `35-25-15-8-2`          | tanh / sigmoid  | xavier_normal  | lbfgs  | bce  |
| `deep`            | `35-30-20-12-6-2`       | tanh / sigmoid  | xavier_normal  | adam   | bce  |
| `wide_relu`       | `35-20-2`               | relu / sigmoid  | he_normal      | adam   | mse  |
| `naive_init`      | `35-25-15-8-2`          | tanh / sigmoid  | normal         | adam   | bce  |
| `denoising`       | `35-29-22-15-8-2`       | tanh / sigmoid  | xavier_normal  | adam   | bce  |

### Formato del config

```json
{
  "name": "base_adam",
  "data":         { "font_path": "font/font.h", "subset": null },
  "architecture": { "encoder_layers": [35, 25, 15, 8, 2],
                    "activation": "tanh", "output_activation": "sigmoid",
                    "init": "xavier_normal" },
  "training":     { "optimizer": "adam", "loss": "bce", "epochs": 20000,
                    "lr": 1e-3, "restarts": 20, "seed": 42, "log_every": 200,
                    "stop_at": 0 },
  "denoising":    { "enabled": false, "noise_type": "salt_pepper", "level": 0.1,
                    "sweep_levels": [0.05, 0.1, 0.2, 0.3] },
  "output":       { "metrics_csv": "out/base_adam/metrics.csv",
                    "plots_dir": "out/plots" }
}
```

- **`architecture.encoder_layers`**: define el encoder; el decoder se construye como su
  espejo automático (`[35,25,15,8,2]` → decoder `[2,8,15,25,35]`). El primer valor debe
  ser `35` y el último (latente) típicamente `2`.
- **`activation`** ∈ `tanh|relu|sigmoid`; **`output_activation`** ∈ `sigmoid|tanh|linear`.
- **`init`** ∈ `xavier_uniform|xavier_normal|he_uniform|he_normal|normal|uniform`.
- **`optimizer`** ∈ `adam|lbfgs` (L-BFGS-B vía scipy con gradiente propio; **Powell no se
  usa**). **`loss`** ∈ `bce|mse`.
- **`restarts`**: cantidad de reentrenamientos con semillas distintas; se conserva el de
  menor `max_pixel_error`. **`stop_at`**: corta los restarts apenas un modelo alcanza
  `max_pixel_error <= stop_at` (default `0`, es decir 32/32 exacto); poné `null` para
  correr siempre los `restarts` completos (útil para el barrido estadístico del informe).
- Combinaciones subóptimas (p. ej. `bce` con salida no-`sigmoid`, `relu`+`xavier`) emiten
  un **warning** y continúan: permiten estudiar el efecto en el informe.

### Salidas (`out/`)

- `metrics.csv`: `loss`, `max_pixel_error`, `mean_pixel_error` por época (mejor restart).
- `metrics_restarts.csv`: métricas finales de cada restart.
- `plots/<name>/`: `latent_scatter.png` (1a3), `new_letter.png` (1a4, interpolación en el
  latente → decode → umbral), `reconstruction.png` (entrada vs reconstrucción) y, con
  denoising, `denoising_l<nivel>.png` (ruidoso → reconstruido → limpio a cada nivel del
  barrido), `denoising_sweep.png` (curvas de error y % de glifos perfectos vs nivel) +
  `denoising_sweep.csv`.

## Resultados (objetivo: `max_pixel_error <= 1` sobre los 32 patrones)

Con multi-restart (20 semillas, 20000 épocas, `stop_at: null` para forzar las 20 y poder
comparar la robustez), las configuraciones tanh/sigmoid alcanzan **32/32 patrones
exactos** (`max_pixel_error = 0`). En uso normal `stop_at: 0` corta apenas se logra el
32/32, así que para un experimento exitoso suele bastar **1 restart**:

| Config       | Mejor `max_pix` | Restarts con ≤1px | `max_pix` medio | Observación                          |
|--------------|-----------------|-------------------|-----------------|--------------------------------------|
| `base_adam`  | **0** (32/32)   | 17/20             | 0.90            | converge robusto y rápido (~3s/run)  |
| `base_lbfgs` | **0** (32/32)   | 11/20             | 3.35            | más propenso a mínimos locales       |
| `naive_init` | **0** (32/32)   | 16/20             | 0.80            | init `normal`: más varianza entre runs |
| `deep`       | **0** (32/32)   | —                 | —               | red más profunda, también 32/32      |
| `wide_relu`  | 3               | —                 | —               | `35-20-2` relu/mse: cuello demasiado abrupto |

> La **progresión completa de la búsqueda 1a2** (de la config ingenua que no aprende
> hasta las de menor error) está en [`configs/1a2/`](configs/1a2/README.md), con sus
> configs ejecutables y la tabla de robustez por diseño.

**Lectura del barrido (1a2):**

- **Optimizador (Adam vs L-BFGS):** ambos llegan a 32/32, pero L-BFGS-B cae en mínimos
  locales mucho más seguido (sólo 11/20 restarts ≤1px, `max_pix` medio 3.35 vs 0.90 de
  Adam). El multi-restart es lo que garantiza el 32/32 en ambos casos.
- **Inicialización (`xavier_normal` vs `normal`):** con multi-restart ambas alcanzan el
  objetivo; la init `normal` no es catastrófica a esta escala pero introduce más
  dispersión entre restarts (algunos quedan en `max_pix` alto, p. ej. 9), evidenciando
  por qué una init escalada (Xavier) es preferible.
- **Arquitectura:** el cuello `35-20-2` con `relu`/`mse` (`wide_relu`) no logra ≤1px
  (mejor 3) — comprimir 35→2 en un solo paso es demasiado abrupto frente a la rampa
  progresiva `35-25-15-8-2`.

## Denoising Autoencoder (1b)

```bash
uv run autoencoder --config configs/denoising.json
```

Entrena `X̃ → X` (entrada corrompida, objetivo limpio) y evalúa la capacidad de denoising
barriendo niveles de ruido (`0.05/0.1/0.2/0.3`), reportando por nivel el error de píxeles
(`max`/`mean`) y el **% de glifos reconstruidos perfecto** (0 píxeles de error) en
`denoising_sweep.csv` / `denoising_sweep.png`, más los tripletes ruidoso→reconstruido→limpio
`denoising_l<nivel>.png`.

**El espacio latente es de 2 dimensiones**, igual que en 1a. La capacidad para denoisear
**no** se gana ensanchando el cuello, sino con tres palancas a latente 2 fijo:

1. **Corrupción online por época** (`denoising.resample_per_epoch: true`): cada época de
   Adam ve una realización de ruido *fresca* sobre los glifos limpios, así la red aprende
   la *operación* de denoising en vez de memorizar un patrón de ruido fijo. (Vuelve el
   objetivo estocástico → requiere Adam; con `lbfgs` el config falla con un error claro.)
2. **Nivel de ruido mixto** (`denoising.train_level_range: [0.0, 0.3]`): el nivel se
   samplea uniforme por época, así la red es robusta en todo el barrido y no solo cerca de
   un nivel fijo.
3. **Arquitectura profunda hacia el cuello 2D** (`encoder_layers: [35, 29, 22, 15, 8, 2]`):
   los valores intermedios son capas *ocultas*, no el latente — la capacidad viene de la
   profundidad/ancho del encoder y su decoder espejo, con el latente fijo en 2.
4. **Selección del restart por denoising**: con denoising habilitado, la mayoría de los
   restarts reconstruyen el set limpio igual de bien (`max_pix = 0`) pero denoisean
   distinto. En vez de quedarse con el primero, el sistema elige entre ellos el que maximiza
   el **% de glifos perfectos** del barrido (mejora gratis, sin reentrenar).

### Búsqueda de arquitectura (latente 2 fijo)

Comparando arquitecturas con el latente **siempre en 2** (Adam, BCE/sigmoid, schedule
coseno, corrupción online `[0, 0.3]`), una rampa profunda hacia el cuello mejora claramente
el denoising frente al cuello abrupto o la rampa corta:

| Arquitectura (latente 2) | mejor `max_pix` limpio | `mean_pix`@0.3 | % perfectos @0.1 | % perfectos (prom.) |
|--------------------------|:----------------------:|:--------------:|:----------------:|:-------------------:|
| `35-20-2`                | 2                      | 3.43           | 61.9             | 53.4                |
| `35-25-15-8-2`           | 1                      | 2.08           | 86.1             | 78.8                |
| `35-30-15-2`             | 0                      | 2.03           | 95.6             | 86.8                |
| `35-30-20-10-2`          | 0                      | 1.88           | 96.4             | 89.1                |
| **`35-28-20-12-2`**      | **0**                  | **1.78**       | **97.7**         | **89.0**            |

El cuello abrupto `35-20-2` (comprimir 35→2 de golpe) es el peor; agregar profundidad sube
el % de glifos perfectos. La config final lleva esa rampa un paso más allá
(`35-29-22-15-8-2`): más profundidad da más restarts que reconstruyen limpio y mejor
denoising a ruido alto.

### Resultado de `configs/denoising.json` (latente 2)

Con la config final (60000 épocas, 10 restarts, schedule coseno, selección por denoising),
**los 10/10 restarts reconstruyen los 32/32 glifos limpios exactos** (`max_pix = 0`) y el
barrido de denoising da:

| nivel de ruido | `max_pix` | `mean_pix` | % glifos perfectos |
|:--------------:|:---------:|:----------:|:------------------:|
| 0.05           | 0.4       | 0.01       | 99.4               |
| 0.10           | 2.4       | 0.08       | 98.8               |
| 0.20           | 7.4       | 0.28       | 95.6               |
| 0.30           | 11.6      | 0.94       | 85.0               |

A niveles bajos/medios (0.05–0.1) recupera ~99% de los glifos perfecto, y degrada de forma
suave al subir el ruido (85% a 0.3) — muy por encima del comportamiento previo
(`max_pix ~10–23` ya a niveles bajos) **sin tocar la dimensión del latente**.

> **Sobre las épocas:** subir de 15k a ~40–60k mejora el barrido, pero más allá los
> rendimientos son decrecientes — 60k da prácticamente lo mismo que 500k a una fracción del
> costo (~2 min vs ~17 min). Ampliar `train_level_range` por encima de `[0, 0.3]` (el rango
> de evaluación) **no** ayuda: empeora levemente todos los niveles y reduce los restarts que
> reconstruyen limpio. El residuo a 0.3 (≈10 de 35 píxeles invertidos) está cerca del límite
> de información de un latente 2D: con tanto ruido varios glifos se vuelven ambiguos.

## Ejercicio 2 — Autoencoder Variacional (VAE)

Extiende el autoencoder a un **VAE generativo** sobre un dataset nuevo de **emojis**. El VAE
**reutiliza** las piezas validadas del Ej1 (`layers.Dense`, `optim.Adam`, las pérdidas
`bce`/`mse` y el patrón de pack/unpack); lo nuevo, escrito desde cero y verificado por
gradient-check, es la matemática variacional: cabezas `μ`/`logσ²`, reparametrización y KL.

```bash
uv run autoencoder-vae --config configs/vae/base.json
```

### 2a — Dataset de emojis

`emoji_data.py` rasteriza un set curado de 32 emojis desde `NotoColorEmoji.ttf` (Pillow) a
imágenes **28×28 en escala de grises** `[0,1]` (tinta=1, fondo=0), centradas por
bounding-box. Con `data.augment.enabled` se expande el set con copias levemente
rotadas/trasladadas/escaladas, lo que enriquece el latente y suaviza la generación (un set
chico de emojis muy distintos tiende a formar clusters discretos).

### 2b — Esquema variacional

El encoder produce, en vez de un código fijo, una distribución `q(z|x) = N(μ, σ²)` vía dos
**cabezas lineales** (`μ` y `logσ²`). El muestreo usa el **truco de reparametrización**
`z = μ + e^{logσ²/2}·ε`, `ε ~ N(0, I)`, que lo hace diferenciable. La pérdida es el **ELBO
negativo**:

```
L = recon(x, x̂) + β · KL(N(μ,σ) ‖ N(0, I))
```

El término KL regulariza el latente hacia el prior `N(0,I)`, dándole estructura continua
(lo que un AE determinista no garantiza). `β` (con **warmup lineal** opcional, `beta_warmup`)
balancea reconstrucción vs. regularización y mitiga el *posterior collapse*. El backward del
ELBO está verificado por **gradient-check numérico** (diff relativa < 1e-5, BCE y MSE).

### 2c — Generación

Como el latente sigue el prior `N(0,I)`, se generan **muestras nuevas** muestreando
`z ~ N(0,I)` y decodificando (`samples.png`). Con latente 2D se grafica además el **manifold
generativo** (grilla de `z` decodificada, `manifold.png`), el **scatter de medias**
(`latent_means.png`), la **interpolación** entre dos emojis (`interpolation.png`) y la
**reconstrucción** (`reconstruction.png`).

### Diagnósticos del encoder (métricas)

Cada corrida genera además gráficos de las métricas para **validar el encoder**:

- **`training_curves.png`**: ELBO/recon y KL vs época con el warmup de β — valida
  convergencia y que la KL **no colapse a 0**.
- **`kl_per_dim.png`**: KL aportada por cada dimensión latente — detecta dimensiones
  "muertas" (sin usar).
- **`posterior_stats.png`**: histogramas de μ y σ del posterior — sano = σ en (0,1), ni
  colapso (σ→0) ni ignorar la entrada (σ→1, μ→0).
- **`aggregate_posterior.png`**: scatter de los μ con los contornos del prior N(0,I) — si
  q(z) matchea el prior, muestrear de N(0,I) genera bien.

El barrido de β (ablación recon-vs-KL del README) se reproduce con un script aparte (entrena
un modelo por β):

```bash
uv run python scripts/vae_beta_sweep.py
```

### Formato del config del VAE

```json
{
  "name": "base",
  "data":         { "size": 28, "subset": null,
                    "augment": { "enabled": true, "n_aug": 8 } },
  "architecture": { "encoder_layers": [784, 256, 64], "latent_dim": 2,
                    "activation": "relu", "output_activation": "sigmoid",
                    "init": "he_normal" },
  "training":     { "loss": "bce", "epochs": 5000, "lr": 1e-3,
                    "beta": 1.0, "beta_warmup": 1000, "seed": 0 },
  "output":       { "metrics_csv": "out/vae_base/metrics.csv",
                    "plots_dir": "out/plots" }
}
```

- **`architecture.encoder_layers`** es el cuerpo del encoder (incluida la entrada, que debe
  ser `data.size²`); las cabezas `μ`/`logσ²` mapean el último oculto a `latent_dim`, y el
  decoder es el espejo automático. **`latent_dim`** típicamente `2` (habilita el manifold).
- **`training.beta`** ≥ 0 (peso de la KL; `β=1` = VAE canónico, `β>1` = β-VAE más
  regularizado); **`beta_warmup`** sube `β` de 0 al objetivo en esa cantidad de épocas.
  **`loss`** ∈ `bce|mse`.
- **`architecture.kind`** ∈ `mlp|conv` (default `mlp`, retrocompatible). Con `conv` el
  encoder/decoder son convolucionales y se usan `conv_channels` (canales de las convs
  stride-2, p. ej. `[16, 32]`) y `dense_hidden`; `encoder_layers` se ignora. Ver la sección
  de iteraciones para la comparación MLP vs CNN.

### Resultados y el rol de β (posterior collapse)

El balance `β` entre reconstrucción y KL es **el** parámetro crítico del VAE. El ELBO usa la
convención **canónica** ("nats por muestra"): la reconstrucción se **suma** sobre los 784
píxeles y la KL sobre las dims latentes (por eso `recon` se reporta en cientos y la KL en
unidades), de modo que **ambos términos quedan en la misma escala** y `β=1` es el VAE estándar.
`β>1` regulariza más (*β-VAE*); solo un `β` enorme (~784) colapsa el latente (la red lo
ignora, `kl→0`, y reconstruye la "cara promedio" para todo):

| β       | recon (det.) | KL   | qué se observa                                       |
|:-------:|:------------:|:----:|------------------------------------------------------|
| **1.0** | **~313**     | ~6.5 | **default (VAE vanilla)**: recon nítida + muestras diversas |
| 8       | ~319         | 3.2  | más regularizado (latente más compacto)              |
| 16      | ~337         | 1.8  | latente apenas usado                                 |
| ~784    | colapso      | ~0.0 | **posterior collapse**: misma imagen para toda entrada |

Con `β=1` (config `base`, con augment: ELBO≈336, recon≈329, KL≈7), la reconstrucción distingue
clases (p. ej. `cool` conserva los anteojos; `cry`/`rage` su boca), las **muestras del prior**
salen variadas y reconocibles (caras, osos, pandas, frutas, una luna), el **manifold** varía de
forma continua y los **clusters** del scatter de medias tienen sentido. La contrapartida: con
latente 2D y emojis muy distintos en grises, las imágenes son inevitablemente **difusas** — el
límite de información de comprimir 784 píxeles a 2 números. Subir `β` hacia ~8 compacta el
latente (mejor matcheo con el prior) a cambio de algo más de difuminado; si aparece colapso
(`kl≈0`, muestras idénticas), bajá `β`, subí `beta_warmup` o sumá augment.

### Iteraciones de mejora: ¿el cuello es el latente o las *features*? (MLP vs CNN)

La reconstrucción a latente 2 es **difusa**: ¿por comprimir demasiado (cuello latente) o por
*features* pobres del MLP? Dos iteraciones diagnósticas lo responden. La arquitectura se
conmuta **por config** (`architecture.kind: "mlp" | "conv"`), sin tocar código.

#### Iteración 1 — Agrandar el latente

Barrido `latent_dim ∈ {2,4,8,16,32}` (MLP, 32 emojis sin augment, β=1):

| latent_dim | recon_det | unidades activas |
|:----------:|:---------:|:----------------:|
| 2  | 309.65 | 2  |
| 4  | 308.74 | 4  |
| 8  | 308.84 | 8  |
| 16 | 309.10 | 13 |
| 32 | 308.93 | 13 |

**La reconstrucción es plana** (~309 ± 0.5 nats). El modelo activa más dimensiones cuando se
las das (KL y unidades activas suben), pero la calidad de píxeles **no mejora**; las activas
saturan en **~13** (la dimensionalidad intrínseca del set). → el **cuello latente no es el
limitante**; lo son las *features*. Esto motiva probar convoluciones.

Soporte de visualización para `latent_dim > 2`: `active_units(vae, X)` cuenta dims con KL sobre
umbral, `kl_per_dim.png` las ordena y marca las activas, y `latent_means.png` proyecta los `μ` a
2D por **PCA desde cero** (`project_pca`). Barrido: `scripts/vae_latent_sweep.py --kind mlp`.

#### Iteración 2 — Encoder/decoder convolucional

`ConvVAE` (`conv_vae.py`) implementa convoluciones **desde cero** (`conv.py`: `Conv2D` vía
im2col/col2im, `Upsample2D` nearest — sin checkerboard —, `Flatten`/`Reshape`), con cada capa y
el **ELBO completo** verificados por gradient-check. **No toca `vae.py`**: reutiliza
`reparameterize`/`kl_divergence`/`losses` y expone la misma interfaz, así que todas las viz y
diagnósticos funcionan igual.

```bash
uv run autoencoder-vae --config configs/vae/conv.json    # CNN end-to-end (figuras)
uv run python scripts/vae_latent_sweep.py --kind conv    # barrido de latente CNN
```

**Cara a cara a latente 2** (augment on): MLP `recon_det 327.6` vs CNN `341.8` — la CNN **~14
nats peor** y visualmente más borrosa (colapsa casi todos los emojis a un smiley genérico).

**Barrido de latente, MLP vs CNN** (sin augment, 2000 épocas):

| latent | MLP   | CNN   | gap (CNN−MLP) |
|:------:|:-----:|:-----:|:-------------:|
| 2  | 315.7 | 332.4 | +16.7 |
| 4  | 309.3 | 320.6 | +11.3 |
| 8  | 309.6 | 317.4 | +7.7  |
| 16 | 309.9 | 317.3 | +7.4  |
| 32 | 310.3 | 317.9 | +7.6  |

La CNN **nunca cruza** al MLP: el gap cae de +16.7 (latente 2) a ~+7.5 (latente 8) y se **clava**
ahí. Además, en latente 32 el MLP poda a 23 dims activas mientras la CNN mantiene las 32 (KL
mayor): **codifica más y reconstruye peor → el cuello está en el decoder.**

**Prueba de capacidad** (canales del decoder `[16,32]` → `[32,64]`):

| latent | CNN `[16,32]` | CNN `[32,64]` | MLP   | gap restante |
|:------:|:-------------:|:-------------:|:-----:|:------------:|
| 8  | 317.4 | **313.1** | 309.6 | +3.5 |
| 16 | 317.3 | **313.8** | 309.9 | +3.9 |

Duplicar canales **cierra ~la mitad del gap** (+7.5 → +3.7) → el cuello del decoder era
**mixto**: ~mitad **capacidad** (la cierra el ancho) y ~mitad **sesgo estructural** del
upsample-nearest + pesos compartidos (que más canales no arreglan).

**Generación (punto c).** Comparando `samples.png` del prior, el **MLP genera emojis
reconocibles y diversos** (pandas con sus parches, corazones, gatos, smileys) tanto a latente 2
como a latente 8; la **CNN produce manchones incoherentes** salvo cerca de la zona poblada de su
latente — su posterior agregado **no cubre bien el prior `N(0,I)`**, así que la mayoría de los
`z` muestreados caen en zonas que el decoder convoluciona como artefactos. El manifold 2D
confirma un latente del MLP más organizado.

#### Conclusión

En las **tres** dimensiones evaluables —**reconstrucción**, **generación** (punto c) y
**estructura del latente**— y en ambos latentes (2 y 8), el **MLP-VAE es el mejor modelo para
este dataset.** El *prior* convolucional (pesos compartidos, invariancia a traslación,
suavizado) es un **regularizador** que en 32 emojis chicos y centrados **cuesta más capacidad de
memorización de la que aporta**; pagaría en **generalización a datos no vistos o imágenes
naturales**, no en el recon/generación de este set fijo. La `ConvVAE` queda **implementada,
validada y caracterizada**; se elige el MLP.

#### ¿Cambia con otro dataset? Pruebas a color y a mayor resolución

Para chequear si el veredicto depende del dataset, repetimos la comparación con los emojis en
**color** (RGB, 3 canales) a latente 8 sin augment (`scripts/color_emoji_experiment.py`):

| modelo | recon_det | parámetros |
|:------:|:---------:|:----------:|
| MLP | **679** | 1.24M     |
| CNN | 728      | **126K**  |

El **MLP reconstruye casi perfecto** (caras, bocas, lágrimas azules, cachetes rosados); la
**CNN sale borrosa** aunque **acierta el color** y la forma gruesa. En generación, el mismo
patrón que en grises. **El color no dio vuelta el resultado** — al contrario: agregó *más para
memorizar* (3× valores), lo que favorece a la capacidad bruta del MLP, y el color **plano y
uniforme** de los emojis no aporta la textura local que la convolución explotaría. El único
punto a favor de la CNN: logra color + forma gruesa con **10× menos parámetros** (eficiencia, no
calidad). Refuerza que la palanca no es el número de canales sino **el tamaño/centrado del
dataset** (régimen de memorización); el prior convolucional pagaría con más datos o imágenes
naturales, no acá.

Lo mismo subiendo la **resolución** (40×40 en grises, `scripts/resolution_experiment.py`):

| modelo | recon_det/px | parámetros |
|:------:|:------------:|:----------:|
| MLP | 0.386 | 855K     |
| CNN | 0.396 | 244K     |

El **gap por píxel es idéntico al de 28×28** (+0.0102 vs +0.0100): subir la resolución **no mueve
el veredicto** (MLP nítido, CNN borroso; misma historia en generación). Ni el color ni la
resolución cambian el resultado porque tocan la *representación*, no la *naturaleza de la tarea*
(memorizar 32 plantillas centradas). Matiz: la `Dense` del cuello de la CNN también crece con la
resolución (125K→244K), así que su eficiencia de parámetros es real pero más matizada que "la
conv se mantiene chica".

**Más datos** (de 32 a **1294 glifos distintos**, mini-batch; `scripts/more_data_experiment.py`):

| modelo | recon_det/px | parámetros |
|:------:|:------------:|:----------:|
| MLP | 0.397 | 437K     |
| CNN | 0.403 | 125K     |

Esta es **la única palanca que movió la aguja**: el gap por píxel bajó de **+0.0100 (32 emojis) a
+0.0060 (1294)**. Con tantas plantillas el MLP **ya no puede memorizarlas** en un latente 8, así
que pierde su ventaja y las reconstrucciones de ambos se vuelven **comparables** (borrosas); en
**generación**, las muestras de la CNN salen **más suaves y coherentes** (sin el ruido
sal-y-pimienta del MLP). No llega a cruzar a 1294 glifos / latente 8 / 28px, pero la **tendencia
es inequívoca**: más datos → la CNN cierra el gap, tal como predice la teoría (a escala MNIST la
conv gana). Confirma que el factor que ataba el resultado era el **régimen de memorización**, no
la representación. Habilitado por el soporte de **mini-batch** en `train_vae` (`batch_size`), que
desacopla el costo de `N`.

## Tests

```bash
uv run pytest -q
```

Cubre: parseo de `font.h` y desempaquetado de bits, gradient-check numérico (diff
relativa < 1e-5) de la backprop para BCE y MSE, construcción por espejo, descenso de la
pérdida con Adam, round-trip de pack/unpack de pesos, validación de config y denoising.
Para el VAE: reparametrización, KL (valor y gradiente), **gradient-check del ELBO**,
descenso del ELBO con Adam, `beta_schedule`, rasterizado/determinismo del dataset de emojis,
validación del config del VAE y smoke de las visualizaciones.

## Estructura

```
src/autoencoder/
  data.py      # parse font.h -> X (32x35), unpack bits, subset, ruido, glyph utils
  layers.py    # capa densa: forward/backward, activaciones, inicializaciones
  network.py   # autoencoder por espejo: encode/decode/forward/backprop, pack/unpack
  losses.py    # bce/mse forward+grad, validación de coherencia
  optim.py     # Adam propio + wrapper L-BFGS-B (scipy)
  train.py     # loop full-batch, multi-restart, métrica de píxeles, MetricsTracker
  config.py    # carga/validación del JSON
  viz.py       # scatter latente (1a3), letra nueva (1a4), heatmaps, denoising
  cli.py       # entrypoint: --config corre el experimento end-to-end
  vae.py         # VAE: cabezas mu/logvar, reparametrización, KL, backward (Ej2)
  conv.py        # capas espaciales desde cero: Conv2D (im2col/col2im), Upsample2D, Flatten/Reshape
  conv_vae.py    # ConvVAE: encoder/decoder convolucional, reutiliza reparam/KL del VAE (It.2)
  emoji_data.py  # dataset de emojis (Pillow + Noto -> 28x28 grises) (Ej2a)
  vae_train.py   # train_vae full-batch Adam, beta-warmup, tracker (Ej2b)
  vae_config.py  # carga/validación del JSON del VAE (kind: mlp|conv)
  vae_viz.py     # manifold, muestras nuevas, scatter de medias, interpolación (Ej2c)
  vae_metrics_viz.py # diagnósticos del encoder: curvas, KL por dim, PCA, posterior vs prior
  vae_cli.py     # entrypoint: autoencoder-vae corre el VAE (mlp o conv) end-to-end
configs/        # base_adam, base_lbfgs, deep, wide_relu, naive_init, denoising
  vae/          # configs del VAE: base, latent8/16 (mlp), conv (cnn)
scripts/        # vae_beta_sweep.py (β), vae_latent_sweep.py (barrido de latente, --kind mlp|conv)
```
