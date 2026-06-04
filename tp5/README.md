# TP5 — Ejercicio 1: Autoencoder

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
| `denoising`       | `35-25-15-8-2`          | tanh / sigmoid  | xavier_normal  | adam   | bce  |

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
  denoising, `denoising.png` (ruidoso → reconstruido → limpio) + `denoising_sweep.csv`.

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
barriendo niveles de ruido (`0.05/0.1/0.2/0.3`), reportando el error de reconstrucción
contra los patrones limpios en `denoising_sweep.csv`.

## Tests

```bash
uv run pytest -q
```

Cubre: parseo de `font.h` y desempaquetado de bits, gradient-check numérico (diff
relativa < 1e-5) de la backprop para BCE y MSE, construcción por espejo, descenso de la
pérdida con Adam, round-trip de pack/unpack de pesos, validación de config y denoising.

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
configs/        # base_adam, base_lbfgs, deep, wide_relu, naive_init, denoising
```
