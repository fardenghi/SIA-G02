# Compresor de Imagenes Evolutivo

> Aproximacion visual de imagenes con Algoritmos Geneticos (AG) usando formas traslucidas sobre fondo blanco.

Este proyecto corresponde al TP2 de SIA (ITBA). El motor evoluciona individuos compuestos por una cantidad fija de formas (`triangle` o `ellipse`) para aproximar una imagen objetivo.

## Cumplimiento de la consigna

- Seleccion implementada: `elite`, `roulette`, `universal`, `boltzmann`, `tournament`, `probabilistic_tournament`, `rank` (`ranking` como alias).
- Supervivencia implementada: `additive` y `exclusive`.
- Cruzas obligatorias implementadas: `single_point`, `two_point`, `uniform`, `annular` (ademas incluye `spatial_zindex` y `arithmetic` via YAML).
- Mutaciones implementadas: `single_gene`, `limited_multigen`, `uniform_multigen`, `complete`, `error_map_guided`.
- Exporta resultados visuales y metricas por generacion (`metrics.csv`) con pandas.

## Instalacion

Requisitos:

- Python 3.10+
- `uv`

Instalacion base:

```bash
git clone <URL_DEL_REPOSITORIO>
cd tp2
uv sync --dev
```

Instalacion con soporte GPU (opcional):

```bash
uv sync --dev --extra gpu
```

Verificacion rapida:

```bash
uv run --dev pytest tests/ -v
```

## Inicio rapido

Ejecucion minima:

```bash
uv run main.py --image input/mi_imagen.png --triangles 50
```

Modo elipses:

```bash
uv run main.py --image input/mi_imagen.png --triangles 50 --shape ellipse
```

Ejemplo completo:

```bash
uv run main.py \
  --image input/foto.png \
  --triangles 100 \
  --shape triangle \
  --population 50 \
  --generations 5000 \
  --selection tournament \
  --crossover uniform \
  --mutation uniform_multigen \
  --output output/mi_experimento
```

## Interfaz de linea de comandos (main.py)

Los argumentos de CLI pisan los valores de `config.yaml`.

Argumentos principales:

- `--image`, `-i`: imagen objetivo (requerido).
- `--triangles`, `-t`: cantidad de genes.
- `--shape`: `triangle` | `ellipse`.
- `--population`, `-p`: tamano de poblacion.
- `--generations`, `-g`: generaciones maximas.
- `--output`, `-o`: directorio de salida.

Operadores geneticos:

- `--selection`: `elite`, `tournament`, `probabilistic_tournament`, `roulette`, `universal`, `boltzmann`, `rank`, `ranking`.
- `--crossover`: `single_point`, `two_point`, `uniform`, `annular`.
- `--mutation`: `single_gene`, `limited_multigen`, `uniform_multigen`, `complete`, `error_map_guided`.
- `--mutation-rate`, `-m`: probabilidad de mutar individuo.
- `--guided-ratio`: fraccion guiada para `error_map_guided`.
- `--field-probability`: probabilidad de mutar cada valor interno del gen.

Supervivencia y fitness:

- `--survival`: `additive` | `exclusive`.
- `--offspring-ratio`: K = N * ratio.
- `--elite-count`: cantidad de elites clonados sin modificar.
- `--fitness`: `linear`, `rmse`, `inverse_normalized`, `exponential`, `inverse_mse`, `detail_weighted`, `composite`.
- `--fitness-scale`: escala para `exponential`.

Renderizado, salida e islas:

- `--renderer`: `cpu` | `gpu`.
- `--save-interval`: guarda `gen_XXXXX.png` cada N generaciones.
- `--max-size`: reescala imagen de entrada si excede este tamano.
- `--quiet`, `-q`: menos logs.
- `--islands`: cantidad de islas (si >1 activa IMGA).
- `--migration-size`: tamano de migracion.
- `--migration-interval`: intervalo de migracion.

Ayuda completa:

```bash
uv run main.py --help
```

## Configuracion YAML

Archivo por defecto: `config.yaml`.

El proyecto incluye, entre otras, estas secciones:

- `image`: ruta de imagen objetivo.
- `genotype`: `shape_type`, `num_triangles`, `alpha_min`, `alpha_max`.
- `genetic`: poblacion, generaciones, parada temprana, curriculum learning (`transition_methods`), `seed_ratio`.
- `fitness`: metodo y parametros (`exponential_scale`, `detail_weight_base`, pesos de `composite`).
- `selection`: metodo y parametros de torneo/boltzmann.
- `crossover`: metodo, probabilidad y modo por fases (`phased`).
- `mutation`: metodo, probabilidades, deltas y sigma adaptativo.
- `survival`: metodo (`additive`/`exclusive`), seleccion de sobrevivientes, `offspring_ratio`, `elite_count`.
- `output`: guardado de imagenes, JSON, CSV y grafico.
- `island`: modelo IMGA (anillo, migracion, paralelo).
- `rendering`: backend (`cpu` o `gpu`).

Notas:

- `fitness.method` soporta tambien `ssim` y `edge_loss` via YAML.
- `crossover.method` soporta tambien `spatial_zindex` y `arithmetic` via YAML.
- Si `rendering.backend: gpu` y no esta `moderngl`, el sistema hace fallback a CPU con warning.

## Modelo del individuo y fitness

- Un individuo es una lista ordenada de genes geometricos (el orden define Z-index).
- Cada corrida usa una sola familia de formas por individuo: `triangle` o `ellipse`.
- Triangulo: 3 vertices normalizados + color RGBA.
- Elipse: centro, radios, angulo y color RGBA.
- El fitness maximiza similitud entre imagen renderizada y objetivo. El metodo por defecto es `linear` (`1 - NMSE`).

## Salidas generadas

En `output/` se generan (segun configuracion):

- `result.png`: mejor individuo en resolucion de entrenamiento.
- `result_train_res.png`: alias explicito de entrenamiento.
- `result_high_res.png`: render del mejor individuo en resolucion original.
- `gen_XXXXX.png`: snapshots intermedios (si `save_interval > 0`).
- `seed_preview_gen0.png`: preview de generacion 0 (si `seed_ratio > 0` y no islas).
- `shapes.json`: export estructurado de formas.
- `triangles.json`: export legacy (solo `triangle`).
- `fitness_evolution.png`: grafico de fitness.
- `metrics.csv`: metricas por generacion.
- `shapes.csv` y `triangles.csv` (solo si `output.export_triangles_csv: true`; `triangles.csv` solo en modo `triangle`).

## Script de reconstruccion

Reconstruye una imagen desde `shapes.json`/`triangles.json`:

```bash
uv run scripts/reconstruct.py output/shapes.json -o reconstruida.png -W 256 -H 256
uv run scripts/reconstruct.py output/shapes.json -o reconstruida_hd.png -W 256 -H 256 --scale 2
```

Opciones principales:

- `--output`, `-o`
- `--width`, `-W`
- `--height`, `-H`
- `--scale`, `-s`
- `--renderer {cpu,gpu}`

## Visualizador interactivo

Uso:

```bash
uv run scripts/visualize.py output/mi_experimento
```

Controles:

- `←` / `→`: generacion anterior/siguiente.
- `Espacio`: play/pause.
- `Home` / `End`: inicio/fin.

Exportaciones:

```bash
uv run scripts/visualize.py output/mi_experimento --export-gif evolucion.gif --fps 2
uv run scripts/visualize.py output/mi_experimento --export-video evolucion.mp4 --fps 5
uv run scripts/visualize.py output/mi_experimento --summary resumen.png
uv run scripts/visualize.py output/mi_experimento --no-interactive
```

Notas:

- `--export-video` requiere `opencv-python`.
- `--fps` default en el visualizador: `2`.

## Testing

```bash
uv run --dev pytest tests/ -v
uv run --dev pytest tests/test_engine.py -v
uv run --dev pytest tests/ --cov=src --cov-report=html
```

## Scripts auxiliares

Automatizacion de barridos y analisis:

```bash
./scripts/run_many.sh
./scripts/collect_results.sh
uv run scripts/analyze_runs.py output/run_many
uv run scripts/experiment_configs.py
uv run scripts/reconstruct.py output/shapes.json -o reconstruida.png
uv run scripts/visualize.py output/mi_experimento
```

Notas:

- `scripts/run_many.sh` invoca `scripts/analyze_runs.py` al finalizar.
- Los scripts auxiliares viven en `scripts/` para mantener la raiz enfocada en entrypoints.

## Estructura del proyecto

```text
tp2/
├── main.py
├── config.yaml
├── pyproject.toml
├── scripts/
│   ├── analyze_runs.py
│   ├── run_many.sh
│   ├── collect_results.sh
│   ├── experiment_configs.py
│   ├── reconstruct.py
│   ├── visualize.py
│   └── ...
├── src/
│   ├── fitness/
│   │   └── mse.py
│   ├── genetic/
│   │   ├── engine.py
│   │   ├── island.py
│   │   ├── individual.py
│   │   ├── population.py
│   │   ├── selection.py
│   │   ├── crossover.py
│   │   ├── mutation.py
│   │   └── survival.py
│   ├── rendering/
│   │   ├── canvas.py
│   │   ├── ellipse_canvas.py
│   │   ├── gpu_canvas.py
│   │   ├── gpu_ellipse_canvas.py
│   │   └── factory.py
│   └── utils/
│       ├── config.py
│       ├── export.py
│       └── metrics.py
└── tests/
```
