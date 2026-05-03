# TP3 — Perceptrones y MLP

Proyecto organizado por ejercicios. La lógica reutilizable está en `common/`; cada ejercicio tiene sus propios runners, datos y configuraciones.

Todos los comandos de este README se ejecutan desde `tp3/`.

## Setup

```bash
uv sync
```

Para verificar que el entorno quedó bien:

```bash
make test
```

## Estructura

```text
tp3/
├── common/                 # código compartido: perceptrón simple, MLP, loaders, losses, optimizers
├── exercises/              # código específico de cada ejercicio
│   ├── xor/                # validación del MLP con XOR
│   ├── ej1_fraud/          # Ej1: fraude con perceptrón simple
│   ├── ej2_digits/         # Ej2: clasificación con digits.csv
│   └── ej3_more_digits/    # Ej3: clasificación con more_digits.csv
├── configs/                # configuraciones separadas por ejercicio
├── data/                   # datasets separados por ejercicio
├── docs/                   # enunciado, teoría y guías
├── reports/                # resultados versionables
├── outputs/                # modelos, métricas y plots generados (ignorado por git)
├── tests/                  # tests de common y ejercicios
├── Makefile                # comandos principales
└── pyproject.toml          # dependencias y config de pytest
```

## Comandos Principales

```bash
make test
make run-ej1
make run-xor
make run-ej2 EJ2_CONFIG=configs/ej2_digits/baseline.json
make run-ej3 EJ3_CONFIG=configs/ej3_more_digits/historical/best_l2_aug.json
make run-ej3-ensembles                  # entrena todos los configs de configs/ej3_more_digits/ensembles/
make inspect-digit DIGIT_DATASET=test DIGIT_INDEX=0
make clean-ej1
make clean-ej2
make clean-ej3
make clean-outputs
```

`run-ej2` y `run-ej3` también generan automáticamente el plot de métricas
(`outputs/<ej>/metrics/<config>_plot.png`) a partir del CSV exportado por la corrida.

`outputs/` se regenera al correr experimentos. No está pensado para versionarse.

## Código Común

`common/` contiene la implementación compartida por los ejercicios:

| Archivo | Uso |
|---------|-----|
| `activations.py` | `step`, `linear`, `sigmoid`, `tanh`, `relu`, `softmax` y derivadas |
| `simple_perceptron.py` | perceptrón simple usado en Ej1 |
| `mlp.py` | perceptrón multicapa con backprop, early stopping, data augmentation y LR scheduler |
| `layers.py` | capa densa vectorizada con `xavier` o `he` init |
| `losses.py` | MSE y cross entropy |
| `optimizers.py` | `SGD`, `Momentum`, `RMSProp`, `Adam` (con `weight_decay` L2) y schedulers `StepDecay`, `ExponentialDecay`, `AdaptiveLR` |
| `ensemble.py` | promedio de probabilidades sobre varios MLP softmax |
| `datasets.py` | loaders de XOR, dígitos, helpers de one-hot y `k_fold_indices` |
| `metrics.py` | `MetricsTracker` (CSV por corrida) + `binary_confusion`, `precision_recall_f1`, `threshold_sweep`, `confusion_matrix` |

### Opciones del MLP en los configs JSON

Todos los configs usados por XOR/Ej2/Ej3 aceptan estas claves:

| Clave | Valores | Notas |
|-------|---------|-------|
| `architecture` | `[in, h1, ..., out]` | tamaños de capa |
| `activation` | `tanh`, `relu` | activación de capas ocultas |
| `output_activation` | `tanh`, `softmax` | ver combinaciones válidas |
| `loss` | `mse`, `cross_entropy` | combos: `mse`+`tanh` o `cross_entropy`+`softmax` |
| `weight_init` | `xavier`, `he` | `he` recomendado con ReLU |
| `optimizer` | `sgd`, `momentum`, `rmsprop`, `adam` | |
| `lr` | float | learning rate inicial |
| `weight_decay` | float | L2 sobre `W` (no sobre biases). 0 desactiva |
| `epochs`, `batch_size` | int | |
| `val_split` | float | fracción de validación |
| `patience`, `min_delta` | int, float | early stopping (omitir = desactivado) |
| `seed` | int | seed única |
| `seeds` | `[s1, s2, ...]` | si está, entrena un modelo por seed y promedia probs en un Ensemble |
| `data_augmentation` | bool | aug afín por batch sobre imágenes 28×28 |
| `aug_rotation_deg` | float | rango simétrico de rotación en grados |
| `aug_scale_range` | `[min, max]` | escala uniforme |
| `lr_scheduler` | objeto | ver abajo |
| `save_model` | path | guarda `.npz` (sufija `_seedN` si hay `seeds`) |
| `export_metrics` | path | guarda CSV de métricas por época |

**Schedulers** (`lr_scheduler` en el JSON):

```json
{ "type": "step_decay",        "decay_rate": 0.5, "step_size": 40, "lr_min": 1e-6 }
{ "type": "exponential_decay", "decay_rate": 0.99,                  "lr_min": 1e-6 }
{ "type": "adaptive",          "k": 5, "a": 1e-4, "b": 0.1, "lr_min": 1e-6, "lr_max": 1.0 }
```

## Validación XOR

Sirve para comprobar que el MLP y backpropagation funcionan sobre un problema no lineal clásico.

```bash
make run-xor
```

Equivalente directo:

```bash
uv run python -m exercises.xor.run configs/xor/default.json
```

Config:

```text
configs/xor/default.json
```

## Ejercicio 1 — Fraude

Objetivo: entrenar un `TinyModel` basado en perceptrón simple para aproximar la probabilidad de fraude producida por el `BigModel`.

Código:

```text
exercises/ej1_fraud/
```

Datos:

```text
data/ej1_fraud/fraud_dataset.csv
data/ej1_fraud/fraud_dataset_documentation.pdf
```

Target de entrenamiento:

```text
big_model_fraud_probability
```

`flagged_fraud` no se usa para entrenar. Se usa solo al final para elegir el umbral de clasificación, como indica la documentación del dataset.

### Flujo Completo

```text
explore
-> train_linear
-> train_sigmoid
-> train_relu                  # variante con perceptrón ReLU
-> compare_models
-> compare_models_relu
-> train_generalization        # k-fold cross-validation (sigmoid)
-> train_generalization_relu   # k-fold cross-validation (ReLU)
-> threshold_analysis          # barrido de umbrales con threshold_sweep
-> threshold_analysis_relu
```

`train_generalization*` usan `k_fold_indices` de `common/datasets.py` y producen
predicciones out-of-fold; `threshold_analysis*` reutilizan ese OOF para barrer
umbrales y elegir el de máximo F1 con `common.metrics.threshold_sweep`.

Ejecutar todo:

```bash
make run-ej1
```

Limpiar outputs de Ej1:

```bash
make clean-ej1
```

### Ejecución Fase Por Fase

```bash
uv run python -m exercises.ej1_fraud.explore
uv run python -m exercises.ej1_fraud.train_linear
uv run python -m exercises.ej1_fraud.train_sigmoid
uv run python -m exercises.ej1_fraud.train_relu
uv run python -m exercises.ej1_fraud.compare_models
uv run python -m exercises.ej1_fraud.compare_models_relu
uv run python -m exercises.ej1_fraud.train_generalization
uv run python -m exercises.ej1_fraud.train_generalization_relu
uv run python -m exercises.ej1_fraud.threshold_analysis
uv run python -m exercises.ej1_fraud.threshold_analysis_relu
```

### Outputs De Ej1

```text
outputs/ej1_fraud/
├── plots/
├── linear.npz / linear.json
├── sigmoid.npz / sigmoid.json
├── generalization.npz / generalization.json
├── selected_config.json
└── threshold_results.json
```

## Ejercicio 2 — Digits

Objetivo: entrenar y comparar variantes de MLP para clasificar `digits.csv`, evaluando contra `digits_test.csv`.

Código:

```text
exercises/ej2_digits/run.py
```

Datos:

```text
data/ej2_digits/digits.csv
data/ej2_digits/digits_test.csv
```

Ejecutar con la config baseline:

```bash
make run-ej2
```

Ejecutar con una config específica:

```bash
make run-ej2 EJ2_CONFIG=configs/ej2_digits/softmax.json
```

Equivalente directo:

```bash
uv run python -m exercises.ej2_digits.run configs/ej2_digits/softmax.json
```

### Configs Disponibles Para Ej2

`configs/ej2_digits/` contiene los configs canónicos (`baseline`, `adam`, `softmax`,
`momentum`, `sgd_lr_low`, `sgd_lr_high`, `relu_he`, `softmax_aug`, `best`) más
varios sweeps reproducibles:

- `arch_*hidden*.json` — sweep de profundidad y ancho (1 a 5 hidden, 16/32/64/128/256/512 neuronas).
- `lr_*.json` — sweep de learning rate (1e-4, 1e-3, 1e-2) con arquitectura standard y wide.
- `opt_*.json` — sweep cruzado de optimizador (`sgd`/`momentum`/`rmsprop`/`adam`) × lr.

Cada config define arquitectura, función de pérdida, activación de salida, optimizador,
learning rate, epochs, batch size, early stopping, augmentation y paths de outputs.

### Plot De Métricas Por Corrida

`make run-ej2` ya genera `outputs/ej2_digits/metrics/<config>_plot.png`.
Para regenerarlo desde un CSV existente:

```bash
uv run python -m exercises.ej2_digits.plot_metrics \
    --csv outputs/ej2_digits/metrics/softmax.csv \
    --out outputs/ej2_digits/metrics/softmax_plot.png
```

Para promediar métricas de varias corridas (ej. mismo config con seeds distintos):

```bash
uv run python exercises/ej2_digits/plot_ensemble_metrics.py \
    --csvs run1.csv run2.csv run3.csv \
    --out  ensemble_curves.png \
    --label "ensemble 3 seeds"
```

### Inspeccionar Una Imagen De Dígitos

Para ver una muestra como imagen `28x28`:

```bash
make inspect-digit DIGIT_DATASET=test DIGIT_INDEX=0
```

Opciones para `DIGIT_DATASET`:

```text
train   # data/ej2_digits/digits.csv
test    # data/ej2_digits/digits_test.csv
more    # data/ej3_more_digits/more_digits.csv
```

El PNG se guarda en:

```text
outputs/ej2_digits/samples/
```

También se puede ejecutar directamente:

```bash
uv run python -m exercises.ej2_digits.inspect_sample --dataset more --index 20 --show
```

## Ejercicio 3 — More Digits

Objetivo: entrenar MLP con `more_digits.csv`, opcionalmente combinado con `digits.csv`,
para mejorar la generalización contra `digits_test.csv` aplicando regularización
(L2, augmentation, early stopping, schedulers) y ensembles.

Código:

```text
exercises/ej3_more_digits/
├── run.py                       # entrenar 1 config (1 o N seeds)
├── run_parallel.py              # entrenar N configs en paralelo (multiprocessing)
├── eval.py                      # evaluar 1 modelo .npz sobre digits_test
├── eval_all.py                  # evaluar TODOS los modelos en outputs/.../models/
├── ensemble.py                  # ensemble por nombre sobre modelos guardados
├── ensemble_search.py / ensemble_heterogeneo.py
├── interpretability.py          # saliency + occlusion por dígito
├── interpretability_compare.py  # comparar dos modelos sobre el mismo dígito
└── plot_*.py                    # plots de regularización, ensembles, gaps, etc.
```

Datos:

```text
data/ej3_more_digits/more_digits.csv
data/ej2_digits/digits.csv          # usado si combine_datasets=true
data/ej2_digits/digits_test.csv     # test final
```

### Entrenar Un Config

```bash
make run-ej3                                        # usa el default del Makefile
make run-ej3 EJ3_CONFIG=configs/ej3_more_digits/historical/best_l2_aug.json
uv run python -m exercises.ej3_more_digits.run configs/ej3_more_digits/baselines/pure.json
```

### Carpetas De Configs Para Ej3

```text
configs/ej3_more_digits/
├── baselines/         # pure, only_l2, only_aug, only_es, no_aug — para ablations
├── ensembles/
│   ├── aug_variations/        # mismas hyperparams, distinta política de aug
│   ├── diverse_architectures/ # mismas hyperparams, distinta arquitectura
│   └── wd_variations/         # mismas hyperparams, distinto weight_decay
├── historical/        # mejores configs encontrados (best, best_l2, best_l2_aug, ...)
└── vanilla/           # sweeps single-axis sin regularización (lr, opt, arch, act)
```

### Ensembles

Hay dos formas de armar un ensemble:

1. **Multi-seed con un solo config.** Definir `seeds: [42, 0, 7, 13]` en el JSON.
   `run.py` entrena un modelo por seed, los guarda como `<save_model>_seedN.npz`,
   reporta cada uno y luego un `ENSEMBLE` que promedia las probs.

2. **Combinar modelos heterogéneos ya entrenados** (distintas arquitecturas /
   regularizaciones):

   ```bash
   uv run python -m exercises.ej3_more_digits.ensemble best best_l2 best_l2_aug
   ```

   Lee `outputs/ej3_more_digits/models/<name>.npz`, evalúa cada uno sobre
   `digits_test.csv`, los promedia y compara contra el mejor individual.

`make run-ej3-ensembles` entrena todos los configs de `configs/ej3_more_digits/ensembles/`
y al final corre `plot_ensembles_comparison.py`.

### Entrenamiento En Paralelo

Para entrenar varios configs concurrentemente (cada uno en su propio proceso, con
1 hilo de BLAS por worker para no pisarse):

```bash
uv run python -m exercises.ej3_more_digits.run_parallel \
    configs/ej3_more_digits/ensembles/diverse_architectures/*.json \
    --workers 4
```

Cada worker escribe su log en `outputs/ej3_more_digits/parallel_logs/<config>.log`
y al final se imprime una tabla resumen con tiempos y status.

### Evaluar Modelos Guardados

```bash
# un modelo
uv run python -m exercises.ej3_more_digits.eval \
    outputs/ej3_more_digits/models/best_l2_aug.npz

# todos los modelos en outputs/ej3_more_digits/models/, agrupados por
# técnicas de regularización detectadas en sus configs
uv run python -m exercises.ej3_more_digits.eval_all
```

`eval_all` también escribe un CSV resumen en
`outputs/ej3_more_digits/metrics/all_models_eval.csv`.

### Interpretabilidad (Saliency + Occlusion)

Para inspeccionar qué mira el modelo en cada clase:

```bash
# un modelo, un panel por dígito 0-9
uv run python -m exercises.ej3_more_digits.interpretability \
    outputs/ej3_more_digits/models/arch_wide.npz

# solo errores y/o clases puntuales
uv run python -m exercises.ej3_more_digits.interpretability \
    outputs/ej3_more_digits/models/arch_wide.npz --wrong --classes 3,6

# comparar dos modelos sobre un caso donde A acierta y B falla
uv run python -m exercises.ej3_more_digits.interpretability_compare \
    outputs/ej3_more_digits/models/arch_wide.npz \
    outputs/ej3_more_digits/models/baseline_pure.npz \
    --target-class 8
```

Las figuras se guardan en `outputs/ej3_more_digits/interpretability/`.

### Plots Disponibles

Todos leen modelos/CSVs de `outputs/ej3_more_digits/` y guardan PNGs en
`outputs/ej3_more_digits/metrics/`:

| Script | Qué muestra |
|--------|-------------|
| `plot_ensembles_comparison.py` | barras por estrategia: 4 modelos individuales + ensemble |
| `plot_ensemble_story.py`       | ensemble winner + gap train/test por modelo constituyente |
| `plot_test_acc.py`             | barras de test acc para todos los modelos guardados |
| `plot_combined_lines.py`       | curvas train/val con/sin regularización (loss y acc) |
| `plot_gaps.py`                 | gap acc y loss train→test, con vs sin regularización |
| `plot_regularization_sweeps.py`| sweep de L2 (λ) y de augmentation |
| `plot_with_vs_without.py`      | comparativa con/sin una técnica concreta (CLI) |

Ejemplo del último:

```bash
uv run python -m exercises.ej3_more_digits.plot_with_vs_without \
    --without outputs/ej3_more_digits/models/baseline_pure.npz \
    --with    outputs/ej3_more_digits/models/baseline_only_l2.npz \
    --label   "L2 / Weight Decay" \
    --out     outputs/ej3_more_digits/metrics/l2_with_vs_without.png
```

## Tests

Suite completa:

```bash
make test
```

Equivalente directo:

```bash
uv run pytest
```

Tests por área:

```bash
uv run pytest tests/common
uv run pytest tests/exercises/ej1_fraud
```

## Resultados Documentados

El resumen de experimentos MLP está en:

```text
reports/results.md
```

Los resultados regenerables de nuevas corridas se guardan en `outputs/`.
