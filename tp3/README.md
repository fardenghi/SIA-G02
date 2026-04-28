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
make run-ej3 EJ3_CONFIG=configs/ej3_more_digits/softmax.json
make inspect-digit DIGIT_DATASET=test DIGIT_INDEX=0
make clean-ej1
make clean-outputs
```

`outputs/` se regenera al correr experimentos. No está pensado para versionarse.

## Código Común

`common/` contiene la implementación compartida por los ejercicios:

| Archivo | Uso |
|---------|-----|
| `activations.py` | `step`, `linear`, `sigmoid`, `tanh`, `softmax` y derivadas |
| `simple_perceptron.py` | perceptrón simple usado en Ej1 |
| `mlp.py` | perceptrón multicapa usado en XOR, Ej2 y Ej3 |
| `layers.py` | capa densa vectorizada para MLP |
| `losses.py` | MSE y cross entropy |
| `optimizers.py` | SGD, Momentum y Adam |
| `datasets.py` | loaders de XOR, dígitos y helpers de one-hot |
| `metrics.py` | registro/exportación de métricas por época |

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
-> compare_models
-> train_generalization
-> threshold_analysis
```

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
uv run python -m exercises.ej1_fraud.compare_models
uv run python -m exercises.ej1_fraud.train_generalization
uv run python -m exercises.ej1_fraud.threshold_analysis
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

```text
configs/ej2_digits/baseline.json
configs/ej2_digits/adam.json
configs/ej2_digits/softmax.json
configs/ej2_digits/momentum.json
configs/ej2_digits/sgd_lr_low.json
configs/ej2_digits/sgd_lr_high.json
configs/ej2_digits/arch_small.json
configs/ej2_digits/arch_large.json
configs/ej2_digits/arch_deep.json
```

Cada config define arquitectura, función de pérdida, activación de salida, optimizador, learning rate, epochs, batch size, early stopping y paths de outputs.

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

Objetivo: entrenar MLP con `more_digits.csv`, opcionalmente combinado con `digits.csv`, para mejorar la generalización contra `digits_test.csv`.

Código:

```text
exercises/ej3_more_digits/run.py
```

Datos:

```text
data/ej3_more_digits/more_digits.csv
data/ej2_digits/digits.csv          # usado si combine_datasets=true
data/ej2_digits/digits_test.csv     # test final
```

Ejecutar:

```bash
make run-ej3
```

Equivalente directo:

```bash
uv run python -m exercises.ej3_more_digits.run configs/ej3_more_digits/softmax.json
```

Config:

```text
configs/ej3_more_digits/softmax.json
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
