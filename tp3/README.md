# TP3 — Perceptrón y MLP

## Estructura del proyecto

```
tp3/
├── src/
│   ├── activation.py       # funciones de activación (step, tanh, sigmoid, softmax)
│   ├── perceptron.py       # SimplePerceptron (Ejercicio 1)
│   ├── datasets.py         # loaders: xor_dataset, load_digits, load_more_digits, to_one_hot
│   ├── layers.py           # DenseLayer — forward/backward vectorizados
│   ├── losses.py           # mse, mse_grad, cross_entropy, cross_entropy_softmax_grad
│   ├── mlp.py              # MLP — entrenamiento, save/load, evaluate
│   ├── optimizers.py       # SGD, Momentum, Adam
│   └── utils/
│       └── metrics.py      # MetricsTracker — registro por época, export CSV
├── configs/
│   ├── xor.json                  # [2,2,1] MSE+tanh para XOR
│   ├── digits_baseline.json      # [784,64,10] MSE+tanh SGD
│   ├── digits_adam.json          # [784,64,10] MSE+tanh Adam
│   ├── digits_softmax.json       # [784,64,10] CE+softmax Adam
│   └── more_digits_softmax.json  # [784,256,128,10] CE+softmax Adam
├── tests/                  # pytest — 51 tests, todos verdes
├── experiments/            # CSVs de métricas exportadas por cada corrida
├── models/                 # modelos .npz guardados por los scripts
├── run_xor.py              # Ejercicio XOR
├── run_digits.py           # Ejercicio 2 — digits
├── run_more_digits.py      # Ejercicio 3 — more_digits (goal 98%)
└── main.py                 # Ejercicio 1 — SimplePerceptron (sin cambios)
```

## Ejercicio 1 — Detector de Fraude (Perceptrón Simple)

### Objetivo

Entrenar un `TinyModel` (perceptrón simple) que replique el comportamiento del `BigModel` para predecir la probabilidad de fraude (`big_model_fraud_probability`) en transacciones online. El dataset contiene 7 500 transacciones reales con 9 features numéricas.

### Estructura del módulo

```
src/ej1/
├── data.py        # carga, normalización z-score y split train/val
└── explore.py     # script EDA ejecutable (Fase 1)
```

Los resultados y plots se guardan en `experiments/ej1/`.

### Cómo ejecutar el EDA

```bash
uv run python -m src.ej1.explore
```

### Hallazgos del dataset (Fase 1)

| Feature                       | Correlación con target | Observaciones              |
|-------------------------------|------------------------|----------------------------|
| `account_age_days`            | −0.585                 | feature más informativa    |
| `quantity_purchased`          | +0.563                 |                            |
| `amount_usd`                  | +0.557                 | 2.1% outliers (|z| > 3)    |
| `session_duration_seconds`    | −0.514                 |                            |
| `days_since_last_purchase`    | −0.404                 |                            |
| `items_viewed_before_purchase`| +0.334                 |                            |
| `device_screen_resolution`    | +0.025                 | correlación casi nula      |
| `time_since_last_login_s`     | +0.002                 | correlación casi nula      |
| `timestamp`                   | +0.001                 | **descartado** (no predictivo) |

- Sin valores faltantes → no se elimina ninguna fila (D1 N/A).
- Tasa de fraude binaria (`flagged_fraud`): 11.6% (dataset desbalanceado para clasificación).
- Target continuo (`big_model_fraud_probability`): distribución [0, 1], media 0.42.

### Decisiones de diseño

| ID | Decisión                        | Elección                     | Justificación                                                  |
|----|---------------------------------|------------------------------|----------------------------------------------------------------|
| D1 | Valores faltantes               | Eliminar filas               | No hay faltantes; política defensiva para nuevos datos         |
| D2 | Learning rate inicial           | **0.01**                     | Balance entre estabilidad y velocidad de convergencia          |
| D3 | Épocas máximas                  | **100–200**                  | Suficientes para observar convergencia; ajustar si hay plateau |
| D4 | Activación no lineal            | **sigmoid**                  | Target en [0,1]; sigmoid es el mapeo natural                   |
| D5 | Proporción train/val            | **80/20**                    | Estándar para datasets medianos (7 500 muestras)               |
| D6 | Estrategia de split             | **Random shuffle**           | Target continuo → split aleatorio simple y suficiente          |
| D7 | Métrica para umbral de detección| **F1-score** (Fase 7)        | Balance precision-recall; curva completa disponible al cliente |

### Preprocesamiento aplicado

- **Normalización:** z-score sobre las 8 features (excluye `timestamp`).
- **Outliers:** conservados — representan transacciones reales; la normalización modera su influencia.
- **Features de baja correlación** (`device_screen_resolution`, `time_since_last_login_s`): incluidas; el modelo asignará pesos cercanos a cero si no aportan.

---

## MLP

### Cómo correr cada script

```bash
# XOR (con plot si "plot": true en el config)
python run_xor.py configs/xor.json

# Digits — Ejercicio 2
python run_digits.py configs/digits_baseline.json
python run_digits.py configs/digits_adam.json
python run_digits.py configs/digits_softmax.json

# More digits — Ejercicio 3
python run_more_digits.py configs/more_digits_softmax.json
```

### Formato del config JSON

| Campo              | Tipo      | Descripción                                |
|--------------------|-----------|--------------------------------------------|
| `architecture`     | list[int] | tamaños de capas incl. entrada y salida    |
| `activation`       | str       | activación capas ocultas: `"tanh"`         |
| `output_activation`| str       | `"tanh"` o `"softmax"`                    |
| `loss`             | str       | `"mse"` o `"cross_entropy"`               |
| `optimizer`        | str       | `"sgd"`, `"momentum"`, `"adam"`           |
| `lr`               | float     | learning rate                              |
| `epochs`           | int       | épocas máximas                             |
| `batch_size`       | int       | tamaño de mini-batch                       |
| `patience`         | int/null  | early stopping; `null` para desactivarlo   |
| `val_split`        | float     | fracción de datos para validación (0.0–1.0)|
| `seed`             | int       | semilla para reproducibilidad              |
| `plot`             | bool      | (solo XOR) muestra decision boundary      |
| `save_model`       | str/null  | ruta `.npz` para guardar el modelo        |
| `export_metrics`   | str/null  | ruta CSV para exportar métricas por época  |

### Combinaciones de pérdida + activación soportadas

- `"mse"` + `"tanh"`: baseline de clase, target one-hot signed `{-1, +1}`
- `"cross_entropy"` + `"softmax"`: variante para clasificación multi-clase, target one-hot `{0, 1}`

### Tests

```bash
pytest tests/           # 51 tests, ~1 segundo
```
