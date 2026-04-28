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
