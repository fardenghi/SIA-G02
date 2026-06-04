# TP3: Perceptrones y Multi-Layer Perceptron (MLP)

Este proyecto estructura la implementacion y evaluacion de modelos basados en Perceptrones Simples y Redes Neuronales Multicapa (MLP) aplicados a diferentes problemas de clasificacion. La logica central y reutilizable se encuentra aislada en el directorio `common/`, mientras que cada ejercicio especifico gestiona sus propios flujos de ejecucion, datos y configuraciones.

Todos los comandos detallados en este documento deben ejecutarse desde la raiz del directorio `tp3/`.

## Configuracion del Entorno

Para instalar las dependencias y sincronizar el entorno virtual, ejecute:

```bash
uv sync
```

Para verificar la integridad del entorno y la correctitud de las implementaciones base:

```bash
make test
```

## Estructura del Proyecto

```text
tp3/
├── common/                 # Modulos compartidos: Perceptron simple, MLP, loaders, funciones de perdida, optimizadores, metricas.
├── exercises/              # Logica especifica por problema
│   ├── xor/                # Validacion inicial del MLP resolviendo el problema XOR.
│   ├── ej1_fraud/          # Ejercicio 1: Deteccion de fraude empleando Perceptrones Simples.
│   ├── ej2_digits/         # Ejercicio 2: Clasificacion multiclase sobre el dataset digits.csv.
│   └── ej3_more_digits/    # Ejercicio 3: Generalizacion, regularizacion y ensembles sobre more_digits.csv.
├── configs/                # Archivos JSON de configuracion hiperparametrica, separados por ejercicio.
├── data/                   # Datasets de entrenamiento y prueba.
├── docs/                   # Enunciado, documentacion teorica y lineamientos.
├── reports/                # Reportes de resultados y experimentacion (versionables).
├── outputs/                # Artefactos generados: modelos (.npz), metricas (.csv) y graficos (ignorado en control de versiones).
├── tests/                  # Suite de pruebas unitarias para el modulo common y los ejercicios.
├── Makefile                # Definicion de comandos principales para facilitar la ejecucion.
└── pyproject.toml          # Declaracion de dependencias y configuracion de pytest.
```

## Comandos Principales de Ejecucion

```bash
make test
make run-ej1
make run-xor
make run-ej2 EJ2_CONFIG=configs/ej2_digits/baseline.json
make run-ej3 EJ3_CONFIG=configs/ej3_more_digits/historical/best_l2_aug.json
make run-ej3-ensembles                  # Entrena de forma secuencial las configuraciones en configs/ej3_more_digits/ensembles/
make inspect-digit DIGIT_DATASET=test DIGIT_INDEX=0
make clean-ej1
make clean-ej2
make clean-ej3
make clean-outputs
```

La ejecucion de `run-ej2` y `run-ej3` conlleva la generacion automatica de graficos de metricas (`outputs/<ej>/metrics/<config>_plot.png`), a partir de los datos tabulares exportados durante el entrenamiento. El directorio `outputs/` es temporal y no esta diseñado para ser persistido en el repositorio.

## Arquitectura de Codigo Comun (`common/`)

El directorio `common/` encapsula toda la matematica y los algoritmos fundamentales que sustentan los modelos:

- `activations.py`: Funciones de activacion (escalon, lineal, sigmoide, tanh, relu, softmax) y sus respectivas derivadas para backpropagation.
- `simple_perceptron.py`: Implementacion orientada a objetos del Perceptron Simple, utilizada en el Ejercicio 1.
- `mlp.py`: Implementacion completa del Perceptron Multicapa. Soporta propagacion hacia adelante, retropropagacion de errores, parada temprana (early stopping), data augmentation on-the-fly, y ajuste dinamico de la tasa de aprendizaje.
- `layers.py`: Representacion vectorizada de capas densas, incluyendo inicializacion de pesos mediante metodos de Xavier/Glorot y He.
- `losses.py`: Funciones de perdida como Error Cuadratico Medio (MSE) y Entropia Cruzada Categorica.
- `optimizers.py`: Estrategias de optimizacion de gradiente descendente, incluyendo SGD estandar, SGD con Momentum, RMSProp y Adam (con soporte para decaimiento de pesos L2). Incluye planificadores de tasa de aprendizaje (`StepDecay`, `ExponentialDecay`, `AdaptiveLR`).
- `ensemble.py`: Funcionalidad para combinar predicciones de multiples modelos (soft voting) basados en salidas de tipo softmax.
- `datasets.py`: Rutinas de carga de datos, codificacion one-hot y division para validacion cruzada (k-fold).
- `metrics.py`: Seguimiento exhaustivo de metricas de rendimiento por epoca. Herramientas para analisis de umbrales, matrices de confusion y metricas de clasificacion binaria/multiclase.

### Estructura de Configuracion de Modelos (JSON)

Los experimentos de XOR, Ejercicio 2 y Ejercicio 3 son dirigidos de forma declarativa mediante archivos JSON. Las propiedades soportadas incluyen:

- `architecture`: Lista de enteros definiendo la dimension de cada capa (ej. `[in_features, hidden1, ..., out_classes]`).
- `activation`: Funcion de activacion para capas ocultas (`tanh`, `relu`).
- `output_activation`: Funcion de activacion para la capa de salida (`tanh`, `softmax`).
- `loss`: Criterio a minimizar (`mse`, `cross_entropy`).
- `weight_init`: Metodo de inicializacion de pesos (`xavier`, `he`). Se recomienda `he` al usar activaciones ReLU.
- `optimizer`: Algoritmo de optimizacion (`sgd`, `momentum`, `rmsprop`, `adam`).
- `lr`: Tasa de aprendizaje inicial (float).
- `weight_decay`: Coeficiente de penalizacion L2 (float). El valor 0 deshabilita la regularizacion.
- `epochs`, `batch_size`: Parametros del bucle de entrenamiento (int).
- `val_split`: Fraccion del dataset reservada para validacion (float).
- `patience`, `min_delta`: Configuracion de la parada temprana.
- `seed`: Semilla para el generador pseudoaleatorio.
- `seeds`: Lista de semillas. Entrena secuencias independientes y consolida un Ensemble.
- `data_augmentation`: Boolean. Si es verdadero, aplica transformaciones afines estocasticas.
- `aug_rotation_deg`, `aug_scale_range`: Parametros de rotacion y escalado para aumento de datos.
- `save_model`: Ruta donde se exportan los pesos del modelo `.npz`.
- `export_metrics`: Ruta de exportacion de metricas historicas a formato CSV.

**Planificadores de Tasa de Aprendizaje (`lr_scheduler`)**:

```json
{ "type": "step_decay", "decay_rate": 0.5, "step_size": 40, "lr_min": 1e-6 }
{ "type": "exponential_decay", "decay_rate": 0.99, "lr_min": 1e-6 }
{ "type": "adaptive", "k": 5, "a": 1e-4, "b": 0.1, "lr_min": 1e-6, "lr_max": 1.0 }
```

## Ejercicio: Validacion XOR

Prueba base para demostrar la capacidad del MLP de separar dominios no lineales empleando el algoritmo de backpropagation.

```bash
make run-xor
# Comando subyacente:
# uv run python -m exercises.xor.run configs/xor/default.json
```

## Ejercicio 1: Prediccion de Fraude

Consiste en modelar la probabilidad de fraude en un escenario simplificado mediante variantes del Perceptron Simple, aproximando las decisiones de un modelo de mayor complejidad.

- Datos origen: `data/ej1_fraud/fraud_dataset.csv`
- Objetivo (Target): Columna `big_model_fraud_probability`
- La columna `flagged_fraud` se usa exclusivamente a posteriori para calibrar el umbral de decision.

Flujo de trabajo completo:
1. `explore`: Analisis preliminar.
2. `train_linear`, `train_sigmoid`, `train_relu`: Entrenamiento de variantes funcionales.
3. `compare_models`, `compare_models_relu`: Comparativa de resultados base.
4. `train_generalization`, `train_generalization_relu`: Validacion cruzada de K-Pliegues.
5. `threshold_analysis`, `threshold_analysis_relu`: Seleccion de umbral optimo mediante validacion fuera-de-pliegue (OOF).

Para ejecutar todo el analisis secuencial:
```bash
make run-ej1
```

## Ejercicio 2: Clasificacion de Digitos Base

Abarca el entrenamiento y comparacion exhaustiva de arquitecturas MLP para la clasificacion de digitos empleando `digits.csv` como base de entrenamiento y `digits_test.csv` para evaluacion final.

```bash
make run-ej2 EJ2_CONFIG=configs/ej2_digits/softmax.json
```

El directorio `configs/ej2_digits/` provee escenarios base (baseline, adam, momentum, softmax) y estudios sistematicos de variables (sweeps de arquitectura mediante parametros ocultos, learning rates y optimizadores).

Para consolidar las metricas de varias ejecuciones en un unico grafico:
```bash
uv run python exercises/ej2_digits/plot_ensemble_metrics.py \
    --csvs run1.csv run2.csv run3.csv \
    --out ensemble_curves.png \
    --label "ensemble multi-seed"
```

## Ejercicio 3: Generalizacion Extendida ("More Digits")

Este ejercicio requiere modelar `more_digits.csv` aplicando tecnicas avanzadas de regularizacion y conformacion de ensembles, buscando minimizar el error de generalizacion contra `digits_test.csv`.

```bash
make run-ej3 EJ3_CONFIG=configs/ej3_more_digits/historical/best_l2_aug.json
```

### Regularizacion y Ensembles

La carpeta `configs/ej3_more_digits/` dispone de:
- `baselines/`: Modelos de control.
- `vanilla/`: Experimentos sin regularizacion para definir arquitectura y optimizador base.
- `historical/`: Configuraciones probadas que presentaron buen rendimiento de generalizacion (con L2, augmentation).
- `ensembles/`: Subcarpetas tematicas (`aug_variations`, `diverse_architectures`, `wd_variations`) destinadas a generar la base para ensembles heterogeneos. Recientemente se han incorporado variaciones como `arch_wide_sgdm` en sus distintas versiones (`v1`, `v2`, `v3`) bajo `diverse_architectures/` para incrementar la varianza del ensemble final.

El codigo soporta Ensembles Homogeneos (distinta inicializacion estocastica del mismo hiperparametro) mediante el uso de la propiedad `seeds` en el JSON. Tambien provee Ensembles Heterogeneos que combinan diferentes arquitecturas pre-entrenadas:

```bash
uv run python -m exercises.ej3_more_digits.ensemble best best_l2 best_l2_aug
```

### Ejecucion Paralelizada

Se provee soporte multiprocesamiento para entrenar conjuntos masivos de configuraciones (ideal para evaluar los ensembles heterogeneos):

```bash
uv run python -m exercises.ej3_more_digits.run_parallel \
    configs/ej3_more_digits/ensembles/diverse_architectures/*.json \
    --workers 4
```

### Interpretabilidad Visual

Mecanismos de Saliency y Occlusion para entender la sensibilidad del modelo a las regiones de la entrada:

```bash
uv run python -m exercises.ej3_more_digits.interpretability outputs/ej3_more_digits/models/arch_wide.npz
```

## Pruebas de Integracion y Unidad

La verificacion automatizada se implementa sobre `pytest` asegurando el comportamiento del modulo matematico base y de las rutinas de ejecucion de ejercicios.

```bash
make test
```
