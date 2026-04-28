# Ejercicio 1 — Detector de Fraude con Perceptrón Simple

## Índice

1. [Descripción del problema](#1-descripción-del-problema)
2. [Fase 1 — Exploración de datos](#2-fase-1--exploración-de-datos)
3. [Fase 2 — Arquitectura del código](#3-fase-2--arquitectura-del-código)
4. [Fase 3 — Perceptrón lineal](#4-fase-3--perceptrón-lineal)
5. [Fase 4 — Perceptrón no lineal (sigmoid)](#5-fase-4--perceptrón-no-lineal-sigmoid)
6. [Fase 5 — Comparación y selección](#6-fase-5--comparación-y-selección)
7. [Fase 6 — Estudio de generalización](#7-fase-6--estudio-de-generalización)
8. [Fase 7 — Umbral de detección](#8-fase-7--umbral-de-detección)
9. [Cómo replicar los experimentos](#9-cómo-replicar-los-experimentos)

---

## 1. Descripción del problema

El objetivo es entrenar un **TinyModel** (perceptrón simple) que replique el comportamiento del **BigModel** para predecir la probabilidad de fraude en transacciones de comercio electrónico.

- **Input:** 8 features numéricas por transacción
- **Target:** `big_model_fraud_probability` ∈ [0, 1] — probabilidad continua asignada por el BigModel
- **Ground truth binario:** `flagged_fraud` ∈ {0, 1} — usado para evaluar el umbral de detección

---

## 2. Fase 1 — Exploración de datos

**Script:** `src/ej1/explore.py`
**Plots generados:** `experiments/ej1/plots/target_distribution.png`, `correlation_matrix.png`, `feature_boxplots.png`

### Dataset

| Dimensión | Valor |
|-----------|-------|
| Filas | 7 500 |
| Columnas | 11 (9 features + target continuo + label binario) |
| Valores faltantes | 0 |
| Tasa de fraude (`flagged_fraud`) | 11.6% |
| Target medio (`big_model_fraud_probability`) | 0.42 ± 0.30 |

### Correlación de features con el target

| Feature | Correlación | Nota |
|---------|-------------|------|
| `account_age_days` | −0.585 | más informativa |
| `quantity_purchased` | +0.563 | |
| `amount_usd` | +0.557 | 2.1% outliers |
| `session_duration_seconds` | −0.514 | |
| `days_since_last_purchase` | −0.404 | |
| `items_viewed_before_purchase` | +0.334 | |
| `device_screen_resolution` | +0.025 | correlación casi nula |
| `time_since_last_login_s` | +0.002 | correlación casi nula |
| `timestamp` | +0.001 | **descartado** |

### Decisiones de preprocesamiento

- `timestamp` eliminado — no es predictivo (correlación ~0).
- Las 8 features restantes se normalizan con **z-score**.
- Outliers conservados — representan transacciones reales válidas.
- `device_screen_resolution` y `time_since_last_login_s` se mantienen; el modelo asignará pesos bajos si no aportan.

---

## 3. Fase 2 — Arquitectura del código

**Módulos creados en `src/ej1/`:**

| Archivo | Responsabilidad |
|---------|-----------------|
| `data.py` | Carga del CSV, normalización z-score, split train/val |
| `config.py` | Hiperparámetros por defecto, load/save JSON |
| `training.py` | `FraudTrainer`: entrena, monitorea validación, guarda/carga modelos |
| `evaluation.py` | `evaluate()` (MSE, MAE), plots de curvas y distribuciones |

**Decisiones de diseño:**

| ID | Decisión | Valor elegido |
|----|----------|---------------|
| D1 | Valores faltantes | Eliminar filas |
| D2 | Learning rate | 0.01 |
| D3 | Épocas máximas | 100 |
| D4 | Activación no lineal | sigmoid |
| D5 | Split train/val | 80/20 |
| D6 | Estrategia de split | Random shuffle |
| D7 | Métrica para umbral | F1-score |

La normalización en inferencia usa siempre las estadísticas del conjunto de entrenamiento para evitar data leakage.

---

## 4. Fase 3 — Perceptrón lineal

**Script:** `src/ej1/train_linear.py`
**Modelo guardado:** `experiments/ej1/linear.*`
**Plot:** `experiments/ej1/plots/linear_loss.png`

### Configuración

```
Activación:     linear (identidad)
Learning rate:  0.01
Épocas:         100
Datos:          dataset completo (7 500 muestras, sin split)
```

### Resultados

| Métrica | Valor |
|---------|-------|
| MSE final | 0.02881 |
| MAE final | 0.12903 |
| RMSE | 0.168 |
| Épocas hasta convergencia | ~90 |
| Gap vs OLS (cota inferior) | 0.00275 |

El perceptrón lineal converge al mínimo teórico del modelo lineal con una diferencia de solo 0.003 respecto a la solución analítica OLS. El loss se estabiliza por completo a partir de la época 90.

---

## 5. Fase 4 — Perceptrón no lineal (sigmoid)

**Script:** `src/ej1/train_nonlinear.py`
**Modelo guardado:** `experiments/ej1/nonlinear.*`
**Plot:** `experiments/ej1/plots/linear_vs_nonlinear_loss.png`

### Configuración

```
Activación:     sigmoid  (rango natural [0,1] = mismo que el target)
Learning rate:  0.01
Épocas:         100
Datos:          dataset completo (mismos que Fase 3, comparación justa)
```

### Resultados

| Métrica | Lineal | Sigmoid | Mejora |
|---------|--------|---------|--------|
| MSE | 0.02827 | **0.01089** | −61.5% |
| MAE | 0.12903 | **0.07636** | −40.8% |
| RMSE | 0.168 | **0.104** | −38.1% |
| Convergencia | ~90 épocas | ~90 épocas | igual |

La sigmoid reduce el MSE en un **61.5%** con idénticos hiperparámetros. Ambos modelos convergen en el mismo rango de épocas, confirmando que 100 es el default adecuado.

---

## 6. Fase 5 — Comparación y selección

**Script:** `src/ej1/compare_models.py`
**Plot:** `experiments/ej1/plots/phase5_comparison.png`

### Análisis de curvas

- **Underfitting:** No — ambos modelos bajan consistentemente hasta convergencia.
- **Saturación:** No — sigmoid sigue mejorando frente al lineal durante todo el entrenamiento.
- **Overfitting:** No aplica en este estudio (sin split)

### Decisión

**Modelo elegido: Sigmoid.**

La activación sigmoid captura las relaciones no lineales entre las features y la probabilidad de fraude, y su rango de salida [0, 1] coincide naturalmente con el target continuo. La mejora del 61.5% en MSE es consistente y reproducible.

---

## 7. Fase 6 — Estudio de generalización

**Script:** `src/ej1/train_generalization.py`
**Modelo guardado:** `experiments/ej1/generalization.*`
**Plots:** `experiments/ej1/plots/generalization_loss.png`, `generalization_pred_dist.png`

### Configuración

```
Split:          80% train (6 000) / 20% val (1 500) — random, seed=42
Normalización:  stats computadas solo sobre train (sin data leakage)
```

### Resultados

| Split | MSE | MAE | RMSE |
|-------|-----|-----|------|
| Train | 0.010861 | 0.076767 | 0.1042 |
| Val | 0.011535 | 0.078347 | 0.1074 |
| **Gap** | **+6.2%** | **+2.1%** | — |

### Análisis

- **Overfitting:** No — el gap val-train del 6.2% en MSE está muy por debajo del umbral de alarma (10%).
- **Underfitting:** No — el MSE de train (0.0109) es bajo para este problema.
- El modelo generaliza bien a datos no vistos con una degradación mínima de rendimiento.

---

## 8. Fase 7 — Umbral de detección

**Script:** `src/ej1/threshold_analysis.py`
**Plots:** `experiments/ej1/plots/threshold_metrics.png`, `precision_recall_curve.png`
**Resultados:** `experiments/ej1/threshold_results.json`

Las predicciones continuas del modelo se binariza aplicando un umbral θ: si `p̂ ≥ θ` se clasifica como fraude. Se evalúa sobre el val set usando `flagged_fraud` como ground truth.

### Métricas por umbral (selección)

| Umbral | Precision | Recall | F1 |
|--------|-----------|--------|----|
| 0.50 | 0.338 | 1.000 | 0.505 |
| 0.65 | 0.528 | 0.995 | 0.690 |
| 0.75 | 0.689 | 0.973 | 0.806 |
| 0.80 | 0.768 | 0.962 | 0.854 |
| **0.90** | **0.914** | **0.813** | **0.861** ← óptimo |
| 0.95 | 0.951 | 0.753 | 0.841 |

### Recomendación final

**Umbral sugerido: 0.90**

- De cada 100 alertas generadas, ~91 corresponden a fraudes reales.
- Se detecta el 81% de los fraudes existentes.
- F1-score: 0.861

**Guía de ajuste según prioridad del cliente:**

| Prioridad | Umbral recomendado | Trade-off |
|-----------|-------------------|-----------|
| Detectar todos los fraudes (recall máximo) | ≤ 0.55 | Recall ~100%, Precision ~40% |
| Balance F1 óptimo | **0.90** | Recall 81%, Precision 91% |
| Minimizar falsas alarmas (precision máxima) | ≥ 0.95 | Precision 95%, Recall 75% |

---

## 9. Cómo replicar los experimentos

Todos los comandos se ejecutan desde la raíz del proyecto (`tp3/`).

### Requisitos

```bash
uv sync          # instala dependencias del entorno virtual
```

### Ejecución completa (recomendado)

El `Makefile` en la raíz del proyecto agrupa los comandos más comunes:

```bash
make run-ej1     # ejecuta las 7 fases en orden y regenera todos los outputs
make clean-ej1   # elimina todos los .json, .npz y .png de experiments/ej1/
```

### Ejecución fase por fase

```bash
# Fase 1 — Exploración del dataset
uv run python -m src.ej1.explore

# Fase 3 — Entrenar perceptrón lineal
uv run python -m src.ej1.train_linear

# Fase 4 — Entrenar perceptrón sigmoid
uv run python -m src.ej1.train_nonlinear

# Fase 5 — Comparación y selección de modelo
uv run python -m src.ej1.compare_models

# Fase 6 — Estudio de generalización (split 80/20)
uv run python -m src.ej1.train_generalization

# Fase 7 — Análisis de umbral de detección
uv run python -m src.ej1.threshold_analysis
```

> Las fases 3–7 dependen de los outputs de las anteriores; ejecutarlas en orden o usar `make run-ej1`.

### Tests

```bash
uv run pytest tests/test_ej1_data.py tests/test_ej1_training.py -v   # Ej1
uv run pytest                                                          # suite completa
```

### Outputs generados

```
experiments/ej1/
├── plots/
│   ├── target_distribution.png        # distribución del target
│   ├── correlation_matrix.png         # matriz de correlación
│   ├── feature_boxplots.png           # boxplots de features
│   ├── linear_loss.png                # loss del modelo lineal
│   ├── linear_vs_nonlinear_loss.png   # comparación de loss
│   ├── phase5_comparison.png          # comparación de modelos
│   ├── generalization_loss.png        # train vs val loss
│   ├── generalization_pred_dist.png   # distribución de predicciones
│   ├── threshold_metrics.png          # precision, recall y F1 vs umbral
│   └── precision_recall_curve.png     # curva precision-recall
├── linear.npz / linear.json           # modelo lineal (pesos + historial)
├── nonlinear.npz / nonlinear.json     # modelo sigmoid
├── generalization.npz / ...           # modelo con split 80/20
├── selected_config.json               # config del modelo elegido (Fase 5)
└── threshold_results.json             # umbral óptimo y métricas finales
```

### Reproducibilidad

Todos los experimentos usan `seed=42` para el split y la inicialización de pesos. Para cambiar hiperparámetros, modificar `DEFAULT_CONFIG` en `src/ej1/config.py` o los bloques `CFG` en cada script de entrenamiento.
