# Resultados de experimentos MLP

## XOR — validación de implementación

| Arquitectura | Optimizer | lr  | Épocas | Acc train |
|-------------|-----------|-----|--------|-----------|
| [2, 2, 1]   | SGD       | 0.1 | 5000   | 100%      |
| [2, 3, 2, 1]| SGD       | 0.5 | 10000  | 100%      |

Ambas arquitecturas convergen correctamente. El MLP resuelve XOR, que el perceptrón simple escalón no puede por ser linealmente no separable.

---

## Ejercicio 2 — `digits.csv` → evaluación en `digits_test.csv`

### Variantes de tasa de aprendizaje (arquitectura fija [784, 64, 10], SGD)

| Config               | lr    | Épocas (stop) | Acc train | Acc val | Acc test |
|----------------------|-------|---------------|-----------|---------|----------|
| `digits_sgd_lr_low`  | 0.001 | 100           | 92.1%     | 89.6%   | 76.5%    |
| `digits_baseline`    | 0.01  | 50            | 96.3%     | 93.9%   | 83.9%    |
| `digits_sgd_lr_high` | 0.1   | 92 ✓          | 99.3%     | 95.9%   | **86.1%** |

**Conclusión lr:** lr=0.001 produce underfitting (el modelo converge muy lento, 100 épocas no alcanzan). lr=0.1 es el mejor: converge más rápido, el early stopping lo frena antes de sobreajustar.

### Variantes de mecanismo de optimización (arquitectura [784, 64, 10], lr=0.01)

| Config             | Optimizer | Épocas (stop) | Acc train | Acc test |
|--------------------|-----------|---------------|-----------|----------|
| `digits_baseline`  | SGD       | 50            | 96.3%     | 83.9%    |
| `digits_momentum`  | Momentum  | 63 ✓          | 99.3%     | 85.9%    |
| `digits_adam`      | Adam      | 50            | 99.5%     | 85.4%    |

**Conclusión optimizadores:** Momentum y Adam convergen más rápido que SGD puro. Momentum alcanza 85.9% — similar a Adam pero con menos parámetros internos. SGD necesita más épocas o lr más alto para competir.

### Variantes de arquitectura (Adam, lr=0.001)

| Config              | Arquitectura     | Parámetros | Épocas (stop) | Acc test |
|---------------------|-----------------|------------|---------------|----------|
| `digits_arch_small` | [784, 32, 10]   | ~25k       | 40 ✓          | 84.4%    |
| `digits_adam`       | [784, 64, 10]   | ~50k       | 50            | 85.4%    |
| `digits_arch_large` | [784, 128, 10]  | ~101k      | 40 ✓          | **85.9%** |
| `digits_arch_deep`  | [784, 64, 32, 10]| ~52k      | 34 ✓          | 85.7%    |

**Conclusión arquitectura:** más neuronas en la capa oculta mejora el accuracy hasta cierto punto. La red profunda [784, 64, 32, 10] no supera a la ancha [784, 128, 10], y sobreajusta más rápido (early stop en epoch 34).

### Variantes de función de costo / activación de salida

| Config              | Loss + salida     | Optimizer | Acc test |
|---------------------|------------------|-----------|----------|
| `digits_baseline`   | MSE + tanh       | SGD       | 83.9%    |
| `digits_adam`       | MSE + tanh       | Adam      | 85.4%    |
| `digits_softmax`    | CE + softmax     | Adam      | 85.8%    |

**Conclusión loss:** CE+softmax da mejor accuracy y es la función teóricamente correcta para clasificación multi-clase. Con MSE+tanh la clase 8 siempre queda en 0% de accuracy en el test set (problema de saturación / escala de targets).

### Mejor resultado Ejercicio 2

**`digits_sgd_lr_high`**: SGD lr=0.1, [784, 64, 10], early stop epoch 92 → **86.1% accuracy en test**.

---

## Ejercicio 3 — `more_digits.csv` (+ `digits.csv`) → evaluación en `digits_test.csv`

### ¿Por qué más datos mejoran el rendimiento?

El dataset `more_digits.csv` tiene ~15.7k muestras vs ~12.4k de `digits.csv` (+27%). Más datos reducen el sobreajuste porque el modelo generaliza mejor al ver mayor diversidad de ejemplos por clase.

### Resultados

| Config / variante                  | Dataset train            | Arquitectura      | Épocas (stop) | Acc test |
|------------------------------------|--------------------------|------------------|---------------|----------|
| `more_digits_softmax` (original)   | more_digits (15.7k)      | [784,128,64,10]  | 25 ✓          | 95.3%    |
| — tuned arch                       | more_digits (15.7k)      | [784,300,10]     | 41 ✓          | 95.6%    |
| — tuned deep                       | more_digits (15.7k)      | [784,128,64,10]  | 39 ✓          | 95.6%    |
| `more_digits_softmax` (final)      | more_digits + digits (28k) | [784,256,10]  | ~36 ✓         | **96.4%** |

### Técnicas que mejoraron el rendimiento vs Ejercicio 2

1. **Más datos** (+27% solo con more_digits, +127% combinando ambos): reduce sobreajuste.
2. **CE + softmax**: función de costo correcta para multi-clase, gradientes más limpios.
3. **Early stopping** (patience=25): evita sobreajustar, restaura pesos del mejor epoch.
4. **Adam**: convergencia rápida y adaptativa.

### Resultado final Ejercicio 3

**96.4% accuracy** en `digits_test.csv`. La meta de 98% no se alcanzó con MLP puro (sin dropout ni regularización L2). El límite práctico con este enfoque parece ser ~96-97%.

---

## Resumen comparativo

| Experimento         | Mejor config                    | Acc test |
|---------------------|---------------------------------|----------|
| Ej2 — mejor lr      | SGD lr=0.1, [784,64,10]        | 86.1%    |
| Ej2 — mejor optim   | Momentum, [784,64,10]          | 85.9%    |
| Ej2 — mejor arch    | Adam, [784,128,10]             | 85.9%    |
| Ej2 — mejor loss    | Adam, CE+softmax, [784,64,10]  | 85.8%    |
| **Ej3 — final**     | Adam, CE+softmax, [784,256,10], datos combinados | **96.4%** |
