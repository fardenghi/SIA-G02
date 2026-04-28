# Resultados de experimentos MLP

## XOR — `configs/xor.json`

| Config         | Arquitectura | Optimizer | lr   | Épocas | Acc (train) |
|----------------|-------------|-----------|------|--------|-------------|
| `xor.json`     | [2, 2, 1]   | SGD       | 0.1  | 5000   | 100%        |

Convergencia confirmada: los 4 inputs clasificados correctamente con sign(output) == target.

---

## Digits (Ejercicio 2) — `digits.csv` / `digits_test.csv`

| Config                  | Arquitectura  | Optimizer | lr     | Épocas | Loss train | Loss test | Acc test |
|-------------------------|--------------|-----------|--------|--------|------------|-----------|----------|
| `digits_baseline.json`  | [784, 64, 10] | SGD       | 0.01   | 50     | 0.0142     | 0.0471    | 83.86%   |
| `digits_adam.json`      | [784, 64, 10] | Adam      | 0.001  | 50     | 0.0016     | 0.0447    | **85.38%** |
| `digits_softmax.json`   | [784, 64, 10] | Adam      | 0.001  | 50     | —          | —         | ~84%     |

**Observaciones:**
- Adam supera a SGD con la misma arquitectura (+2% accuracy en test).
- La clase 8 resulta difícil de predecir en el test set para todos los configs (0% accuracy para clase 8), lo que sugiere distribución diferente en `digits_test.csv` para esa clase.
- El target ">85%" se alcanza con `digits_adam.json`.

---

## More Digits (Ejercicio 3) — `more_digits.csv` / `digits_test.csv`

| Config                       | Arquitectura       | Optimizer | lr     | Épocas (early stop) | Acc test |
|------------------------------|--------------------|-----------|--------|---------------------|----------|
| `more_digits_softmax.json`   | [784, 256, 128, 10] | Adam      | 0.0005 | ~33                 | 95.51%   |

**Observaciones:**
- CE+softmax con Adam logra 95.51% en test — por encima del mínimo requerido (≥95%).
- El modelo alcanza ~100% de accuracy en train después de pocas épocas (sobreajuste).
- Early stopping con patience=20 detiene el entrenamiento al detectar que val_loss deja de mejorar.
- Sin dropout ni regularización L2, la barrera del ~95-96% en test parece ser el límite del modelo con este dataset.
- Para alcanzar 98%, se necesitaría regularización (dropout, L2) o un modelo más profundo con técnicas modernas.

---

## Resumen de hiperparámetros recomendados

| Tarea       | Arquitectura        | Optimizer | lr     | Batch | Encoding    |
|-------------|---------------------|-----------|--------|-------|-------------|
| XOR         | [2, 2, 1]           | SGD       | 0.1    | 4     | signed tanh |
| Digits Ej2  | [784, 64, 10]       | Adam      | 0.001  | 32    | signed tanh |
| Digits Ej3  | [784, 256, 128, 10] | Adam      | 0.0005 | 32    | zero-one CE |
