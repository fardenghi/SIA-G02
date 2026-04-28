# TP3 - métricas, sobreajuste, normalización y optimización

Este archivo junta lo visto en:

- clases de **métricas de evaluación, sobreajuste y normalización**
- PDF de **optimización (extras)**
- lineamientos del **enunciado del TP3**

La idea es que esto sirva como documento operativo para agentes que tengan que desarrollar, evaluar y comparar modelos para el TP3.

---

# 1. Cómo evaluar correctamente un modelo

La clase plantea una pregunta base:

> ¿Cómo medir el desempeño del modelo?

No alcanza con mirar solo el error sobre entrenamiento. Hay que separar:

- conjunto de entrenamiento (`train`)
- conjunto de prueba (`test`)

y evaluar ambos.

## Flujo base de evaluación

1. dividir el dataset en `training set` y `testing set`
2. entrenar con `training set`
3. evaluar con `training set`
4. evaluar con `testing set`

Esto permite comparar:

- qué tan bien aprende el modelo
- qué tan bien generaliza

---

# 2. Matriz de confusión

Es la tabla base para medir el desempeño en clasificación.

## Ejes

- **columnas**: predicción del método
- **filas**: clase real / actual

## Caso binario

| Actual \ Predicción | Positivo | Negativo |
|---|---:|---:|
| Positivo | TP | FN |
| Negativo | FP | TN |

donde:

- `TP`: verdaderos positivos
- `TN`: verdaderos negativos
- `FP`: falsos positivos
- `FN`: falsos negativos

## Ejemplo conceptual de la clase

Se muestra un caso perros vs gatos con una matriz tipo:

| Real \ Predicho | Perro | Gato |
|---|---:|---:|
| Perro | 11 | 4 |
| Gato | 2 | 10 |

---

# 3. Métricas estándar

La clase lista explícitamente:

- Accuracy
- Precision
- Recall
- F1-Score
- Tasa de TP
- Tasa de FP

## 3.1 Accuracy

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Mide el porcentaje total de aciertos.

### Ojo

Si el dataset está desbalanceado, puede ser engañosa.

---

## 3.2 Precision

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

De todo lo que el modelo marcó como positivo, cuánto era realmente positivo.

---

## 3.3 Recall

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

De todo lo que era realmente positivo, cuánto logró detectar.

---

## 3.4 F1-Score

\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Es el balance armónico entre precision y recall.

---

## 3.5 Tasa de verdaderos positivos (TPR)

\[
TPR = \frac{TP}{TP + FN}
\]

En binario coincide con recall.

---

## 3.6 Tasa de falsos positivos (FPR)

\[
FPR = \frac{FP}{FP + TN}
\]

Importa muchísimo si el costo de falsos positivos es relevante.

---

# 4. Qué métricas conviene usar en TP3

## Ejercicio 1 - fraude / probabilidad

Como la empresa pide estimar probabilidad de fraude y además recomendar un **umbral de detección**, conviene separar dos niveles:

### a) como regresión / score continuo

- MSE o MAE para la salida continua
- inspección de calibración si se profundiza

### b) como decisión binaria según umbral

- precision
- recall
- F1
- FPR
- accuracy

### Recomendación práctica

Guardar resultados para varios umbrales, por ejemplo:

```python
thresholds = [0.1, 0.2, 0.3, ..., 0.9]
```

y comparar métricas en cada uno.

---

## Ejercicios 2 y 3 - clasificación de dígitos

Como es clasificación multiclase `0..9`, conviene medir:

- accuracy global
- matriz de confusión multiclase
- accuracy por clase si se quiere analizar qué dígitos se confunden más

---

# 5. Procedimiento experimental visto en clase

La PPT sugiere explícitamente:

1. calcular `w` usando el conjunto de entrenamiento
2. clasificar los datos del conjunto de prueba con esos `w`
3. calcular métricas tanto para entrenamiento como para prueba
4. repetir para distintas épocas

Ejemplo sugerido en clase:

```text
epoch = 1, 10, 20, ..., 300
```

## Traducción práctica a implementación

Para cada configuración:

- entrenar hasta cierta época
- registrar métricas por época en train
- registrar métricas por época en valid/test
- graficar curvas

---

# 6. Underfitting y overfitting

## 6.1 Underfitting

El modelo no tiene suficiente capacidad o no aprendió lo suficiente.

### Señal típica

- error alto en training
- error alto en testing

## 6.2 Overfitting / sobreajuste

En la clase se define como el efecto de **sobreentrenar** el algoritmo sobre datos donde se conoce el resultado deseado.

### Señal típica

- el método clasifica muy bien el conjunto de entrenamiento
- pero no generaliza bien a datos nuevos

### Patrón clásico en curvas

- accuracy de train sigue subiendo
- accuracy de test/valid se estanca o empeora

---

# 7. Causas de sobreajuste mencionadas en clase

La PPT enumera varias causas:

- dataset de entrenamiento no balanceado
- pocos registros de entrenamiento
- mucho ruido en el dataset de entrenamiento

A eso, para implementación práctica, se le puede sumar:

- modelo demasiado grande
- demasiadas épocas
- falta de regularización
- selección inadecuada de hiperparámetros

---

# 8. Resumen de diagnóstico del modelo

La clase lo resume así:

- error alto en train -> **underfitting**
- error bajo en train y alto en test -> **overfitting**
- error bajo en ambos -> **buen modelo**

## Traducción práctica

Siempre mirar **las dos curvas**:

- train
- valid / test

Nunca decidir solo con train.

---

# 9. Train/test split y validación cruzada

La clase pregunta:

> ¿Cómo sabemos si la partición train/test es apropiada?

La respuesta que presenta es usar métodos de experimentación como **K-Fold Cross Validation**.

## Procedimiento K-Fold

1. dividir aleatoriamente el dataset en `k` partes parecidas
2. para cada iteración:
   - usar `k-1` partes para entrenar
   - usar la parte restante para test / validación
3. repetir `k` veces
4. promediar métricas

## Pseudocódigo

```python
split dataset into k folds

for j in range(k):
    training_set = all folds except j
    validation_set = fold j

    train model on training_set
    evaluate on validation_set
    store metrics

final_metric = average(metrics_over_folds)
```

## Para TP3

### Ejercicio 1

Muy útil para elegir:

- perceptrón lineal vs no lineal
- learning rate
- activation
- umbral de decisión

### Ejercicios 2 y 3

Recordar la aclaración del enunciado:

- `digits.csv` se usa para ajustar parámetros e hiperparámetros
- `digits_test.csv` es equivalente a producción / mundo real

Eso significa que el set `digits_test.csv` **no debería usarse para tunear**.

---

# 10. Normalización de datos

La clase cubre tres ideas importantes:

- feature scaling / min-max scaling
- estandarización
- unit length scaling

Esto es clave en redes neuronales porque escalas muy distintas entre features pueden romper el entrenamiento o volverlo inestable.

---

## 10.1 Min-Max Scaling

Se escala al intervalo `[a, b]`:

\[
X' = \frac{X - X_{min}}{X_{max} - X_{min}}(b-a) + a
\]

### Caso especial `[0,1]`

\[
X' = \frac{X - X_{min}}{X_{max} - X_{min}}
\]

### Cuándo conviene

- cuando querés un rango fijo
- cuando la activación es sensible a escala
- útil para inputs de perceptrones simples

---

## 10.2 Estandarización / Z-Score

Dada una variable \(X_i\), con media \(\bar{X_i}\) y desvío estándar \(s_i\):

\[
\tilde{X_i} = \frac{X_i - \bar{X_i}}{s_i}
\]

Esto deja la variable centrada y con escala comparable.

### Cuándo conviene

- en muchos problemas reales
- cuando hay features con órdenes de magnitud distintos
- muy recomendable para MLP

---

## 10.3 Unit Length Scaling

Se divide cada vector por su norma 2:

\[
x' = \frac{x}{\|x\|_2}
\]

### Cuándo conviene

- cuando importa más la dirección que el módulo
- en ciertos problemas de representación de vectores

---

## Recomendación crítica para implementación

**Fittear el escalado solo con train** y luego transformar valid/test con esos parámetros.

Nunca recalcular media, desvío o min/max usando test, porque eso produce **data leakage**.

## Pseudocódigo

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)
```

---

# 11. Optimización - idea general

En el PDF de optimización se recuerda que el gradiente descendente da una dirección \(\Delta W\) para bajar la función de costo.

También se marca el problema central:

- el gradiente usa información **local**
- puede oscilar
- puede avanzar lento
- puede atascarse

Por eso aparecen mejoras sobre el descenso básico.

---

# 12. Momentum

## Fórmula

\[
\Delta w_{ij}(t+1) = -\eta \frac{\partial E}{\partial w_{ij}} + \alpha \Delta w_{ij}(t)
\]

donde:

- `η`: learning rate
- `α`: parámetro de momentum
- valores comunes para `α`: `0.8` o `0.9`

## Intuición

- en regiones planas acelera
- en valles u oscilaciones ayuda a compensar zig-zag
- suaviza el entrenamiento

## Pseudocódigo

```python
velocity = alpha * velocity - eta * grad
w = w + velocity
```

---

# 13. Eta adaptativo

La PPT sugiere adaptar `η` según la evolución del error.

## Idea

- si `E()` decrece consistentemente durante varias iteraciones, podría convenir **aumentar** `η`
- si `E()` empieza a crecer consistentemente, podría convenir **disminuir** `η`

## Regla esquemática vista en clase

\[
\Delta \eta =
\begin{cases}
+a & \text{si } \Delta E < 0 \text{ consistentemente} \\
-b\eta & \text{si } \Delta E > 0 \text{ empieza a ser consistente} \\
0 & \text{en otro caso}
\end{cases}
\]

La clase aclara que “consistentemente” se puede parametrizar de muchas formas, por ejemplo:

- si en `K` épocas el error varía menos de `P%`

## Pseudocódigo posible

```python
if error_decreases_consistently:
    eta = eta + a
elif error_increases_consistently:
    eta = eta - b * eta
```

---

# 14. RMSProp

## Fórmulas

\[
g_t = \frac{\partial E}{\partial w_{ij}}
\]

\[
S_t = \gamma S_{t-1} + (1-\gamma) g_t^2
\]

\[
\Delta w_{ij} = - \frac{\eta}{\sqrt{S_t + \epsilon}} \, g_t
\]

donde:

- `g_t^2` es cuadrado elemento a elemento
- `ε` evita divisiones por cero
- `γ` controla el promedio exponencial

## Intuición de la clase

- RMS alta -> learning rate efectivo más bajo -> amortigua cambios
- RMS baja -> learning rate efectivo más alto -> puede aprender más rápido

## Pseudocódigo

```python
S = gamma * S + (1 - gamma) * (grad ** 2)
w = w - eta * grad / (sqrt(S) + eps)
```

---

# 15. Adam

La PPT lo presenta como combinación de ideas de:

- momentum
- RMSProp

y aclara que es muy usado en la práctica.

## Fórmulas

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\]

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]

Bias correction:

\[
\hat{m_t} = \frac{m_t}{1-\beta_1^t}
\]

\[
\hat{v_t} = \frac{v_t}{1-\beta_2^t}
\]

Update:

\[
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}
\]

## Defaults mostrados en la PPT

- \(\alpha = 0.001\)
- \(\beta_1 = 0.9\)
- \(\beta_2 = 0.999\)
- \(\epsilon = 10^{-8}\)

## Pseudocódigo

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * (grad ** 2)

m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)

w = w - alpha * m_hat / (sqrt(v_hat) + eps)
```

---

# 16. Qué variantes hay que explorar sí o sí en TP3

El enunciado pide como mínimo analizar:

- variantes de tasa de aprendizaje
- variantes de arquitectura
- variantes de mecanismos de optimización

## Traducción práctica

### Para perceptrón lineal y no lineal

- learning rate
- activación
- normalización
- épocas
- umbral de clasificación final

### Para multicapa

- arquitectura: cantidad de capas y neuronas
- learning rate
- optimizador: GD, momentum, RMSProp, Adam
- batch size
- activación
- inicialización
- cantidad de épocas

---

# 17. Recomendaciones de ingeniería del enunciado

El PDF del TP3 recomienda considerar:

- operaciones matriciales para mejorar performance
- reportar progreso mientras corren los modelos
- configuración extensible
- guardar y levantar modelos
- separar almacenamiento de resultados y análisis

## Traducción a estructura de proyecto

```text
common/
  activations.py
  simple_perceptron.py
  mlp.py
  optimizers.py
  losses.py
  metrics.py

exercises/
  ej1_fraud/
  ej2_digits/
  ej3_more_digits/

configs/
  ej1_fraud/
  ej2_digits/
  ej3_more_digits/

data/
  ej1_fraud/
  ej2_digits/
  ej3_more_digits/

outputs/
  ej1_fraud/
  ej2_digits/
  ej3_more_digits/

reports/
```

---

# 18. Qué debería guardar cada experimento

Cada corrida debería persistir al menos:

- nombre del experimento
- modelo
- hiperparámetros
- arquitectura
- optimizer
- learning rate
- batch size
- cantidad de épocas
- seed
- métricas por época
- tiempo total
- mejor época
- path del modelo guardado

## Ejemplo de registro

```json
{
  "experiment_name": "mlp_digits_adam_lr_1e-3",
  "model": "mlp",
  "architecture": [64, 32, 10],
  "optimizer": "adam",
  "learning_rate": 0.001,
  "epochs": 200,
  "batch_size": 32,
  "seed": 42,
  "train_accuracy_history": [...],
  "valid_accuracy_history": [...],
  "train_loss_history": [...],
  "valid_loss_history": [...],
  "best_epoch": 137
}
```

---

# 19. Criterio operativo para elegir mejor modelo

## Ejercicio 1

Elegir el modelo que:

- tenga buena performance en generalización
- produzca scores razonables para el umbral
- sea más chico / barato, como pide CompanyX

## Ejercicios 2 y 3

Elegir el modelo que:

- se ajuste usando `digits.csv` / `more_data_digits.csv`
- no tunee sobre `digits_test.csv`
- tenga la mejor performance final en el test de producción
- si es posible, mantenga entrenamiento estable y reproducible

---

# 20. Checklist para agentes

- [ ] Separar train / valid / test correctamente
- [ ] Implementar matriz de confusión
- [ ] Implementar accuracy, precision, recall, F1, TPR, FPR
- [ ] Registrar métricas por época
- [ ] Detectar underfitting / overfitting con curvas
- [ ] Implementar min-max y estandarización
- [ ] Evitar data leakage al escalar
- [ ] Implementar SGD base
- [ ] Implementar momentum
- [ ] Implementar RMSProp
- [ ] Implementar Adam
- [ ] Permitir barrer hiperparámetros
- [ ] Guardar resultados de experimentos
- [ ] Guardar y cargar modelos

---

# 21. Resumen corto

- Las métricas permiten medir aprendizaje y generalización
- La matriz de confusión es la base para métricas de clasificación
- Overfitting = train muy bien, test mal
- Normalizar datos ayuda muchísimo a estabilizar redes
- Momentum, RMSProp y Adam son mejoras prácticas sobre gradiente descendente
- El TP3 pide comparar variantes y dejar una infraestructura experimental prolija
