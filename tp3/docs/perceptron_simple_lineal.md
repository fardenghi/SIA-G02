# Perceptrón simple lineal - teoría, algoritmo y guía de implementación para TP3

## 1. Qué es

El perceptrón simple lineal, también presentado en clase como **ADALINE** (`ADAptive LINear Element`), es una neurona artificial con **una sola neurona** cuya salida es un **valor real**.

Se obtiene reemplazando la activación escalón por la **función identidad**. Por eso ya no devuelve una clase binaria sino un número continuo.

Es el modelo central de la parte lineal de la clase 10.2 y es muy importante para TP3 porque en el ejercicio 1 se pide compararlo contra el perceptrón simple no lineal en una tarea donde la salida representa una **probabilidad entre 0 y 1**.

---

## 2. Modelo matemático

Dado un vector de entrada:

\[
x = (x_1, x_2, \dots, x_n)
\]

la neurona calcula:

\[
h = \sum_{i=1}^{n} x_i w_i + w_0
\]

Si usamos activación identidad:

\[
O(h) = \theta(h) = h
\]

entonces:

\[
O(x) = \sum_{i=1}^{n} x_i w_i + w_0
\]

La salida es un número real.

---

## 3. Bias

Se implementa como en el resto de los perceptrones:

- o como un término independiente `w0`
- o agregando `x0 = 1`

quedando:

\[
O(x) = \sum_{i=0}^{n} x_i w_i
\]

con `x0 = 1`.

En implementación para TP3 conviene usar siempre la segunda forma porque:

- unifica el tratamiento matemático
- simplifica código y operaciones matriciales
- hace más simple el guardado del modelo

---

## 4. Qué problema resuelve

Este modelo sirve para ajustar un **hiperplano** a un conjunto de datos.

### Casos típicos

- regresión lineal simple
- regresión lineal múltiple
- aproximar una función lineal
- baseline para problemas con salida continua

### Intuición

Si la relación entre entradas y salida es aproximadamente lineal, este modelo debería funcionar bien.

---

## 5. Diferencia clave con el perceptrón escalón

El perceptrón escalón devuelve una clase.

El lineal devuelve un valor real.

### Entonces

- escalón -> clasificación
- lineal -> regresión / aproximación continua

---

## 6. Error / función de costo

En la clase se introduce una **función de error cuadrática**:

\[
E(O) = \frac{1}{2}\sum_{\mu=0}^{p-1} (\zeta^\mu - O^\mu)^2
\]

Como la salida depende de los pesos, se la reescribe como función de `w`:

\[
E(w) = \frac{1}{2}\sum_{\mu=0}^{p-1} \left(\zeta^\mu - \theta\left(\sum_{i=0}^{n} x_i^\mu w_i\right)\right)^2
\]

donde:

- `p`: cantidad de muestras
- `μ`: índice del dato
- `ζ^μ`: valor esperado
- `O^μ`: salida del perceptrón

En el caso lineal, `θ(h)=h`.

---

## 7. Aprendizaje con gradiente descendente

A diferencia del escalón, acá sí tiene sentido optimizar la función de costo con derivadas.

La regla general vista en clase es:

\[
w_{nuevo} = w_{anterior} + \Delta w
\]

con:

\[
\Delta w = - \eta \frac{\partial E}{\partial w}
\]

donde:

- `η` es la tasa de aprendizaje
- `∂E / ∂w` es el gradiente del error respecto a los pesos

---

## 8. Desarrollo de la regla de actualización

En la clase se deriva que para un dato \(\mu\):

\[
\Delta w_i = \eta(\zeta^\mu - O^\mu)\theta'(h^\mu)x_i^\mu
\]

Como en el perceptrón lineal:

\[
\theta(h) = h
\quad\Rightarrow\quad
\theta'(h) = 1
\]

entonces la actualización queda:

\[
\Delta w_i = \eta(\zeta^\mu - O^\mu)x_i^\mu
\]

y luego:

\[
w_i \leftarrow w_i + \Delta w_i
\]

Esta es la regla práctica que conviene implementar.

---

## 9. Pseudocódigo limpio para agentes

```python
initialize weights w to small random values
set learning rate eta

for epoch in range(max_epochs):
    for each training example mu in dataset:
        # 1) weighted sum
        h_mu = sum(x_i_mu * w_i for i in range(n + 1))   # incluye bias con x_0 = 1

        # 2) activation (identity)
        O_mu = h_mu

        # 3) update
        for each weight i:
            w_i = w_i + eta * (zeta_mu - O_mu) * x_i_mu

    # 4) evaluate error
    compute mse or another stopping criterion
    if converged:
        break
```

---

## 10. Entrenamiento online visto en clase

La PPT aclara que este algoritmo, tal como está presentado, corresponde al formato **online**:

- se procesa un patrón
- se actualizan pesos
- se pasa al siguiente

### Más adelante

Para TP3 conviene diseñarlo de forma que después pueda convivir con:

- batch
- mini-batch

aunque este modelo de base se puede implementar primero en online.

---

## 11. Qué devuelve

Devuelve un **valor real**.

Esto tiene dos consecuencias prácticas:

### Ventaja

Puede aproximar cantidades continuas.

### Problema para TP3 ejercicio 1

Si el objetivo es modelar una **probabilidad de fraude**, el perceptrón lineal puede devolver valores fuera del rango `[0,1]`.

Por eso, aunque es útil como baseline, no es el candidato natural para producir probabilidades bien comportadas.

---

## 12. Para qué usarlo en TP3

## Validación

El enunciado sugiere generar un conjunto de muestras de una función lineal, por ejemplo:

\[
y = x
\]

y verificar que el modelo la ajuste bien.

## Ejercicio 1

Compararlo con el perceptrón simple no lineal para estudiar:

- underfitting
- saturación de capacidades
- potencial de aprendizaje
- cuál conviene elegir para el estudio de generalización

---

## 13. Buenas prácticas de implementación

## API sugerida

- `forward(X)`
- `predict(X)`
- `fit(X, y, ...)`
- `compute_loss(X, y)`
- `save(path)`
- `load(path)`

## Parámetros útiles

- `learning_rate`
- `max_epochs`
- `shuffle`
- `seed`
- `tolerance`
- `verbose`

## Historial útil

- loss de train por época
- si se evalúa, loss de validación / test por época
- pesos por época si se quiere debuggear
- tiempo por época

---

## 14. Métricas convenientes

Como es un modelo de salida continua, para entrenamiento conviene al menos registrar:

- MSE
- MAE opcional
- error absoluto medio
- curvas por época

Si además se convierte la salida a clase según un umbral, también pueden medirse métricas de clasificación.

---

## 15. Limitaciones

- solo modela relaciones lineales
- no restringe naturalmente la salida a `[0,1]`
- no capta patrones no lineales
- si el problema tiene saturación o curvatura, se queda corto

---

## 16. Checklist de implementación

- [ ] Bias con `x0 = 1`
- [ ] Activación identidad
- [ ] Función de costo cuadrática
- [ ] Regla de actualización correcta
- [ ] Entrenamiento online funcional
- [ ] Historial de loss por época
- [ ] Validación con muestras de `y = x`
- [ ] Soporte de guardado/carga

---

## 17. Resumen en 5 líneas

- Tiene **1 neurona**
- Devuelve un **valor real**
- Usa activación **identidad**
- Aprende con **gradiente descendente**
- Sirve para **regresión lineal** y como baseline para TP3
