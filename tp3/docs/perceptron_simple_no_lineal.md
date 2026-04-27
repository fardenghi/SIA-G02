# Perceptrón simple no lineal - teoría, algoritmo y guía de implementación para TP3

## 1. Qué es

El perceptrón simple no lineal es una neurona artificial con **una sola neurona**, igual que el perceptrón lineal, pero cambia la función de activación por una **función no lineal**.

En la clase 10.2 se mencionan especialmente:

- sigmoidea / logística
- tangente hiperbólica (`tanh`)

La gran idea es introducir **no linealidad** en la salida, lo que permite modelar relaciones más ricas que las del perceptrón lineal.

---

## 2. Modelo matemático

Se parte del mismo cálculo interno:

\[
h = \sum_{i=0}^{n} x_i w_i
\]

usando `x0 = 1` para absorber el bias.

La diferencia es que ahora la salida es:

\[
O = \theta(h)
\]

donde `θ` ya no es identidad ni escalón, sino una función suave no lineal.

---

## 3. Activaciones relevantes vistas en clase

## a) Función logística / sigmoidea

\[
\theta(h) = \frac{1}{1 + e^{-2\beta h}}
\]

### Rango

\[
Im = (0,1)
\]

### Interpretación

Muy útil cuando la salida buscada representa algo tipo probabilidad o score entre 0 y 1.

---

## b) Tangente hiperbólica

\[
\theta(h) = \tanh(\beta h)
\]

### Rango

\[
(-1,1)
\]

### Interpretación

Útil cuando el target está centrado en cero o cuando se quiere una salida simétrica.

---

## 4. Rol del parámetro beta

En la PPT se muestra que `β` modifica la forma de la función.

### Intuición

- `β` grande -> transición más brusca
- `β` chico -> transición más suave

Esto afecta:

- sensibilidad
- saturación
- estabilidad del entrenamiento

Conviene exponerlo como hiperparámetro configurable.

---

## 5. Bias

Como en los otros modelos, conviene implementarlo como:

\[
x_0 = 1
\]

de forma que:

\[
h = \sum_{i=0}^{n} x_i w_i
\]

Esto evita tener una rama especial para el bias.

---

## 6. Qué devuelve

Depende de la activación elegida.

## Si se usa logística

Devuelve un valor en:

\[
(0,1)
\]

## Si se usa tanh

Devuelve un valor en:

\[
(-1,1)
\]

---

## 7. Diferencia con el perceptrón lineal

El lineal devuelve:

\[
O = h
\]

El no lineal devuelve:

\[
O = \theta(h)
\]

con `θ` no lineal.

### Consecuencia

El perceptrón no lineal puede representar transformaciones que el lineal no puede.

---

## 8. Función de costo

En clase se aclara que la **fórmula de error se mantiene igual que en el perceptrón lineal**:

\[
E(w) = \frac{1}{2}\sum_{\mu=0}^{p-1} (\zeta^\mu - O^\mu)^2
\]

La diferencia no está en la forma del error, sino en que ahora la salida depende de una activación no lineal.

---

## 9. Aprendizaje

La regla general sigue siendo gradiente descendente:

\[
\Delta w = -\eta \frac{\partial E}{\partial w}
\]

y para una muestra \(\mu\):

\[
\Delta w_i = \eta(\zeta^\mu - O^\mu)\theta'(h^\mu)x_i^\mu
\]

Esta fórmula es la más importante a implementar.

### Punto clave

A diferencia del lineal, acá:

\[
\theta'(h) \neq 1
\]

y depende de la activación elegida.

---

## 10. Derivadas útiles para implementación

## a) Logística

Si:

\[
\theta(h) = \frac{1}{1 + e^{-2\beta h}}
\]

una forma práctica de implementar la derivada es usar la salida ya calculada.

Puede reescribirse como una función de `O`.

## b) Tanh

Si:

\[
\theta(h) = \tanh(\beta h)
\]

entonces:

\[
\theta'(h) = \beta(1 - \tanh^2(\beta h))
\]

o equivalentemente, si `O = tanh(βh)`:

\[
\theta'(h) = \beta(1 - O^2)
\]

### Recomendación

En código, calcular la derivada a partir de la salida suele ser más estable y más simple.

---

## 11. Pseudocódigo limpio para agentes

```python
initialize weights w to small random values
set learning rate eta

for epoch in range(max_epochs):
    for each training example mu in dataset:
        # 1) weighted sum
        h_mu = sum(x_i_mu * w_i for i in range(n + 1))   # incluye bias con x_0 = 1

        # 2) non-linear activation
        O_mu = activation(h_mu)

        # 3) update
        for each weight i:
            w_i = w_i + eta * (zeta_mu - O_mu) * activation_derivative(h_mu, O_mu) * x_i_mu

    # 4) evaluate loss
    compute mse or stopping criterion
    if converged:
        break
```

---

## 12. Interpretación para TP3 ejercicio 1

El problema pide estimar la **probabilidad de fraude** de una transacción.

Eso hace que, en principio, la activación **logística** sea la opción natural, porque:

- la salida queda acotada entre `0` y `1`
- se interpreta mejor como score / probabilidad
- evita predicciones fuera de rango, a diferencia del lineal

### Comparación conceptual con el lineal

## Lineal

- puede extrapolar fuera de `[0,1]`
- útil como baseline
- menos capacidad para aprender curvaturas

## No lineal con logística

- devuelve valores dentro de `(0,1)`
- suele adaptarse mejor a targets tipo probabilidad
- puede saturar si la activación se lleva a extremos

---

## 13. Saturación

Este punto es importante para TP3 porque el enunciado explícitamente pide analizar:

- underfitting
- saturación de capacidades

## Qué significa saturación acá

Si `h` toma valores muy grandes en módulo, funciones como logística y tanh se acercan a sus extremos y la derivada se vuelve muy pequeña.

### Efectos

- aprendizaje más lento
- gradientes pequeños
- estancamiento

Por eso conviene:

- normalizar / estandarizar inputs
- cuidar learning rate
- inicializar pesos pequeños

---

## 14. Validación mínima pedida por el enunciado

El TP3 recomienda validar este modelo con muestras de una función no lineal, por ejemplo:

\[
y = \tanh(x)
\]

### Qué conviene verificar

- que el loss baje
- que la curva aprendida siga la forma esperada
- que la salida quede en el rango correcto
- que la derivada y la actualización estén bien implementadas

---

## 15. Buenas prácticas de implementación

## API sugerida

- `forward(X)`
- `predict(X)`
- `fit(X, y, ...)`
- `compute_loss(X, y)`
- `activation(...)`
- `activation_derivative(...)`
- `save(...)` / `load(...)`

## Configuración útil

- `activation = "logistic" | "tanh"`
- `beta`
- `learning_rate`
- `max_epochs`
- `shuffle`
- `seed`
- `tolerance`

---

## 16. Qué medir en experimentos

Para el ejercicio 1, registrar al menos:

- loss de train
- loss de validación/test si se usa
- salida mínima y máxima del modelo
- curvas por época
- comparación con el perceptrón lineal
- comportamiento para distintos umbrales de decisión

---

## 17. Limitaciones

- sigue teniendo **1 sola neurona**
- es más flexible que el lineal, pero mucho menos poderoso que un multicapa
- puede saturarse
- la elección de activación importa mucho
- según el dataset, puede seguir siendo insuficiente

---

## 18. Checklist de implementación

- [ ] Bias con `x0 = 1`
- [ ] Activación logística o tanh
- [ ] Derivada bien implementada
- [ ] Regla de actualización correcta
- [ ] Historial de loss por época
- [ ] Validación con `y = tanh(x)`
- [ ] Comparación directa con perceptrón lineal
- [ ] Soporte de guardado/carga

---

## 19. Resumen en 5 líneas

- Tiene **1 neurona**
- Devuelve un **valor continuo no lineal**
- Usa activación **logística** o **tanh**
- Aprende con **gradiente descendente**
- Para TP3 ejercicio 1 es el candidato natural cuando la salida representa una **probabilidad**
