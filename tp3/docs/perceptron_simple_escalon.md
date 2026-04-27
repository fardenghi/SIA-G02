# Perceptrón simple escalón - teoría, algoritmo y guía de implementación para TP3

## 1. Qué es

El perceptrón simple escalón es el modelo clásico de neurona artificial para **clasificación binaria**.

Recibe un vector de entrada `x`, calcula una combinación lineal con pesos `w` y bias `b`, y luego aplica una **función de activación escalón** o **signo** para decidir la clase.

Es el modelo de la clase 10.1 y sirve como base conceptual para todo lo que viene después: perceptrón lineal, no lineal y multicapa.

---

## 2. Modelo matemático

Dado un patrón de entrada:

\[
x = (x_1, x_2, \dots, x_n)
\]

se calcula el estado interno o nivel de excitación:

\[
h = \sum_{i=1}^{n} x_i w_i - u
\]

donde:

- `w_i`: peso sináptico asociado a la entrada `x_i`
- `u`: umbral

La salida del perceptrón es:

\[
O = \theta(h)
\]

### Activación escalón

En la PPT aparecen dos convenciones:

#### Convención 0 / 1

\[
\theta(x) =
\begin{cases}
1 & \text{si } x \ge 0 \\
0 & \text{en otro caso}
\end{cases}
\]

#### Convención -1 / 1

\[
\theta(x) =
\begin{cases}
1 & \text{si } x \ge 0 \\
-1 & \text{en otro caso}
\end{cases}
\]

Para TP3 conviene usar **-1 / 1** cuando el dataset o el ejercicio de validación venga expresado así.

---

## 3. Bias / umbral

En la clase se muestra primero el modelo con umbral `u` y luego cómo transformarlo a la formulación con bias.

Se puede reescribir la neurona como:

\[
h = \sum_{i=1}^{n} x_i w_i + b
\]

donde:

\[
b = -u
\]

Una forma práctica de implementarlo es agregar una entrada constante:

\[
x_0 = 1
\]

y definir un peso extra `w_0`, quedando:

\[
h = \sum_{i=0}^{n} x_i w_i
\]

con `x_0 = 1`.

### Idea práctica

- los pesos cambian la inclinación del hiperplano de separación
- el bias desplaza el hiperplano

Esto simplifica mucho la implementación porque el bias se actualiza igual que cualquier otro peso.

---

## 4. Qué tipo de problemas resuelve

El perceptrón simple escalón resuelve **problemas linealmente separables**.

Eso significa que existe una recta, plano o hiperplano que separa perfectamente las clases.

### Ejemplos conceptuales

- En \(\mathbb{R}^2\): una recta
- En \(\mathbb{R}^3\): un plano
- En \(\mathbb{R}^n\): un hiperplano

Si el problema no es linealmente separable, el perceptrón simple escalón no alcanza.

---

## 5. Interpretación geométrica

El conjunto de pesos define la orientación del hiperplano de decisión.

La clasificación depende del signo de:

\[
h = w \cdot x + b
\]

- si `h >= 0`, la muestra cae de un lado del hiperplano
- si `h < 0`, cae del otro lado

Por eso el perceptrón simple escalón es, en esencia, un **clasificador lineal**.

---

## 6. Aprendizaje: algoritmo de Rosenblatt

La idea es iterativa:

1. tomar un dato
2. calcular la salida actual
3. compararla con la salida esperada
4. ajustar pesos si se equivocó

Cuando la salida esperada y la salida obtenida coinciden, no hay corrección.

### Variables típicas

- `x^μ`: patrón de entrada \(\mu\)
- `y^μ`: salida esperada para ese patrón
- `O^μ`: salida del perceptrón
- `η`: tasa de aprendizaje

---

## 7. Regla de actualización

Con convención de salidas en `{-1, 1}`:

\[
\Delta w_i = \eta (y^\mu - O^\mu)x_i^\mu
\]

y luego:

\[
w_i \leftarrow w_i + \Delta w_i
\]

Si se implementa con bias como `x_0 = 1`, también se actualiza:

\[
\Delta w_0 = \eta (y^\mu - O^\mu)x_0^\mu
\]

con `x_0 = 1`.

### Observación

Como `O^μ` y `y^μ` son discretos, esta regla:

- mueve los pesos solo cuando hay error
- corrige en la dirección del patrón mal clasificado

---

## 8. Pseudocódigo limpio para agentes

```python
initialize weights w to small random values
set learning rate eta

for epoch in range(max_epochs):
    for each training example mu in dataset:
        # 1) weighted sum
        h_mu = sum(x_i_mu * w_i for i in range(n + 1))   # incluye bias con x_0 = 1

        # 2) activation
        O_mu = step_or_sign(h_mu)

        # 3) update
        for each weight i:
            w_i = w_i + eta * (y_mu - O_mu) * x_i_mu

    # 4) convergence criterion
    compute perceptron error
    if converged:
        break
```

---

## 9. Criterios de error / convergencia vistos en clase

En la PPT se mencionan distintas formas de decidir convergencia:

- la suma del valor absoluto de los errores devuelve cero
- accuracy = 100% para problemas de clasificación
- cualquier criterio equivalente que verifique que no quedan errores

### Recomendación práctica para TP3

Para este perceptrón, registrar por época:

- cantidad de patrones mal clasificados
- accuracy
- pesos actuales
- si hubo o no convergencia

---

## 10. Ventajas y limitaciones

## Ventajas

- simple de entender
- simple de implementar
- rápido
- útil como baseline y como validación de pipeline

## Limitaciones

- solo clasifica linealmente
- no modela probabilidades
- no sirve para relaciones no lineales
- la activación escalón no es diferenciable, por lo que no se usa con gradiente descendente estándar como en multicapa

---

## 11. Qué debería implementar un agente para este modelo

## Estructura recomendada

- `forward(x)`
- `predict(x)` o `predict_class(x)`
- `fit(X, y, ...)`
- `compute_error(...)`
- `save(...)` / `load(...)`

## Detalles prácticos

- trabajar con `numpy`
- agregar `x_0 = 1` para el bias
- permitir inicialización aleatoria reproducible con seed
- exponer `learning_rate`, `max_epochs`, `shuffle`, `seed`
- almacenar historial por época

---

## 12. Validación mínima pedida por el enunciado

El TP3 recomienda validar el perceptrón simple escalón con la función lógica **AND**.

### Entradas

\[
x = \{(-1, 1), (1, -1), (-1, -1), (1, 1)\}
\]

### Salidas esperadas

\[
y = \{-1, -1, -1, 1\}
\]

### Objetivo

El modelo debería aprender a clasificar correctamente los 4 patrones.

Esto es importante porque:

- el problema es pequeño
- es linealmente separable
- permite depurar bias, activación y regla de actualización

---

## 13. Checklist de implementación

- [ ] Bias incorporado como `x0 = 1`
- [ ] Activación signo o escalón bien definida
- [ ] Regla de actualización correcta
- [ ] Shuffle opcional por época
- [ ] Historial de error / accuracy por época
- [ ] Validación con AND
- [ ] Posibilidad de guardar pesos y configuración

---

## 14. Qué reportar en experimentos

Aunque este perceptrón no es el centro del TP3 final, sirve como herramienta de validación. Para dejarlo útil para agentes y experimentos, reportar:

- learning rate
- cantidad de épocas
- convergió o no
- accuracy final
- errores por época
- pesos finales
- tiempo de entrenamiento

---

## 15. Resumen en 5 líneas

- Tiene **1 neurona**
- Devuelve una **clase binaria**
- Usa una activación **escalón/signo**
- Aprende con la **regla de Rosenblatt**
- Solo sirve para **problemas linealmente separables**
