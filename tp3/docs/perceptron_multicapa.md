# Perceptrón multicapa (MLP) - teoría, backpropagation y guía de implementación para TP3

## 1. Qué es

El perceptrón multicapa o **MLP** es una red neuronal con:

- una capa de entrada
- una o más capas ocultas
- una capa de salida

Cada neurona de una capa toma como entrada las salidas de la capa anterior, aplica pesos y activación, y pasa su salida a la capa siguiente.

Es el modelo de la clase 11 y el que se usa en TP3 para los ejercicios de clasificación de dígitos.

---

## 2. Por qué hace falta

El perceptrón simple, incluso el no lineal, sigue teniendo **una sola neurona**.

Eso limita mucho lo que puede representar.

El multicapa aparece como alternativa para modelar **transformaciones más complejas** y resolver problemas que no son linealmente separables, por ejemplo **XOR** o clasificación de dígitos.

---

## 3. Idea general

Una red multicapa puede pensarse como una composición de transformaciones:

\[
x \rightarrow \text{capa oculta 1} \rightarrow \text{capa oculta 2} \rightarrow \dots \rightarrow \text{salida}
\]

Cada capa construye una representación nueva del dato.

---

## 4. Teorema de aproximación universal

En la clase se menciona el **Teorema de Aproximación Universal**:

En teoría, un perceptrón multicapa puede aproximar funciones continuas.

### Pero ojo

Eso no significa automáticamente que:

- sea fácil entrenarlo
- cualquier arquitectura sirva
- la cantidad de neuronas necesaria sea razonable

Para TP3 importa más la parte práctica:

- elegir arquitectura
- entrenar bien
- medir generalización
- comparar variantes

---

## 5. Notación de capas

Usando una notación estándar práctica para implementación:

- `a^(0) = x`: entrada
- `W^(l)`: matriz de pesos de la capa `l`
- `z^(l)`: preactivación
- `a^(l)`: activación / salida de la capa `l`

Entonces para cada capa:

\[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
\]

\[
a^{(l)} = \theta(z^{(l)})
\]

Si el bias se absorbe con una entrada constante `1`, el término `b^(l)` puede incorporarse dentro de `W^(l)`.

---

## 6. Feed-forward

El cálculo de la salida de la red se llama **feed-forward pass**.

### Proceso

1. se toma el vector de entrada
2. se calcula la salida de la primera capa oculta
3. esa salida pasa a la siguiente capa
4. se repite hasta llegar a la salida final

### Fórmulas vistas en clase

Primera capa oculta:

\[
V_j^1 = \theta\left(\sum_{k=1}^{n} x_k^\mu w_{jk}^1\right)
\]

Capas intermedias:

\[
V_j^m = \theta\left(\sum_{k=1}^{n_{m-1}} V_k^{m-1} w_{jk}^m\right)
\]

Capa de salida:

\[
O_i = \theta\left(\sum_{k=1}^{n_{M-1}} V_k^{M-1} W_{ik}\right)
\]

donde:

- `M`: índice de la última capa
- `V_j^m`: salida de la neurona `j` de la capa `m`
- `O_i`: salida final

---

## 7. Rol de la activación no lineal

Esto es central.

Si todas las capas fueran lineales, toda la red colapsaría a una sola transformación lineal equivalente.

Por eso las activaciones no lineales son indispensables.

### Activaciones típicas

- sigmoidea
- tanh
- logística
- opcionalmente ReLU para experimentos extra

---

## 8. Función de costo

En la clase se menciona que puede usarse **MSE**:

\[
E = \frac{1}{2}\sum_i (\zeta_i - O_i)^2
\]

o, para varios datos, el promedio / suma sobre el conjunto.

Para TP3 en clasificación multiclase también es razonable considerar otras funciones, pero si se quiere mantenerse lo más alineado posible con la clase, MSE es perfectamente aceptable para la implementación base.

---

## 9. Problema del entrenamiento

Con una sola capa es relativamente directo calcular cómo cambiar los pesos.

En multicapa aparece el problema de las **capas ocultas**:

- sabemos el error en la salida
- pero no sabemos directamente cuánto contribuyó cada neurona oculta a ese error

La solución es **backpropagation**.

---

## 10. Backpropagation

Backpropagation usa:

- gradiente descendente
- regla de la cadena

para propagar el error desde la salida hacia las capas ocultas.

La idea es calcular deltas capa por capa y usarlos para actualizar los pesos.

---

## 11. Gradiente descendente: regla general

La actualización general sigue siendo:

\[
w_{nuevo} = w_{anterior} + \Delta w
\]

con:

\[
\Delta w = -\eta \frac{\partial E}{\partial w}
\]

---

## 12. Delta en capa de salida

Para MSE y una neurona de salida, una forma práctica y alineada con la teoría de clase es:

\[
\delta_i^{(M)} = (\zeta_i - O_i)\theta'(z_i^{(M)})
\]

donde:

- `ζ_i`: valor esperado
- `O_i`: salida obtenida
- `θ'`: derivada de la activación

---

## 13. Delta en capa oculta

Para una neurona `j` en una capa oculta `m`:

\[
\delta_j^{(m)} = \theta'(z_j^{(m)}) \sum_i w_{ij}^{(m+1)} \delta_i^{(m+1)}
\]

Esta es la fórmula clave de retropropagación:

- la “culpa” de una neurona oculta depende de
  - la derivada local de su activación
  - la suma ponderada de los deltas de la capa siguiente

---

## 14. Actualización de pesos

Una vez calculado el delta de una neurona, el peso entre la neurona `k` de la capa anterior y la neurona `j` de la capa actual se actualiza como:

\[
\Delta w_{jk}^{(m)} = \eta \delta_j^{(m)} a_k^{(m-1)}
\]

y luego:

\[
w_{jk}^{(m)} \leftarrow w_{jk}^{(m)} + \Delta w_{jk}^{(m)}
\]

Si se usa bias con una entrada constante `1`, se actualiza igual que cualquier otro peso.

---

## 15. Pseudocódigo limpio para agentes

```python
initialize network architecture
initialize all weights to small random values
set learning rate eta

for epoch in range(max_epochs):
    for each training example (x, y):
        # 1) forward
        a[0] = x
        for l in 1..L:
            z[l] = W[l] @ a[l-1] + b[l]
            a[l] = activation(z[l])

        # 2) output delta
        delta[L] = (y - a[L]) * activation_derivative(z[L], a[L])

        # 3) hidden deltas
        for l in reversed(1..L-1):
            delta[l] = (W[l+1].T @ delta[l+1]) * activation_derivative(z[l], a[l])

        # 4) update
        for l in 1..L:
            W[l] = W[l] + eta * outer(delta[l], a[l-1])
            b[l] = b[l] + eta * delta[l]

    evaluate train/validation metrics
    if stopping criterion:
        break
```

### Nota

Si se implementa bias absorbido en la matriz de pesos:

- concatenar `1` al vector de entrada de cada capa
- no hace falta `b[l]` por separado

---

## 16. Estrategias de entrenamiento vistas en clase

La clase distingue:

## a) Online / incremental

Se actualiza después de cada ejemplo.

## b) Mini-batch

Se actualiza después de un subconjunto de ejemplos.

## c) Batch

Se actualiza una vez calculado el delta usando todo el conjunto.

### Para TP3

Conviene diseñar la implementación para soportar al menos:

- online
- batch
- mini-batch si hay tiempo

porque el enunciado pide analizar variantes de mecanismos de optimización y esto encaja perfecto.

---

## 17. Inicialización de pesos

La clase remarca que **no conviene inicializar todos los pesos en cero** porque aparece el **problema de simetría**.

### Recomendación

Inicializar pesos con valores aleatorios pequeños:

- uniforme
- o gaussiana pequeña

---

## 18. Bias en multicapa

La clase agrega una sección extra sobre cómo incorporar bias.

La idea práctica es:

- en cada capa, agregar una entrada constante `1`
- aprender el peso asociado a esa constante
- tratarlo como un peso más en la matriz

Esto simplifica mucho la implementación matricial.

### Pregunta interesante de la clase

¿Hace falta bias en la entrada, en todas las capas o en algunas?

No necesariamente en todas, pero para una implementación general y flexible conviene permitir bias en todas las capas densas.

---

## 19. Arquitectura

No hay receta mágica para elegir “la mejor” arquitectura.

Eso significa que en TP3 hay que explorar.

### Variantes mínimas pedidas por el enunciado

- tasa de aprendizaje
- arquitectura
- mecanismo de optimización

### Parámetros que conviene exponer

- lista de capas, por ejemplo `[64, 32, 10]`
- activación por capa
- bias sí / no
- inicialización
- learning rate
- epochs
- batch size
- optimizer

---

## 20. Validación mínima pedida por el enunciado

Para verificar la implementación, el TP3 sugiere probar con **XOR**:

### Entradas

\[
x = \{(-1, 1), (1, -1), (-1, -1), (1, 1)\}
\]

### Salidas esperadas

\[
y = \{1, 1, -1, -1\}
\]

### Arquitecturas recomendadas para probar a mano

- `[2, 2, 1]`
- `[2, 3, 2, 1]`

Esto es importantísimo porque:

- XOR no lo puede resolver el perceptrón simple escalón
- sí debería poder resolverlo un multicapa correctamente implementado

---

## 21. Aplicación a TP3 ejercicios 2 y 3

## Ejercicio 2

Clasificación de dígitos `0..9` usando:

- `digits.csv` para entrenamiento + ajuste de parámetros e hiperparámetros
- `digits_test.csv` como equivalente a producción / mundo real

## Ejercicio 3

Repetir el estudio con `more_data_digits.csv` para intentar llegar a accuracy >= 98%.

### Qué debería poder hacer la implementación

- entrenar distintas arquitecturas
- comparar optimizadores
- guardar métricas por época
- seleccionar mejor modelo
- cargar y seguir entrenando
- evaluar sobre test final solo al cierre del experimento

---

## 22. Diseño recomendado para agentes

## Componentes mínimos

- `DenseLayer`
- `MLP`
- `forward`
- `backward`
- `update`
- `fit`
- `predict`
- `evaluate`
- `save/load`
- `history tracker`

## Internamente guardar

- `W`
- `b` o pesos con bias absorbido
- `z` por capa
- `a` por capa
- `delta` por capa

---

## 23. Operaciones matriciales

El enunciado recomienda explícitamente usar **operaciones matriciales** para mejorar performance.

Eso es especialmente importante en multicapa.

### Recomendación práctica

Usar `numpy` y vectorizar:

- forward por batch
- cálculo de deltas
- actualización de pesos

---

## 24. Qué métricas registrar por época

- loss train
- accuracy train
- loss valid
- accuracy valid
- tiempo por época
- learning rate actual
- hiperparámetros
- arquitectura

Para clasificación multiclase también conviene guardar:

- matriz de confusión
- accuracy global
- accuracy por clase si se quiere profundizar

---

## 25. Checklist de implementación

- [ ] Soporte para arquitectura arbitraria
- [ ] Bias configurable
- [ ] Feed-forward correcto
- [ ] Backprop correcto
- [ ] Gradiente descendente funcionando
- [ ] Inicialización aleatoria pequeña
- [ ] Historial por época
- [ ] Validación con XOR
- [ ] Soporte de guardado/carga
- [ ] Operaciones matriciales
- [ ] Separación entre entrenamiento, evaluación y análisis

---

## 26. Resumen en 5 líneas

- Tiene **muchas neuronas** organizadas en capas
- Devuelve **1 o varios outputs**, según la tarea
- Se entrena con **feed-forward + backpropagation**
- Usa activaciones **no lineales**
- Es el modelo central para resolver los ejercicios de dígitos del TP3
