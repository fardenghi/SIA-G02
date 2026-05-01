# Lineamientos para agente — Clase 13: Regularización

## Objetivo del documento

Este archivo resume los lineamientos conceptuales y prácticos de la presentación de **Regularización** para que un agente pueda usar esta teoría al desarrollar, revisar o explicar modelos de Machine Learning, especialmente redes neuronales/perceptrones multicapa.

La idea central es que el agente entienda **cuándo aparece overfitting**, **cómo se relaciona con la capacidad del modelo** y **qué técnicas de regularización aplicar o explicar**.

---

# 1. Concepto central: regularización

## Definición

La **regularización** es un conjunto de técnicas diseñadas para reducir el error de **test** o **validación**.

No busca necesariamente minimizar al máximo el error de entrenamiento, sino mejorar la **generalización** del modelo.

En términos simples:

> Regularizar es evitar que el modelo memorice el conjunto de entrenamiento y lograr que funcione mejor con datos nuevos.

## Problema que intenta resolver

La regularización ataca principalmente el **overfitting**.

Hay overfitting cuando:

- el error de entrenamiento es bajo;
- el error de validación/test es alto;
- hay una brecha importante entre entrenamiento y validación;
- el modelo aprendió detalles específicos, ruido o particularidades del dataset de entrenamiento.

---

# 2. Capacidad del modelo

## Definición

La **capacidad** de un modelo es su habilidad o potencial para aproximar una variedad determinada de funciones.

Un modelo con poca capacidad solo puede representar funciones simples.  
Un modelo con mucha capacidad puede representar funciones más complejas.

## Ejemplo de capacidad

- Un **perceptrón simple no lineal** tiene menor capacidad.
- Un **perceptrón multicapa** tiene mayor capacidad.

## Formas de modificar la capacidad

La capacidad puede modificarse mediante:

- elección del modelo;
- arquitectura del modelo;
- cantidad de capas;
- cantidad de neuronas;
- cantidad de características de entrada;
- complejidad de las funciones de activación;
- cantidad de parámetros entrenables.

---

# 3. Underfitting vs Overfitting

## Underfitting

Hay **underfitting** cuando el modelo tiene menor capacidad que la necesaria para capturar la estructura real del problema.

Síntomas:

- error de entrenamiento alto;
- error de validación alto;
- el modelo no aprende bien ni siquiera los datos de entrenamiento;
- el modelo es demasiado simple.

Ejemplo:

> Intentar aproximar una curva compleja usando una recta.

## Overfitting

Hay **overfitting** cuando el modelo tiene más capacidad que la necesaria y empieza a ajustar no solo el patrón general, sino también ruido y detalles particulares del entrenamiento.

Síntomas:

- error de entrenamiento bajo;
- error de validación/test alto;
- gran diferencia entre training y validation;
- mala generalización.

Ejemplo:

> Una red muy grande puede memorizar ejemplos concretos en vez de aprender una regla general.

---

# 4. ¿Dónde ayudan las técnicas de regularización?

Las técnicas de regularización ayudan principalmente en la zona de **overfitting**.

Cuando el modelo tiene demasiada capacidad, puede aprender demasiado bien el conjunto de entrenamiento. En ese caso, el error de entrenamiento puede seguir bajando, pero el error de validación/test empieza a subir.

La regularización intenta reducir esa brecha haciendo que el modelo sea:

- menos sensible al ruido;
- menos dependiente de detalles particulares del dataset;
- menos complejo en términos efectivos;
- más robusto ante datos nuevos.

Importante:

> Si el problema es underfitting, regularizar más suele empeorar el modelo, porque ya era demasiado simple.

---

# 5. Métodos de regularización explicados en la presentación

La presentación desarrolla principalmente estos tres métodos:

1. Early Stopping
2. Data Augmentation
3. L2 Penalty Norm / Weight Decay

También menciona otros métodos adicionales:

- Dropout
- Modelos de ensamble
- Aprendizaje semi-supervisado
- Entrenamiento adversarial

---

# 6. Early Stopping

## Idea central

**Early Stopping** consiste en detener el entrenamiento antes de que el modelo empiece a sobreajustar.

Durante el entrenamiento suele pasar lo siguiente:

- el error de entrenamiento sigue bajando;
- el error de validación baja al principio;
- después de cierto punto, el error de validación empieza a subir.

Ese punto marca que el modelo ya no está aprendiendo patrones generales, sino detalles demasiado específicos del entrenamiento.

## Lineamiento para el agente

Cuando el agente implemente o recomiende Early Stopping, debe:

- separar datos en entrenamiento y validación;
- monitorear la métrica de validación en cada época;
- guardar el mejor modelo según validación;
- detener el entrenamiento si la métrica de validación no mejora durante cierta cantidad de épocas;
- no elegir el modelo final solo por menor error de entrenamiento.

## Parámetros importantes

- `patience`: cantidad de épocas sin mejora antes de cortar.
- `min_delta`: mejora mínima necesaria para considerar que hubo progreso.
- métrica monitoreada: pérdida de validación, accuracy de validación, F1, etc.

## Cuándo usarlo

Usar Early Stopping cuando:

- el modelo empieza a sobreajustar al entrenar muchas épocas;
- hay una curva de validación que empeora después de cierto punto;
- se quiere evitar entrenamiento innecesario.

## Qué evitar

No cortar solo porque el error de entrenamiento dejó de bajar.  
La decisión debe basarse principalmente en validación.

---

# 7. Data Augmentation

## Idea central

**Data Augmentation** consiste en generar más ejemplos de entrenamiento a partir de transformaciones sobre los datos existentes.

Es especialmente usado en clasificación de imágenes y objetos.

## Transformaciones mencionadas

- ruido gaussiano;
- rotaciones;
- traslaciones;
- cambios de escala;
- deformaciones leves;
- otras transformaciones que preserven la clase.

## Intuición

El objetivo es que el modelo vea más variabilidad durante el entrenamiento.

Así evita depender demasiado de ejemplos exactos y aprende patrones más generales.

Ejemplo:

> Para reconocer dígitos, se pueden generar versiones del mismo número un poco rotadas, desplazadas o con ruido.

## Regla clave

La transformación **no debe cambiar la etiqueta real del dato**.

Ejemplo de cuidado:

- rotar levemente un `6` puede estar bien;
- rotarlo demasiado puede hacerlo parecer un `9`, lo cual cambia el significado del dato.

## Lineamiento para el agente

Cuando el agente use Data Augmentation, debe:

- aplicar transformaciones realistas para el dominio;
- preservar la clase original;
- evitar transformaciones demasiado agresivas;
- justificar por qué cada transformación tiene sentido;
- revisar visualmente ejemplos aumentados si se trabaja con imágenes.

## Cuándo usarlo

Usar Data Augmentation cuando:

- hay pocos datos;
- el modelo sobreajusta;
- se trabaja con imágenes, señales, audio, texto o datos donde se puedan crear variaciones válidas;
- se quiere mejorar robustez ante pequeñas variaciones.

---

# 8. L2 Penalty Norm / Weight Decay

## Idea central

La regularización **L2** agrega una penalización a la función de error para castigar pesos grandes.

La función regularizada queda:

```math
E_{reg}(w) = E(w) + \frac{1}{2}\lambda ||w||^2
```

Donde:

- `E(w)` es el error original;
- `w` son los pesos del modelo;
- `λ` es el hiperparámetro de regularización;
- `||w||²` mide el tamaño de los pesos.

## Intuición

Pesos muy grandes pueden hacer que el modelo sea demasiado sensible a pequeñas variaciones en la entrada.

L2 empuja los pesos a valores más chicos, haciendo que el modelo sea más estable y menos propenso a memorizar ruido.

## Gradiente

La derivada del error regularizado queda:

```math
\frac{\partial E_{reg}}{\partial w} = \frac{\partial E}{\partial w} + \lambda w
```

Esto significa que al gradiente original se suma un término que empuja los pesos hacia cero.

## Actualización de pesos

La actualización puede escribirse como:

```math
w = w - \eta \left(\frac{\partial E}{\partial w} + \lambda w\right)
```

Reordenando:

```math
w = (1 - \eta \lambda)w - \eta \frac{\partial E}{\partial w}
```

Por eso también se llama **Weight Decay**: en cada actualización, los pesos tienden a achicarse.

## Parámetro λ

`λ` controla la fuerza de la regularización.

- `λ` muy bajo: casi no regulariza.
- `λ` adecuado: reduce overfitting y mejora generalización.
- `λ` muy alto: puede generar underfitting.

## Lineamiento para el agente

Cuando el agente use L2 / Weight Decay, debe:

- agregar el término de penalización al error o usar el parámetro correspondiente del optimizador;
- probar distintos valores de `λ`;
- evaluar siempre contra validación;
- evitar valores excesivos que impidan aprender;
- explicar que L2 achica pesos, pero no los elimina completamente.

## Cuándo usarlo

Usar L2 cuando:

- el modelo sobreajusta;
- los pesos tienden a crecer mucho;
- se quiere una solución más suave y estable;
- se usan redes neuronales, regresión logística, regresión lineal u otros modelos parametrizados.

---

# 9. Otros métodos mencionados

## Dropout

Consiste en apagar aleatoriamente algunas neuronas durante el entrenamiento.

Objetivo:

- evitar que la red dependa demasiado de neuronas específicas;
- forzar representaciones más distribuidas;
- mejorar robustez.

Lineamiento:

> Usar dropout como regularización en redes neuronales, especialmente si hay overfitting y muchas neuronas/parámetros.

## Modelos de ensamble

Consisten en combinar varios modelos para obtener una predicción más estable.

Ejemplos:

- bagging;
- boosting;
- random forests;
- promedios de modelos;
- votación por mayoría.

Objetivo:

- reducir varianza;
- mejorar generalización;
- hacer predicciones más robustas.

## Aprendizaje semi-supervisado

Usa datos etiquetados y datos no etiquetados.

Objetivo:

- aprovechar información de datos sin label;
- mejorar aprendizaje cuando hay pocos datos etiquetados.

La presentación menciona que esta idea sobrevive hoy en el contexto de LLMs.

## Entrenamiento adversarial

Consiste en entrenar con ejemplos perturbados o difíciles, diseñados para hacer fallar al modelo.

Objetivo:

- mejorar robustez;
- reducir sensibilidad a perturbaciones;
- preparar al modelo para casos difíciles.

---

# 10. Criterios de decisión para el agente

## Si el modelo tiene underfitting

Síntomas:

- alto error de entrenamiento;
- alto error de validación.

Acciones recomendadas:

- aumentar capacidad del modelo;
- usar más capas o neuronas;
- agregar mejores features;
- entrenar más tiempo;
- reducir regularización si ya se está usando.

Evitar:

- agregar más regularización fuerte;
- aplicar weight decay excesivo;
- usar dropout alto;
- cortar demasiado temprano con early stopping.

## Si el modelo tiene overfitting

Síntomas:

- bajo error de entrenamiento;
- alto error de validación;
- gran gap entre train y validation.

Acciones recomendadas:

- usar Early Stopping;
- agregar Data Augmentation;
- aplicar L2 / Weight Decay;
- considerar Dropout;
- reducir capacidad del modelo;
- conseguir más datos;
- usar ensambles si corresponde.

---

# 11. Checklist práctico para el agente

Antes de proponer una técnica, el agente debe revisar:

- ¿El problema es underfitting u overfitting?
- ¿Cómo se comportan las curvas de entrenamiento y validación?
- ¿El error de entrenamiento es alto o bajo?
- ¿El error de validación es alto o bajo?
- ¿Hay gap de generalización?
- ¿El modelo tiene demasiada o muy poca capacidad?
- ¿Hay suficientes datos?
- ¿Las transformaciones de Data Augmentation preservan la clase?
- ¿El valor de `λ` es razonable?
- ¿Early Stopping está mirando validación y no solo entrenamiento?

---

# 12. Resumen de métodos

| Método | Qué hace | Cuándo usarlo | Riesgo |
|---|---|---|---|
| Early Stopping | Corta el entrenamiento antes del sobreajuste | Cuando validación empieza a empeorar | Cortar demasiado temprano |
| Data Augmentation | Genera nuevos datos transformados | Cuando faltan datos o hay overfitting | Cambiar accidentalmente la clase |
| L2 / Weight Decay | Penaliza pesos grandes | Cuando el modelo tiene pesos grandes o sobreajusta | `λ` muy alto puede causar underfitting |
| Dropout | Apaga neuronas aleatoriamente | Redes con muchas neuronas/parámetros | Dropout muy alto puede impedir aprendizaje |
| Ensambles | Combina modelos | Cuando se quiere reducir varianza | Mayor costo computacional |
| Semi-supervisado | Usa datos sin etiquetar | Cuando hay pocos labels | Puede propagar errores si pseudo-labels son malos |
| Adversarial Training | Entrena con ejemplos difíciles | Cuando se busca robustez | Puede ser costoso o degradar performance limpia |

---

# 13. Frases clave para mantener consistencia

- Regularizar no significa mejorar el error de entrenamiento, sino mejorar la generalización.
- El objetivo principal es reducir el error de validación/test.
- La regularización ayuda sobre todo contra overfitting.
- Si hay underfitting, el problema suele ser falta de capacidad, no exceso.
- Data Augmentation solo sirve si las transformaciones preservan el significado del dato.
- L2 no elimina pesos: los achica.
- Early Stopping debe decidirse mirando validación.
- Un modelo más grande no siempre es mejor: puede memorizar ruido.

---

# 14. Errores conceptuales que el agente debe evitar

- Decir que regularización siempre mejora el modelo.
- Decir que regularización sirve principalmente para underfitting.
- Confundir bajo error de entrenamiento con buen modelo.
- Aplicar Data Augmentation sin verificar que la clase se mantenga.
- Usar L2 con `λ` demasiado alto sin advertir riesgo de underfitting.
- Cortar entrenamiento mirando solo training loss.
- Recomendar aumentar capacidad cuando ya hay overfitting sin agregar controles.
- Recomendar reducir capacidad cuando hay underfitting sin revisar las métricas.

---

# 15. Guía rápida de respuesta para el agente

Si le preguntan “¿qué es regularización?”:

> Es un conjunto de técnicas para reducir el error de validación/test y mejorar la generalización, evitando que el modelo memorice el entrenamiento.

Si le preguntan “¿dónde ayuda?”:

> Ayuda principalmente en overfitting, cuando el modelo tiene demasiada capacidad y aprende ruido o detalles específicos del training set.

Si le preguntan “¿qué métodos vimos?”:

> Early Stopping, Data Augmentation y L2 / Weight Decay como principales. También se mencionan Dropout, Ensambles, Semi-supervisado y Entrenamiento Adversarial.

Si le preguntan “¿qué hago si hay underfitting?”:

> Aumentar capacidad, mejorar features, entrenar más o reducir regularización.

Si le preguntan “¿qué hago si hay overfitting?”:

> Aplicar regularización: Early Stopping, Data Augmentation, L2/Weight Decay, Dropout, más datos o reducción de capacidad.
