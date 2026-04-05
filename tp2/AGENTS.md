# AGENTS.md — TP2 SIA: Algoritmos Genéticos

## Contexto
Este repositorio corresponde al **Trabajo Práctico 2 de Sistemas de Inteligencia Artificial (ITBA)** sobre **Algoritmos Genéticos**.

El problema a resolver es una aproximación evolutiva de imágenes: dado un objetivo visual, el motor de AG debe construir una imagen compuesta por **triángulos sobre un canvas blanco** que se parezca lo más posible a la imagen de entrada.

---

## Objetivo del TP
Implementar un **motor de Algoritmos Genéticos** que, dada una imagen y una cantidad de triángulos, encuentre una aproximación visual de buena calidad usando individuos compuestos por triángulos.

Cada triángulo puede tener:
- posición
- geometría
- color uniforme
- transparencia (por ejemplo RGBA / HSLA)

El fondo del canvas es **blanco**.

---

## Inputs esperados
El programa debe aceptar como entrada:

- **Imagen objetivo**
- **Cantidad de triángulos** a utilizar
- **Hiperparámetros del AG**

---

## Outputs esperados
El programa debe producir como salida:

- **Imagen generada**
- **Enumeración de triángulos** usados en la solución (posición, color, etc.)
- **Métricas para análisis**, por ejemplo:
  - fitness
  - error
  - generaciones
  - cualquier otra métrica útil para defender la implementación

---

## Requerimientos obligatorios

### 1) Métodos de selección
Deben implementarse los métodos vistos en clase:

- **Elite**
- **Ruleta**
- **Universal**
- **Boltzmann**
- **Torneos**
  - determinístico
  - probabilístico
- **Ranking**

> No alcanza con implementar solo algunos. La consigna pide explícitamente estos métodos.

---

### 2) Estrategias de supervivencia / formación de nuevas generaciones
Deben implementarse ambas:

- **Supervivencia Aditiva**
- **Supervivencia Exclusiva**

---

### 3) Criterios de corte
Hay que **decidir y justificar** cómo termina la ejecución.

Ejemplos válidos mencionados en teoría / consigna:
- máximo de generaciones
- criterio por estructura
- criterio por contenido
- solución aceptable
- tiempo

No hace falta usar todos, pero sí debe estar **justificado**.

---

### 4) Estructura del individuo y fitness
Hay que **justificar**:

- qué representa un individuo
- cuáles son sus genes
- cómo se codifica el genotipo
- cómo se evalúa la calidad de la aproximación
- qué define el fitness

La representación elegida debe ser coherente con el problema de reconstrucción con triángulos.

---

### 5) Métodos de cruza
Hay que decidir qué cruza usar en distintas circunstancias y **por qué**.

Deben implementarse **al menos 2** de estos métodos:

- **Cruce de un punto**
- **Cruce de dos puntos**
- **Cruce uniforme**
- **Cruce anular**

---

### 6) Métodos de mutación
Hay que decidir qué mutación usar en distintas circunstancias y **por qué**.

Deben implementarse **al menos 2** de estos métodos:

- **Gen**
- **MultiGen**
- **Uniforme**
- **No Uniforme**

---

## Restricciones importantes

- **Se pueden usar librerías externas para manejo de imágenes**.
- **No se pueden usar librerías externas para implementar Algoritmos Genéticos**.

Es decir:
- sí: render, lectura/escritura de imágenes, canvas, utilidades matemáticas
- no: frameworks de GA que resuelvan selección/cruza/mutación/supervivencia por afuera del código propio

---

## Entregables
La entrega digital debe incluir:

- **Código fuente**
- **Presentación**
- **README** explicando cómo ejecutar el programa

---

## Preguntas guía que el equipo debería poder responder
Antes de experimentar, la consigna sugiere contestar estas preguntas:

1. **¿Cómo evaluamos la aproximación al dibujo?**
2. **¿Qué es un individuo en este problema?**
3. **¿Cuáles son sus genes?**
4. **¿Qué es el fitness en este problema?**
5. **¿Cómo muta un individuo?**
6. **¿Cómo se cruzan dos individuos para obtener descendencia?**
7. **¿Esa cruza produce hijos con chances razonables de mejorar?**
8. **¿Cuál es la versión más simple del problema para arrancar?**
9. **¿Cómo afectan el tipo de imagen y la cantidad de triángulos a la performance?**
10. **¿Alcanza con una implementación parcial para evaluar el motor antes de completar todo?**

---

## Alcance práctico sugerido
Para desarrollar más rápido y poder iterar:

- empezar con imágenes simples
- usar pocos triángulos al principio
- medir tiempo y error
- validar primero que el motor converge en casos sencillos
- recién después escalar a imágenes más complejas

Ejemplos de imágenes sugeridas por la consigna:
- banderas
- siluetas
- pictogramas
- señales
- logos simples
- íconos
- mapas esquemáticos
- símbolos
- emojis

---

## Criterio de cumplimiento
Una implementación debe considerarse alineada con la consigna solo si:

- resuelve el problema de aproximar imágenes con triángulos
- permite configurar el AG con hiperparámetros razonables
- implementa todos los métodos de selección obligatorios
- implementa ambas supervivencias obligatorias
- implementa al menos 2 cruzas obligatorias
- implementa al menos 2 mutaciones obligatorias
- produce salidas visuales y métricas defendibles
- tiene README claro de ejecución

---

## Qué debería priorizar cualquier agente que trabaje sobre este repo
Si vas a modificar o completar este proyecto, priorizá en este orden:

1. **Cumplir la consigna al pie de la letra**
2. **Mantener una API/configuración clara y consistente**
3. **Evitar romper reproducibilidad y métricas**
4. **Justificar cada decisión de diseño del AG**
5. **No introducir dependencias que implementen AG por afuera**

---

## Resumen ejecutivo
Este TP no es solamente “hacer que funcione”. También hay que poder **defender la implementación**.

Por eso el proyecto debe dejar claros:
- la representación del individuo
- la función de fitness
- la motivación de selección/cruza/mutación
- la estrategia de supervivencia
- el criterio de corte
- las métricas observadas en los experimentos

---

## Fuente de verdad para este archivo
Este archivo resume la consigna del TP2 y los contenidos de teoría vistos en clase sobre Algoritmos Genéticos. Si hay dudas, priorizar siempre:

1. la **consigna oficial del TP2**
2. el material de **Algoritmos Genéticos** visto en clase
3. luego, las decisiones de implementación del equipo
