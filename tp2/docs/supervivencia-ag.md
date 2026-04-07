# Supervivencia en AG

Resumen teorico detallado de las estrategias de supervivencia relevantes para el TP.

Ver tambien:

- `marco-teorico-ag.md`
- `guia-diseno-y-defensa.md`

## Idea general

La supervivencia define **cómo se forma la nueva generación** a partir de:
- la población actual
- los hijos generados por cruza y mutación

En la teoría vista, se presentan **dos estrategias principales de supervivencia**:

- **Supervivencia aditiva**
- **Supervivencia exclusiva**

Además, aparece la idea de **brecha generacional**, que indica cuántos individuos se reemplazan entre generaciones.

---

## 1. Supervivencia aditiva

### Definición

En la supervivencia aditiva, la nueva generación se forma seleccionando `N` individuos del conjunto compuesto por:

- los `N` individuos de la generación actual
- los `K` hijos generados

Es decir, se arma un conjunto combinado de tamaño `N + K` y de ahí se eligen los `N` sobrevivientes.

### Esquema

Si la población actual tiene tamaño `N` y se generan `K` hijos:

Nueva generación = seleccionar `N` individuos de:

`[ población actual (N) + hijos (K) ]`

### Idea intuitiva

Compiten por sobrevivir:
- padres
- hijos

Por eso, **los mejores individuos pueden mantenerse**, aunque pertenezcan a la generación anterior.

### Ventaja conceptual

- Permite **preservar buenos individuos** ya existentes.
- Reduce el riesgo de perder soluciones buenas por una mala generación de hijos.

### Observación

La teoría también remarca que para elegir esos `N` sobrevivientes se pueden usar métodos de selección, igual que para padres.

---

## 2. Supervivencia exclusiva

### Definición

En la supervivencia exclusiva, **no siempre compiten juntos padres e hijos** del mismo modo. La nueva generación se arma según la relación entre `K` y `N`.

### Casos

#### Caso 1: `K > N`

La nueva generación se genera seleccionando `N` individuos **solo de entre los `K` hijos**.

Es decir:

Nueva generación = seleccionar `N` de los `K` hijos

En este caso, la generación anterior no participa directamente en la supervivencia.

---

#### Caso 2: `K ≤ N`

La nueva generación se forma con:

- los `K` hijos generados
- más `N - K` individuos seleccionados de la generación actual

Es decir:

Nueva generación =
- `K` hijos
- `N - K` individuos de la población actual

### Idea intuitiva

Los hijos entran sí o sí, y el resto de lugares se completa con individuos de la generación anterior.

### Diferencia clave con aditiva

- En **aditiva**, todos compiten juntos dentro de un pool `N + K`.
- En **exclusiva**, la forma de construir la nueva generación depende de cuántos hijos hay y puede forzar la entrada de descendencia.

---

## 3. Brecha generacional

### Definición

La brecha generacional determina **qué proporción de la población cambia de una generación a otra**.

Se define con un valor `G`.

### Casos

- **`G = 1`**: toda la población es reemplazada.
- **`G = 0`**: ningún individuo es reemplazado.
- **`G ∈ (0,1)`**: la nueva generación se compone de:
  - `(1 - G) * N` individuos de la generación anterior
  - `G * N` individuos generados

### Interpretación

La brecha generacional controla la intensidad del reemplazo:

- valores altos de `G` ⇒ más recambio, más exploración
- valores bajos de `G` ⇒ más conservación, más estabilidad

---

## 4. Comparación rápida

### Supervivencia aditiva
- Junta padres e hijos en un mismo conjunto.
- Luego selecciona `N` sobrevivientes.
- Favorece que no se pierdan buenos individuos previos.

### Supervivencia exclusiva
- Distingue más fuertemente entre padres e hijos.
- Puede hacer que los hijos entren obligatoriamente.
- El mecanismo depende de la cantidad de hijos `K`.

### Brecha generacional
- No es exactamente un tipo de supervivencia separado, sino una forma de controlar cuánto reemplazo hay entre generaciones.

---

## 5. Qué conviene recordar para el TP

Si te lo preguntan en defensa, lo importante es decir:

- **La supervivencia define cómo se arma la nueva población.**
- **Aditiva**: se eligen `N` individuos del conjunto `padres + hijos`.
- **Exclusiva**:
  - si `K > N`, se eligen `N` de los hijos
  - si `K ≤ N`, entran los `K` hijos y se completa con `N-K` individuos de la población actual
- **Brecha generacional**: controla qué fracción de la población se reemplaza entre generaciones.

---

## Fuente teórica

Resumen armado a partir de la teoría de **Algoritmos Genéticos**, en la sección de implementación y supervivencia:
- **Supervivencia Aditiva**
- **Supervivencia Exclusiva**
- **Brecha Generacional**
