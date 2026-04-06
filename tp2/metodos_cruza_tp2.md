# Métodos de cruza según la teoría

Este resumen está basado en la teoría de **Algoritmos Genéticos** vista en clase, en la sección de **Crossover / Cruza / Recombinación**.

## Qué es la cruza

Una vez seleccionados los individuos, se los aparea y se realiza la **recombinación de genes** para generar descendencia.

Según la teoría, los tipos de crossover vistos son:

- Cruce de un punto
- Cruce de dos puntos
- Cruce anular
- Cruce uniforme

---

## 1. Cruce de un punto

### Idea
Se elige un **locus** al azar y se intercambian los alelos a partir de ese punto.

### Parámetro
- `P ∈ [0, S-1]`
- `S`: cantidad de genes

### Cómo funciona
Dado un punto de corte `P`, los genes anteriores a `P` se mantienen y los genes desde `P` en adelante se intercambian entre ambos padres.

### Ejemplo
Padres:

- `X1: 011011001110`
- `X2: 101001110010`

Con `P = 4`, los hijos quedan:

- `X3: 011001110010`
- `X4: 101011001110`

### Intuición
Es una cruza simple y clásica. Conserva bloques contiguos de genes, lo que puede ser útil cuando la posición de los genes importa.

---

## 2. Cruce de dos puntos

### Idea
Se eligen **dos locus** al azar y se intercambian los alelos entre esos dos puntos.

### Parámetros
- `P1 ∈ [0, S-1]`
- `P2 ∈ [0, S-1]`
- con `P1 ≤ P2`

### Cómo funciona
Se toma el segmento comprendido entre `P1` y `P2` y se intercambia entre ambos padres.

### Ejemplo
Padres:

- `X1: 011011001110`
- `X2: 101001110010`

Con `P1 = 4` y `P2 = 6`, los hijos quedan:

- `X3: 011001001110`
- `X4: 101011110010`

### Intuición
Preserva todavía más estructura que el cruce de un punto en ciertas configuraciones, porque intercambia solo una porción interna del cromosoma.

---

## 3. Cruce anular

### Idea
Se elige un locus inicial `P` y una longitud `L`. Luego se intercambia el segmento de longitud `L` a partir de `P`, considerando el cromosoma como circular.

### Parámetros
- `P ∈ [0, S-1]`
- `L ∈ [0, ⌈S/2⌉]`

### Cómo funciona
A diferencia de la cruza de uno o dos puntos, acá el segmento puede “envolver” el final del cromosoma y continuar desde el inicio.

### Ejemplo
Padres:

- `X1: 011011001110`
- `X2: 101001110010`

Con `P = 11` y `L = 5`, los hijos quedan:

- `X3: 101011001110`
- `X4: 011001110010`

### Intuición
Sirve cuando no querés depender tanto de la linealidad estricta del cromosoma y querés permitir intercambios que atraviesen el final e inicio del arreglo.

---

## 4. Cruce uniforme

### Idea
En cada gen, se decide de manera independiente si se intercambian o no los alelos entre ambos padres.

### Parámetro
- `P ∈ [0, 1]`
- por lo general `P = 0.5`

### Cómo funciona
Cada posición del cromosoma se evalúa por separado:

- si se mantiene, el hijo conserva el alelo del padre correspondiente
- si se intercambia, toma el alelo del otro padre

### Ejemplo
Padres:

- `X1: 0 1 1 0 1 1 0 0 1 1 1 0`
- `X2: 1 0 1 0 0 1 1 1 0 0 1 0`

Con una máscara de intercambio:

- `Pi: > > < < > < > > < > < <`

Los hijos quedan:

- `X3: 0 1 1 0 1 1 0 0 0 1 1 0`
- `X4: 1 0 1 0 0 1 1 1 1 0 1 0`

### Nota de la teoría
La teoría marca que este es el **único tipo de cruce visto que no mantiene correlación posicional entre alelos**.

### Intuición
Mezcla mucho más los padres. Aumenta la exploración, pero puede romper bloques de genes que funcionaban bien juntos.

---

## Probabilidad de recombinación

La teoría también menciona que es común implementar una **probabilidad de recombinación** `Pc`.

### Idea
- Con probabilidad `Pc`, se aplica la cruza.
- Si no se recombinan los genes, los hijos son una **copia idéntica de los padres**, aunque luego todavía pueden pasar por la etapa de mutación.

Esto permite controlar cuánta mezcla genética se introduce en cada generación.

---

## Comparación rápida

| Método | Qué intercambia | Qué conserva | Nivel de mezcla |
|---|---|---|---|
| Un punto | Todo desde un punto en adelante | Bloques contiguos grandes | Medio |
| Dos puntos | Segmento entre dos puntos | Bloques externos e internos | Medio |
| Anular | Segmento circular desde un punto | Bloques con envoltura circular | Medio |
| Uniforme | Cada gen por separado | Muy poca estructura posicional | Alto |

---

## Observación importante de la teoría

En los ejemplos de clase se **asume un bit por gen**, pero la propia teoría aclara que eso **no siempre es correcto** y que muchas veces no lo es.

En problemas reales, un gen puede representar algo más complejo que un solo bit, así que la implementación concreta de la cruza debe respetar la estructura real del cromosoma.
