# Instituto Tecnológico de Buenos Aires

## Sistemas de Inteligencia Artificial
### Trabajo Práctico 1

# Métodos de Búsqueda

---

## Lado A - Métodos de Búsqueda

## Ejercicio 1

Se cuenta con el ejercicio **8-puzzle** donde se parte de un **tablero inicial** al azar y, moviendo los números adyacentes al espacio vacío, se busca llegar al **tablero solución**.

### Ejemplo tablero inicial

|   |   |   |
|---|---|---|
| 5 | 7 | 3 |
| 8 | 2 |   |
| 1 | 6 | 4 |

### Tablero solución

|   |   |   |
|---|---|---|
| 1 | 2 | 3 |
| 8 |   | 4 |
| 7 | 6 | 5 |

Para este ejercicio, **no es necesaria la implementación**.

Pensar:

- ¿Qué estructura de estado utilizarían?
- Al menos 2 heurísticas admisibles no-triviales.
- ¿Qué métodos de búsqueda utilizarían, con qué heurística, y por qué?

---

## Ejercicio 2

Proponemos 2 juegos de los cuales deberán elegir **1** para implementar un motor de búsqueda de soluciones:

- **Sokoban** ([Wikipedia](https://en.wikipedia.org/wiki/Sokoban))
  - No hay restricción de cantidad de movimientos definido en el problema.
  - Queremos optimizar la cantidad de movimientos.
- **Grid World (multiagente)**

Podemos alterar cada uno de los juegos para que sea más o menos complejo. Les aconsejamos probar con la configuración más sencilla y luego ir incrementando la complejidad.

- **Sokoban**
  - Según la configuración del tablero y la cantidad de cajas/objetivos.
- **Grid World**
  - N agentes y objetivos, y cada configuración de tablero tiene su particularidad.

---

## Implementar y resolver

- **Estructura de estado**
- **Métodos de Búsqueda**
  - BFS
  - DFS
  - Greedy
  - A*
  - IDDFS (opcional)
- **Heurísticas encontradas**
  - Admisibles (al menos 2)
  - No admisibles (opcional)
- **Al finalizar el procesamiento...**
  - Resultado (éxito/fracaso) (si es aplicable)
  - Costo de la solución
  - Cantidad de nodos expandidos
  - Cantidad de nodos frontera
  - Solución (camino desde estado inicial al final)
  - Tiempo de procesamiento
- **Entregable (digital)**
  - Código fuente
  - Presentación
  - Un archivo README explicando cómo ejecutar el motor de búsquedas.
