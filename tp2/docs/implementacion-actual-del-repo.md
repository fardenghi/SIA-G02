# Implementacion actual del repo

## Alcance de este documento

Este archivo resume lo que hoy aparece soportado por el repo segun:

- `main.py`
- `README.md`
- `src/genetic/*.py`
- `src/fitness/mse.py`
- scripts comparativos dentro de `scripts/`

La idea es separar que pide la consigna de lo que efectivamente expone hoy el proyecto.

## Representacion y fitness

Segun `src/genetic/individual.py`:

- un individuo es una lista ordenada de triangulos
- cada triangulo tiene 3 vertices normalizados en `[0, 1]`
- el color se guarda como `RGBA`, con `RGB` en `[0, 255]` y `A` en `[0, 1]`
- el orden de la lista define el orden de renderizado

Segun `src/fitness/mse.py`:

- el error base es MSE
- el fitness se calcula como `1 / (1 + error)`
- opcionalmente puede normalizarse el error antes de convertirlo

## Metodos de seleccion

El modulo `src/genetic/selection.py` implementa:

- `elite`
- `tournament`
- `probabilistic_tournament`
- `roulette`
- `universal`
- `boltzmann`
- `rank` y `ranking`

`main.py` expone estos metodos por CLI mediante `--selection`.

## Metodos de cruza

El modulo `src/genetic/crossover.py` implementa:

- `single_point`
- `two_point`
- `uniform`
- `annular`

`main.py` los expone por CLI mediante `--crossover`.

## Metodos de mutacion

El modulo `src/genetic/mutation.py` implementa cuatro variantes:

- `single_gene`
- `limited_multigen`
- `uniform_multigen`
- `complete`

Tambien define perturbaciones sobre:

- vertices
- color
- alpha
- orden relativo mediante swap de triangulos

### Importante sobre la ejecucion principal

Aunque el motor soporta esos cuatro metodos, `main.py` hoy no expone un selector de metodo de mutacion por CLI.

Ademas, `Config.to_params()` construye `MutationParams` sin elegir otro tipo, por lo que la corrida estandar termina usando el default del modulo: `uniform_multigen`.

### Donde si aparecen todos los metodos

El script `scripts/compare_mutation.py` si invoca explicitamente:

- `single_gene`
- `limited_multigen`
- `uniform_multigen`
- `complete`

## Estrategias de supervivencia

El modulo `src/genetic/survival.py` implementa:

- `additive`
- `exclusive`

El factory `create_engine()` tambien soporta:

- `survival_method`
- `survival_selection_method`
- `offspring_ratio`

### Importante sobre la ejecucion principal

`main.py` hoy no reenvia la configuracion de supervivencia a `create_engine()`.

Eso implica que la corrida principal usa los defaults del factory:

- supervivencia `exclusive`
- seleccion de supervivientes `elite`
- `offspring_ratio = 1.0`

### Donde si aparecen las variantes

El script `scripts/compare_survival.py` si compara distintas combinaciones de:

- `exclusive`
- `additive`
- `offspring_ratio` igual a `1.0`, `1.5` y `2.0`

## Configuracion disponible

El repo tiene configuracion YAML y override por CLI.

En la practica, hoy la interfaz principal expone por CLI:

- imagen
- cantidad de triangulos
- poblacion
- generaciones
- mutation rate
- metodo de seleccion
- metodo de cruza
- salida y guardado intermedio

La configuracion de supervivencia existe en `src/utils/config.py`, pero no se usa todavia en `main.py`.

## Scripts comparativos detectados

Ademas de `main.py`, hoy existen scripts para comparar operadores:

- `scripts/compare_selection.py`
- `scripts/compare_crossover.py`
- `scripts/compare_mutation.py`
- `scripts/compare_survival.py`

## Lectura correcta de estado

Con lo que hoy se ve en el repo, la forma mas precisa de describirlo es:

- la base del motor ya implementa todos los metodos obligatorios de seleccion
- la base del motor ya implementa las cuatro cruzas listadas
- la base del motor ya implementa cuatro variantes de mutacion
- la base del motor ya implementa supervivencia aditiva y exclusiva
- la entrada principal `main.py` expone seleccion y cruza, pero no expone todavia la eleccion de mutacion ni supervivencia

Para documentacion final de entrega, conviene no mezclar estas capas y aclarar siempre si se habla de teoria, de motor interno o del flujo principal de ejecucion.
