# Consigna TP2

## Alcance general

El TP2 pide implementar el ejercicio 2: un motor de algoritmos geneticos que reciba una imagen y la aproxime usando triangulos sobre un canvas blanco.

El ejercicio 1 aparece como disparador conceptual sobre arte ASCII, pero no hace falta implementarlo.

## Objetivo del ejercicio 2

La solucion debe tomar una imagen objetivo y una cantidad de triangulos, y buscar la mejor aproximacion posible usando triangulos de color uniforme, potencialmente traslucidos.

Los parametros del problema son:

- imagen objetivo
- cantidad de triangulos

Los hiperparametros del AG quedan a criterio del equipo, pero deben poder justificarse.

## Inputs esperados

- imagen objetivo
- cantidad de triangulos
- hiperparametros de la implementacion de algoritmos geneticos

## Outputs esperados

- imagen generada
- enumeracion de triangulos usados en la solucion
- metricas para defender la implementacion, por ejemplo fitness, error, generaciones y tiempo

## Requerimientos obligatorios

### Metodos de seleccion

La consigna pide explicitamente implementar todos los metodos vistos en clase:

- Elite
- Ruleta
- Universal
- Boltzmann
- Torneos
- Ranking

En torneos deben estar las dos variantes:

- deterministico
- probabilistico

### Estrategias de supervivencia

Hay que implementar ambas:

- supervivencia aditiva
- supervivencia exclusiva

### Criterios de corte

Hay que decidir y justificar como termina la ejecucion. La teoria menciona alternativas como:

- tiempo
- cantidad maxima de generaciones
- solucion aceptable
- criterio por estructura
- criterio por contenido

No hace falta usar todas, pero si justificar la eleccion.

### Individuo y fitness

Hay que justificar:

- que representa un individuo
- cuales son sus genes
- como se codifica el genotipo
- como se evalua la calidad de la aproximacion
- que define el fitness

La representacion tiene que ser coherente con el problema de reconstruccion con triangulos.

### Metodos de cruza

Hay que decidir que cruza usar segun el caso y por que. Deben implementarse al menos 2 de estos metodos:

- cruce de un punto
- cruce de dos puntos
- cruce uniforme
- cruce anular

### Metodos de mutacion

Hay que decidir que mutacion usar segun el caso y por que. Deben implementarse al menos 2 metodos.

La consigna lista:

- Gen
- MultiGen
- Uniforme
- No Uniforme

La teoria de clase, en cambio, desarrolla variantes como:

- mutacion de gen
- mutacion multigen limitada
- mutacion multigen uniforme
- mutacion completa

Conviene dejar muy clara en la documentacion final la equivalencia o decision adoptada, para no generar ambiguedades en la defensa.

## Restricciones

- se pueden usar librerias externas para manejo de imagenes
- no se pueden usar librerias externas para implementar algoritmos geneticos

## Entregables

- codigo fuente
- presentacion
- `README` explicando como ejecutar el programa

## Preguntas guia que el equipo deberia poder responder

- Como evaluar la aproximacion al dibujo.
- Que es un individuo y cuales son sus genes.
- Que es el fitness en este problema.
- Como muta un individuo.
- Como se cruzan dos individuos y si esa cruza produce descendencia razonable.
- Cual es la version mas simple del problema para arrancar.
- Como afectan el tipo de imagen y la cantidad de triangulos a la performance.
- Si una implementacion parcial alcanza para evaluar el motor antes de completar la entrega.

## Consejos practicos que aparecen en la consigna

Se recomienda empezar con imagenes sencillas, por ejemplo:

- banderas
- siluetas
- pictogramas
- senales
- logos simples
- iconos
- mapas esquematicos
- simbolos
- emojis

## Opcionales mencionados

- usar la cantidad de triangulos como cota maxima y combinarla con un error minimo aceptable
- usar otros poligonos u ovalos
- probar otros metodos de cruza o mutacion

## Checklist rapido

- el problema resuelto es realmente aproximacion de imagenes con triangulos
- todos los metodos de seleccion obligatorios existen
- existen supervivencia aditiva y exclusiva
- existen al menos dos cruzas obligatorias
- existen al menos dos mutaciones justificadas
- hay criterio de corte justificado
- hay metricas y salidas defendibles
- el `README` de ejecucion esta claro
