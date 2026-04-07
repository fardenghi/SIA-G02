# Guia de diseno y defensa

## Objetivo de esta guia

La consigna no pide solo que el sistema funcione. Tambien pide poder justificar por que la representacion, el fitness y los operadores elegidos tienen sentido para este problema.

## Como bajar la teoria al problema de triangulos

### Que es un individuo

La respuesta mas natural para este TP es:

- un individuo es una aproximacion completa de la imagen
- el cromosoma es una lista ordenada de `N` triangulos
- el orden importa porque define el orden de pintado

### Cuales pueden ser los genes

Cada triangulo puede verse como un gen o como un bloque de genes. Como minimo, cada triangulo necesita:

- tres vertices
- color
- transparencia
- posicion relativa dentro de la lista si el orden de renderizado forma parte del fenotipo

### Que es el fenotipo

El fenotipo es la imagen renderizada que se obtiene al dibujar todos los triangulos sobre fondo blanco.

## Como evaluar la aproximacion

Antes de correr experimentos conviene responder esto por escrito:

- que error visual se mide
- si menor error equivale a mayor fitness o si se aplica una transformacion
- como se comparan dos individuos de manera deterministica

Una opcion simple y defendible es usar MSE como error base y convertirlo a fitness con una formula monotona, por ejemplo `1 / (1 + error)`.

## Como justificar la cruza

La cruza elegida deberia respetar la estructura del individuo.

- si se quiere preservar bloques contiguos de triangulos, conviene un punto, dos puntos o anular
- si se quiere mezclar mas fuerte entre padres, conviene uniforme
- si el orden de los triangulos es importante, hay que justificar cuanto conviene romperlo

La pregunta clave no es solo como se cruzan dos padres, sino si esa recombinacion tiene chances razonables de producir hijos utiles.

## Como justificar la mutacion

Hay dos decisiones distintas:

- cuantos genes cambian
- como cambia cada gen

En este problema, mutar un gen puede significar:

- mover vertices
- alterar color
- alterar alpha
- intercambiar orden de triangulos

Conviene explicar si se busca una mutacion mas conservadora o mas agresiva, y por que.

## Criterios de corte defendibles

Una combinacion razonable para este TP es usar:

- maximo de generaciones
- estancamiento del mejor fitness
- opcion de solucion aceptable si se alcanza cierto umbral

Eso cubre tanto control de costo como calidad minima.

## Version minima para arrancar

La teoria y la consigna recomiendan empezar con una version barata de evaluar:

- imagenes simples
- pocas generaciones
- pocos triangulos
- resolucion baja o reducida
- poblacion moderada

Esto permite validar si el motor converge antes de pasar a imagenes mas complejas.

## Que medir en experimentos

Para la defensa conviene guardar al menos:

- fitness del mejor individuo por generacion
- fitness promedio por generacion
- error final
- tiempo total
- generaciones hasta la mejor solucion
- imagenes intermedias o checkpoints visuales

Si quieren enriquecer el analisis, tambien pueden medir diversidad o cantidad de mejoras significativas.

## Sobre implementacion parcial

Para evaluar el motor, una implementacion parcial sirve.

Para cumplir la entrega, no alcanza.

La estrategia sensata es:

1. validar primero el ciclo basico con una configuracion chica
2. medir convergencia en casos simples
3. completar toda la cobertura pedida por la consigna
4. recien despues comparar operadores y ajustar hiperparametros

## Checklist para la defensa

- explicar claramente que representa el individuo
- explicar por que el fitness refleja calidad de aproximacion
- justificar cruza y mutacion elegidas
- mostrar por que la supervivencia elegida tiene sentido
- justificar el criterio de corte
- mostrar metricas y ejemplos visuales
- aclarar que partes son requerimiento obligatorio y que partes son decisiones del equipo
