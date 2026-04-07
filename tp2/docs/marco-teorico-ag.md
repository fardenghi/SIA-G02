# Marco teorico de AG

## Idea general

Un algoritmo genetico es un metodo de optimizacion inspirado en seleccion natural y evolucion. Parte de una poblacion de soluciones candidatas y repite un ciclo de:

1. evaluar fitness
2. seleccionar individuos
3. recombinar
4. mutar
5. formar una nueva generacion

La idea de fondo es que, si la representacion y los operadores estan bien elegidos, combinar buenos estados puede producir otros mejores.

## Componentes basicos

La teoria remarca estos componentes:

- estructura o arquitectura del genotipo
- poblacion inicial
- funcion de aptitud o fitness
- metodo de seleccion de padres
- metodo de cruza
- metodo de mutacion
- metodo de seleccion de la nueva generacion
- condicion de corte

## Nomenclatura util

- Genotipo: estructura codificada del individuo.
- Locus: posicion de un gen dentro del cromosoma.
- Cromosoma: secuencia completa de genes que representa a un individuo.
- Fenotipo: caracteristicas observables del individuo.
- Alelo: valor posible de un gen.

En este TP no hace falta usar una codificacion binaria. Lo importante es conservar la idea de representacion, posicion y herencia.

## Seleccion

La teoria separa seleccion deterministica y estocastica.

### Elite

- ordena por fitness y prioriza a los mejores
- es muy performante
- puede introducir demasiada presion de seleccion

### Ruleta

- seleccion proporcional al fitness
- los individuos mas aptos tienen mayor probabilidad, pero no garantia
- puede sufrir si hay diferencias de fitness demasiado grandes

### Universal

- parecido a ruleta, pero con punteros equiespaciados
- reduce la varianza del muestreo respecto de ruleta

### Ranking

- usa una pseudo-aptitud basada en el orden relativo, no en el valor absoluto del fitness
- sirve para moderar el efecto de diferencias extremas entre individuos

### Boltzmann

- usa una pseudo-aptitud dependiente de una temperatura que decrece con el tiempo
- al principio favorece exploracion
- despues aumenta explotacion

### Torneo deterministico

- toma `M` individuos al azar y gana el mejor
- es simple y rapido

### Torneo probabilistico

- compara dos individuos
- con una probabilidad `threshold` gana el mejor y, si no, el peor
- permite bajar la presion de seleccion

## Cruza

La cruza recombina genes de dos padres para producir descendencia.

### Un punto

- corta el cromosoma una vez
- preserva bloques contiguos

### Dos puntos

- intercambia un tramo intermedio
- preserva prefijo y sufijo de cada padre

### Anular

- intercambia un segmento circular
- permite cortar atravesando el final del cromosoma

### Uniforme

- decide gen por gen de que padre hereda
- mezcla mas
- rompe la correlacion posicional con mas facilidad

### Probabilidad de recombinacion

Es comun tener una probabilidad `Pc` que determine si se aplica o no la cruza. Si no se aplica, los hijos pueden ser copias de los padres y pasar directo a mutacion.

## Mutacion

La mutacion es una variacion en la informacion genetica del cromosoma. Sirve para:

- enriquecer diversidad genetica
- evitar maximos locales
- mantener exploracion

El detalle fino de tipos de mutacion esta en `mutacion-ag.md`.

## Supervivencia

La supervivencia define como se arma la nueva generacion a partir de padres e hijos.

- aditiva: compiten juntos padres e hijos
- exclusiva: los hijos entran con prioridad

El detalle esta en `supervivencia-ag.md`.

## Convergencia prematura

Una poblacion converge prematuramente cuando pierde diversidad antes de llegar a una solucion aceptable.

La teoria menciona como causas frecuentes:

- presion de seleccion muy alta
- probabilidad de mutacion muy baja
- poblacion demasiado chica

Esto es especialmente importante en problemas visuales, donde un AG puede quedarse atrapado en aproximaciones mediocres pero muy estables.

## Criterios de corte

La teoria enumera varias alternativas:

- tiempo
- cantidad de generaciones
- solucion aceptable
- estructura que no cambia por varias generaciones
- contenido, entendido como mejor fitness estancado

Una combinacion comun es usar maximo de generaciones mas un criterio de estancamiento.

## Esquemas

Las ultimas diapositivas mencionan esquemas, orden y bloques constructivos. No parecen obligatorios para el TP, pero sirven como intuicion:

- bloques de genes que funcionan bien tienden a preservarse
- la cruza puede recombinarlos
- la mutacion puede destruirlos o descubrir nuevos
