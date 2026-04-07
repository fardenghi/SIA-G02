# Mutacion en AG

Resumen teorico detallado de los tipos de mutacion relevantes para el TP.

Ver tambien:

- `marco-teorico-ag.md`
- `guia-diseno-y-defensa.md`

## Que es la mutacion
La mutacion se define como una **variacion en la informacion genetica** almacenada en el cromosoma.

Su rol dentro de un Algoritmo Genético es:
- **enriquecer la diversidad genética**,
- **evitar máximos locales**,
- **mantener un grado de exploración**.

La teoría también remarca que, para que la mutación sea más eficiente, es importante una **buena arquitectura y separación de genes**.

---

## Variantes de mutación
Dada una probabilidad de mutación **Pm**, la teoría distingue los siguientes tipos:

### 1. Mutación de gen
Se altera **un solo gen** con probabilidad **Pm**.

**Idea:** en cada aplicación del operador, si ocurre la mutación, se elige un único gen del individuo y se modifica.

**Cuándo sirve:**
- cuando se quiere hacer cambios chicos,
- cuando conviene no romper demasiado la estructura del individuo,
- cuando el problema es sensible a cambios grandes.

---

### 2. Mutación multigen limitada
Se selecciona una cantidad aleatoria de genes en **[1, M]** para mutar, con probabilidad **Pm**.

**Idea:** si ocurre la mutación, no cambia un solo gen sino varios, pero con un límite máximo **M**.

**Cuándo sirve:**
- cuando una mutación de un solo gen resulta demasiado conservadora,
- cuando se quiere explorar más sin llegar a modificar todo el individuo,
- cuando tiene sentido controlar cuánto “ruido” se introduce.

---

### 3. Mutación multigen uniforme
Cada gen tiene una probabilidad **Pm** de ser mutado.

**Idea:** se recorre el cromosoma gen por gen, y cada uno decide de forma independiente si muta o no.

**Cuándo sirve:**
- cuando no se quiere fijar de antemano cuántos genes cambiar,
- cuando se busca una mutación distribuida a lo largo de todo el cromosoma,
- cuando el cromosoma tiene muchos genes relativamente independientes.

---

### 4. Mutación completa
Con probabilidad **Pm** se mutan **todos los genes del individuo**, de acuerdo con la función de mutación definida para cada gen.

**Idea:** si se activa este operador, el individuo entero se ve afectado.

**Cuándo sirve:**
- para introducir un cambio muy fuerte,
- para escapar de poblaciones demasiado estancadas,
- como operador más agresivo de exploración.

**Riesgo:** puede romper estructuras buenas que ya se habían encontrado.

---

## Qué significa “mutar un gen”
La teoría aclara que no alcanza con decidir **cuántos genes** mutan: también hay que definir **qué significa mutar un alelo** dentro de cada gen.

Algunas posibilidades que menciona:

### 1. Tomar un nuevo alelo aleatorio
Se reemplaza el valor actual del gen por otro valor válido elegido al azar.

**Ejemplo:**
- si un gen representa un color, se elige otro color válido;
- si un gen representa una posición, se elige otra posición válida.

### 2. Aplicar un delta al alelo
En lugar de reemplazar el gen por completo, se modifica el valor actual aplicándole un **delta** en algún sentido y con alguna distribución.

**Ejemplo:**
- mover una coordenada unos píxeles,
- subir o bajar levemente un canal de color,
- cambiar un ángulo un poco hacia la izquierda o la derecha.

Esto suele ser útil cuando los genes representan valores continuos o numéricos.

---

## Relación con convergencia prematura
La teoría conecta la mutación con el problema de la **convergencia prematura**.

Una población sufre convergencia prematura cuando:
- pierde diversidad,
- deja de generar variantes nuevas relevantes,
- y queda atrapada en una solución subóptima.

Entre las causas mencionadas están:
- una **presión de selección muy alta**,
- una **probabilidad de mutación muy baja**,
- un **tamaño de población muy pequeño**.

Por eso, la mutación no solo modifica individuos: también ayuda a conservar diversidad y a evitar que todo el AG se “planché” demasiado rápido.

---

## Resumen rápido
- **Mutación de gen:** cambia un solo gen.
- **Multigen limitada:** cambia varios genes, con un máximo **M**.
- **Multigen uniforme:** cada gen muta independientemente con probabilidad **Pm**.
- **Mutación completa:** cambia todos los genes del individuo.
- Además, para cada gen hay que definir **cómo** muta: por **reemplazo aleatorio** o por **delta**.

---

## Qué conviene destacar en el TP
Si lo querés bajar al TP, conviene justificar:
- **qué operador de mutación implementás**,
- **sobre qué genes actúa**,
- **cómo muta cada gen concretamente**,
- **qué nivel de exploración introduce**,
- y **por qué ese operador tiene sentido para este problema**.
