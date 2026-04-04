# Compresor de Imágenes Evolutivo

> Aproximación visual de imágenes mediante algoritmos genéticos y triángulos traslúcidos.

Sistema que utiliza un motor de **Algoritmos Genéticos (AG)** para reconstruir una imagen objetivo usando únicamente una cantidad fija de triángulos semitransparentes superpuestos sobre un lienzo blanco. El resultado es una representación artística y comprimida de la imagen original.

---

## Funcionamiento: El Enfoque Genético

El sistema evoluciona una población de soluciones candidatas (individuos), donde cada individuo representa una posible aproximación a la imagen objetivo. A través de generaciones sucesivas, los individuos mejor adaptados son seleccionados, combinados y mutados hasta converger hacia una solución óptima.

### Flujo Evolutivo

1. **Inicialización**: Se genera una población aleatoria de individuos.
2. **Evaluación**: Se calcula el fitness de cada individuo comparándolo con la imagen objetivo.
3. **Selección**: Se eligen los individuos más aptos (ej. selección por torneo).
4. **Cruza (Crossover)**: Se combinan pares de individuos para producir descendencia.
5. **Mutación**: Se aplican alteraciones aleatorias sutiles a los nuevos individuos.
6. **Reemplazo**: La nueva generación reemplaza a la anterior.
7. **Criterio de parada**: Se repite el ciclo hasta alcanzar el número máximo de generaciones o un umbral de error aceptable.

---

## Arquitectura del Algoritmo

### Genotipo (Cromosoma)

Cada individuo está representado por una **lista ordenada de N triángulos**. El orden determina el Z-index (profundidad de renderizado).

Cada triángulo se define por:

| Atributo       | Descripción                                      |
|----------------|--------------------------------------------------|
| Vértice 1      | Coordenadas (x₁, y₁)                             |
| Vértice 2      | Coordenadas (x₂, y₂)                             |
| Vértice 3      | Coordenadas (x₃, y₃)                             |
| Color          | Valor RGBA o HSLA con canal alfa (transparencia) |

### Función de Fitness

El fitness se calcula mediante el **Error Cuadrático Medio (MSE)** entre la imagen renderizada y la imagen objetivo:

```
MSE = (1 / n) × Σ (pixel_original - pixel_renderizado)²
```

Donde `n` es el número total de píxeles. **El objetivo es minimizar este valor.**

### Operadores Genéticos

| Operador       | Descripción                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Selección**  | Torneo, ruleta, ranking u otros métodos configurables.                      |
| **Cruza**      | Intercambio de secuencias de triángulos entre dos padres.                   |
| **Mutación**   | Alteración de coordenadas, colores, transparencia u orden (Z-index).        |

---

## Inputs (Entradas del Sistema)

| Parámetro                | Descripción                                                        |
|--------------------------|--------------------------------------------------------------------|
| Imagen objetivo          | Archivo de imagen a aproximar (PNG, JPG, etc.).                    |
| N (triángulos)           | Cantidad estricta de triángulos permitidos en la solución.         |
| Tamaño de población      | Número de individuos en cada generación.                           |
| Tasa de mutación         | Probabilidad de mutación por gen/triángulo.                        |
| Método de cruza          | Estrategia de combinación de individuos (configurable).            |
| Criterio de parada       | Generaciones máximas y/o umbral de error mínimo.                   |

---

## Outputs (Salidas del Sistema)

| Salida                   | Descripción                                                        |
|--------------------------|--------------------------------------------------------------------|
| Imagen renderizada       | Archivo de imagen con la mejor aproximación encontrada.            |
| Datos de triángulos      | Exportación estructurada (JSON/CSV) con posiciones, colores y orden de cada triángulo. |
| Métricas de evolución    | Gráfico o log con la evolución del fitness por generación.         |
| Estadísticas finales     | Error final (MSE), tiempo de ejecución y número de generaciones.   |

---

## Instalación

### Requisitos previos

- Python 3.10+
- pip

### Pasos de instalación

```bash
# Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd tp2

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

```bash
# Ejecutar el compresor evolutivo
python main.py --image <ruta_imagen> --triangles <N> [opciones]

# Ejemplo
python main.py --image input/foto.png --triangles 100 --generations 5000 --population 50
```

### Opciones disponibles

| Opción               | Descripción                              | Valor por defecto |
|----------------------|------------------------------------------|-------------------|
| `--image`            | Ruta a la imagen objetivo                | (requerido)       |
| `--triangles`        | Cantidad de triángulos                   | (requerido)       |
| `--population`       | Tamaño de la población                   | 100               |
| `--generations`      | Número máximo de generaciones            | 10000             |
| `--mutation-rate`    | Tasa de mutación                         | 0.01              |
| `--output`           | Directorio de salida                     | `output/`         |

---

## Estructura del Proyecto

```
tp2/
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── README.md
├── src/
│   ├── genetic/            # Motor de algoritmos genéticos
│   │   ├── individual.py   # Representación del genotipo
│   │   ├── population.py   # Gestión de la población
│   │   ├── selection.py    # Métodos de selección
│   │   ├── crossover.py    # Operadores de cruza
│   │   └── mutation.py     # Operadores de mutación
│   ├── fitness/            # Evaluación de fitness
│   │   └── mse.py          # Cálculo del MSE
│   ├── rendering/          # Renderizado de triángulos
│   │   └── canvas.py       # Generación de imágenes
│   └── utils/              # Utilidades generales
│       ├── config.py       # Configuración e hiperparámetros
│       └── export.py       # Exportación de resultados
├── input/                  # Imágenes de entrada
└── output/                 # Resultados generados
```

---

## Licencia

Este proyecto fue desarrollado como parte del Trabajo Práctico N°2 de Sistemas de Inteligencia Artificial.
