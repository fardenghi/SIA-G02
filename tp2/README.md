# Compresor de Imágenes Evolutivo

> Aproximación visual de imágenes mediante algoritmos genéticos y triángulos traslúcidos.

Sistema que utiliza un motor de **Algoritmos Genéticos (AG)** para reconstruir una imagen objetivo usando únicamente una cantidad fija de triángulos semitransparentes superpuestos sobre un lienzo blanco. El resultado es una representación artística y comprimida de la imagen original.

---

## Tabla de Contenidos

- [Funcionamiento del Algoritmo](#funcionamiento-el-enfoque-genético)
- [Arquitectura](#arquitectura-del-algoritmo)
- [Instalación](#instalación)
- [Inicio Rápido](#inicio-rápido)
- [Configuración de Parámetros](#configuración-de-parámetros)
- [Ejemplos de Uso](#ejemplos-de-uso)
- [Salidas del Sistema](#salidas-del-sistema)
- [Visualizador Interactivo](#visualizador-interactivo)
- [Testing](#testing)
- [Estructura del Proyecto](#estructura-del-proyecto)

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
7. **Criterio de parada**: Se repite el ciclo hasta alcanzar el número máximo de generaciones o un umbral de fitness objetivo.

---

## Arquitectura del Algoritmo

### Genotipo (Cromosoma)

Cada individuo está representado por una **lista ordenada de N triángulos**. El orden determina el Z-index (profundidad de renderizado).

Cada triángulo se define por:

| Atributo       | Descripción                                      |
|----------------|--------------------------------------------------|
| Vértice 1      | Coordenadas (x₁, y₁) normalizadas [0, 1]         |
| Vértice 2      | Coordenadas (x₂, y₂) normalizadas [0, 1]         |
| Vértice 3      | Coordenadas (x₃, y₃) normalizadas [0, 1]         |
| Color          | Valor RGBA (R, G, B en [0-255], A en [0-1])      |

### Función de Fitness

El sistema usa el **Error Cuadrático Medio (MSE)** como señal de error base y luego lo transforma a fitness:

```
MSE = (1 / n) × Σ (pixel_original - pixel_renderizado)²
Fitness = 1 / (1 + MSE)
```

Donde `n` es el número total de píxeles. Con esta formulación, **mayor fitness = mejor individuo**. Existen múltiples métodos de evaluación (como `linear`, `rmse`, `exponential`, `detail_weighted`) y métricas perceptuales avanzadas (`composite` combinando variables morfológicas o `ssim` evaluando la similitud estructural pura).

### Operadores Genéticos

| Operador | Métodos Disponibles | Descripción |
|----------|---------------------|-------------|
| **Selección** | `elite`, `tournament`, `probabilistic_tournament`, `roulette`, `universal`, `boltzmann`, `rank` | Elige individuos para reproducción |
| **Cruza** | `single_point`, `two_point`, `uniform`, `annular` | Combina genes de dos padres |
| **Mutación** | `single_gene`, `limited_multigen`, `uniform_multigen`, `complete`, `error_map_guided` | Altera atributos de triángulos. Incluye mutación guiada por mapa de error y control de deltas adaptativo. |

---

## Optimizaciones y Rendimiento

El algoritmo implementa diferentes técnicas para mejorar la convergencia visual, la eficiencia y reducir los tiempos de ejecución:

1. **Mutación Guiada por Mapa de Error (`error_map_guided`)**: En vez de alterar parámetros de forma estocástica pura, se favorece la mutación intencionada de aquellos triángulos ubicados en las zonas funcionales específicas que más difieren de la imagen objetivo original.
2. **Mutación con Sigma Adaptativo (`adaptive_sigma_enabled`)**: Las magnitudes de mutación (ej. variaciones de color y saltos en la posición) se ajustan dinámicamente. Crecen si se encadenan mejoras constantes en el fitness de la población (exploración a grandes escalas), y decrecen si se percibe un estancamiento (facilita un refinamiento local y detallista o *fine-tuning*).
3. **Fitness Ponderado por Detalle (`detail_weighted`)**: Para aquellas metas compuestas mayormente de fondos homogéneos con elementos intrincados chicos, los métodos tradicionales desperdician triángulos alisando el fondo. Como alternativa, este método da menor peso al MSE del fondo y escala los castigos en áreas puntillosas con bordes, promoviendo el trazo de figuras más ricas en detalle.
4. **Parada Temprana (`fitness_threshold`)**: Permite la interrupción temprana del proceso evolutivo, si este verifica que se ha alcanzado el umbral de calidad propuesto por el usuario antes de completar `max_generations`, ahorrando una importante carga computacional.
5. **Aceleración Renderizada en GPU (` backend: gpu`)**: Evaluando constantemente candidatos el procesamiento del superpuesto traslúcido resulta muy punitivo para el núcleo de CPU base (donde ejecuta en defecto `Pillow`). Seleccionando el *backend* de renderizado por GPU acelerado invoca subrutinas de la tarjeta gráfica a través de OpenGL (`moderngl`), proveyendo mejor soporte frente a crecimientos exponenciales de `population_size` o `num_triangles`. Para activarlo requiera su dependencia extra en la etapa de instalación.
6. **Curriculum Learning (Transición Dinámica de Fitness)**: Permite ejecutar tu evaluación bajo dos regímenes. Si se lo configura de modo no-Nulo, el algoritmo inicia la optimización con la función rápida normal en búsqueda de siluetas clave y, cuando detecta un evento de agotamiento (`stagnation_threshold`), gatilla un reinicio algorítmico traspasando la brújula evolutiva hacia una función secundaria detallista (e.j. refinamiento de texturas mediante `transition_methods`). Se acompaña de un reinicio en caliente (Hot Restart) interno que reescala históricos de élites al vuelo eliminando saltos matemáticos bruscos.

---

## Instalación

### Requisitos previos

- Python 3.10+
- uv

### Pasos de instalación

```bash
# Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd tp2

# Sincronizar dependencias (runtime + desarrollo)
uv sync --dev

# (Opcional) Instalar soporte acelerado de procesamiento de texturas en GPU
uv sync --dev --extra gpu
```

### Verificar instalación

```bash
# Ejecutar tests para verificar que todo funciona
uv run --dev pytest tests/ -v
```

---

## Inicio Rápido

### Ejecución básica

```bash
# Ejecutar con parámetros mínimos
uv run main.py --image input/mi_imagen.png --triangles 50
```

### Ejemplo completo

```bash
uv run main.py \
    --image input/foto.png \
    --triangles 100 \
    --population 50 \
    --generations 5000 \
    --output output/mi_experimento
```

El sistema mostrará el progreso en consola:

```
Cargando imagen: input/foto.png
Tamaño original: (512, 512)
Tamaño de trabajo: (256, 256)
Triángulos: 100
Población: 50
Generaciones máximas: 5000

Iniciando evolución...
--------------------------------------------------
Gen     0 | Best:   0.000061 | Avg:   0.000048
  >> Mejora en gen 1:   0.000079
  >> Mejora en gen 5:   0.000101
Gen    10 | Best:   0.000114 | Avg:   0.000098
...
```

---

## Configuración de Parámetros

Los parámetros se pueden configurar de **dos formas**:

1. **Argumentos de línea de comandos (CLI)** - tienen prioridad
2. **Archivo de configuración YAML** - valores por defecto

### Opción 1: Argumentos de Línea de Comandos

```bash
uv run main.py --image <ruta> [opciones]
```

#### Parámetros principales

| Opción | Alias | Descripción | Default |
|--------|-------|-------------|---------|
| `--image` | `-i` | **Ruta a la imagen objetivo** (requerido) | - |
| `--triangles` | `-t` | Cantidad de triángulos por individuo | 50 |
| `--population` | `-p` | Tamaño de la población | 100 |
| `--generations` | `-g` | Número máximo de generaciones | 5000 |

#### Parámetros de operadores genéticos

| Opción | Descripción | Valores | Default |
|--------|-------------|---------|---------|
| `--selection` | Método de selección | `elite`, `tournament`, `probabilistic_tournament`, `roulette`, `universal`, `boltzmann`, `rank`, `ranking` | `tournament` |
| `--crossover` | Método de cruza | `single_point`, `two_point`, `uniform`, `annular` | `single_point` |
| `--mutation-rate` | Probabilidad de mutación | 0.0 - 1.0 | 0.3 |

#### Parámetros de salida

| Opción | Alias | Descripción | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Directorio de salida | `output/` |
| `--save-interval` | - | Guardar imagen cada N generaciones | 100 |
| `--max-size` | - | Redimensionar imagen si excede este tamaño | 256 |
| `--quiet` | `-q` | Modo silencioso (menos output) | False |

#### Parámetros de configuración

| Opción | Alias | Descripción | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Archivo de configuración YAML | `config.yaml` |

### Opción 2: Archivo de Configuración YAML

Edita el archivo `config.yaml` para establecer valores por defecto:

```yaml
# Imagen
image:
  target_path: null

# Parámetros del genotipo
genotype:
  num_triangles: 50
  alpha_min: 0.1
  alpha_max: 0.8

# Algoritmo genético
genetic:
  population_size: 100
  max_generations: 5000
  fitness_threshold: null  # null = sin parada temprana
  transition_methods:      # O null para desactivar Curriculum Learning
    - "ssim"
    - "detail_weighted"
  stagnation_threshold: 0.0005
  max_patience: 20

# Selección
selection:
  method: "tournament"    # elite, tournament, probabilistic_tournament, roulette, universal, boltzmann, rank (alias ranking)
  tournament_size: 3       # solo para tournament
  threshold: 0.75          # solo para probabilistic_tournament
  boltzmann_t0: 100.0      # solo para boltzmann
  boltzmann_tc: 1.0        # solo para boltzmann
  boltzmann_k: 0.005       # solo para boltzmann

# Cruza (Crossover)
crossover:
  method: "single_point"  # single_point, two_point, uniform
  probability: 0.8        # probabilidad de cruza

# Mutación
mutation:
  probability: 0.3        # probabilidad de mutar individuo
  gene_probability: 0.1   # probabilidad de mutar cada triángulo
  position_delta: 0.1     # magnitud de perturbación en posición
  color_delta: 30         # magnitud de perturbación en color (0-255)
  alpha_delta: 0.1        # magnitud de perturbación en alfa

# Salida y visualización
output:
  directory: "output"
  save_interval: 100      # 0 = solo guardar al final
  log_interval: 10        # mostrar progreso cada N generaciones
  export_triangles: true  # exportar JSON con triángulos
  plot_fitness: true      # generar gráfico de evolución

# Backend de renderizado
rendering:
  backend: "cpu"          # "cpu" (Pillow, fallback) o "gpu" (moderngl)
```

Los argumentos CLI **sobreescriben** los valores del archivo YAML.

---

## Ejemplos de Uso

### Ejemplo 1: Ejecución rápida para pruebas

```bash
uv run main.py -i input/logo.png -t 30 -g 500 -p 30
```

### Ejemplo 2: Alta calidad (más triángulos y generaciones)

```bash
uv run main.py \
    --image input/retrato.jpg \
    --triangles 200 \
    --generations 10000 \
    --population 100 \
    --save-interval 500 \
    --output output/retrato_hq
```

### Ejemplo 3: Experimentar con operadores

```bash
# Usar selección por ruleta y cruza uniforme
uv run main.py \
    -i input/paisaje.png \
    -t 100 \
    --selection roulette \
    --crossover uniform \
    --mutation-rate 0.5
```

### Ejemplo 4: Modo silencioso (para scripts)

```bash
uv run main.py -i input/imagen.png -t 50 -g 1000 --quiet
```

### Ejemplo 5: Usar configuración personalizada

```bash
# Primero edita mi_config.yaml con tus parámetros
uv run main.py -i input/imagen.png --config mi_config.yaml
```

---

## Salidas del Sistema

Después de ejecutar el algoritmo, se generan los siguientes archivos en el directorio de salida:

```
output/
├── result.png              # Imagen final renderizada
├── triangles.json          # Datos de triángulos (para reconstrucción)
├── fitness_evolution.png   # Gráfico de evolución del fitness
├── gen_00100.png          # Imágenes intermedias (cada save_interval)
├── gen_00200.png
└── ...
```

### Descripción de archivos

| Archivo | Descripción |
|---------|-------------|
| `result.png` | Imagen final con la mejor aproximación encontrada |
| `triangles.json` | Datos estructurados de cada triángulo (posición, color, orden) |
| `fitness_evolution.png` | Gráfico mostrando la evolución del fitness por generación |
| `gen_XXXXX.png` | Capturas intermedias del progreso |

### Reconstruir imagen desde JSON

Puedes reconstruir la imagen a cualquier resolución usando los triángulos exportados:

```bash
# Reconstruir a tamaño original
uv run reconstruct.py output/triangles.json -o reconstruida.png -W 256 -H 256

# Reconstruir a mayor resolución (escala 2x)
uv run reconstruct.py output/triangles.json -o reconstruida_hd.png -W 256 -H 256 --scale 2
```

---

## Visualizador Interactivo

El sistema incluye un visualizador gráfico interactivo para explorar la evolución del algoritmo paso a paso.

### Uso básico

```bash
# Visualizar resultados de un experimento
uv run visualize.py output/mi_experimento
```

### Interfaz gráfica

El visualizador muestra:

- **Panel central**: Imagen del mejor individuo de la generación actual
- **Panel lateral**: Métricas (generación, fitness, número de triángulos)
- **Gráfico inferior**: Evolución del fitness a lo largo de las generaciones
- **Controles**: Slider y botones para navegar entre generaciones

### Controles de teclado

| Tecla | Acción |
|-------|--------|
| `←` / `→` | Generación anterior / siguiente |
| `Espacio` | Play / Pause animación automática |
| `Home` | Ir a la primera generación |
| `End` | Ir a la última generación |

### Opciones de exportación

```bash
# Exportar como GIF animado
uv run visualize.py output/mi_experimento --export-gif evolucion.gif

# Exportar como video MP4 (requiere ffmpeg)
uv run visualize.py output/mi_experimento --export-video evolucion.mp4

# Generar imagen resumen con todos los pasos
uv run visualize.py output/mi_experimento --summary resumen.png
```

### Parámetros del visualizador

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--export-gif` | Exportar animación como GIF | - |
| `--export-video` | Exportar animación como MP4 | - |
| `--summary` | Generar imagen resumen | - |
| `--fps` | Frames por segundo (GIF/video) | 10 |
| `--interval` | Intervalo de animación (ms) | 500 |

---

## Testing

```bash
# Ejecutar todos los tests
uv run --dev pytest tests/ -v

# Ejecutar tests de un módulo específico
uv run --dev pytest tests/test_engine.py -v

# Ejecutar tests con cobertura
uv run --dev pytest tests/ --cov=src --cov-report=html
```

---

## Estructura del Proyecto

```
tp2/
├── main.py                 # Punto de entrada principal (CLI)
├── reconstruct.py          # Script de reconstrucción desde JSON
├── visualize.py            # Visualizador gráfico interactivo
├── config.yaml             # Configuración por defecto
├── pyproject.toml          # Dependencias y metadata del proyecto
├── uv.lock                 # Lockfile generado por uv
├── pytest.ini              # Configuración de pytest
├── README.md
├── src/
│   ├── genetic/            # Motor de algoritmos genéticos
│   │   ├── individual.py   # Triangle + Individual (genotipo)
│   │   ├── population.py   # Gestión de la población
│   │   ├── engine.py       # Motor evolutivo principal
│   │   ├── selection.py    # Métodos de selección
│   │   ├── crossover.py    # Operadores de cruza
│   │   └── mutation.py     # Operadores de mutación
│   ├── fitness/            # Evaluación de fitness
│   │   └── mse.py          # Cálculo de fitness a partir de MSE
│   ├── rendering/          # Renderizado de triángulos
│   │   └── canvas.py       # Generación de imágenes con Pillow
│   └── utils/              # Utilidades generales
│       ├── config.py       # Configuración YAML + CLI
│       └── export.py       # Exportación de resultados
├── tests/                  # Tests unitarios (pytest)
├── input/                  # Imágenes de entrada
└── output/                 # Resultados generados
```

---

## Licencia

Este proyecto fue desarrollado como parte del Trabajo Práctico N°2 de Sistemas de Inteligencia Artificial.
