# SIA-G02 — Sistemas de Inteligencia Artificial

El presente repositorio reúne los cinco trabajos prácticos realizados en el entorno de la materia **Sistemas de Inteligencia Artificial**, cubriendo desde la búsqueda clásica en espacios de estados hasta modelos generativos profundos. Cada TP es un proyecto independiente (Python) con sus propias dependencias, configuraciones y resultados.

<details>
  <summary>Contenidos</summary>
  <ol>
    <li><a href="#trabajos-prácticos">Trabajos Prácticos</a>
      <ul>
        <li><a href="#tp1--métodos-de-búsqueda">TP1 — Métodos de Búsqueda</a></li>
        <li><a href="#tp2--algoritmos-genéticos">TP2 — Algoritmos Genéticos</a></li>
        <li><a href="#tp3--perceptrón-y-redes-multicapa">TP3 — Perceptrón y Redes Multicapa</a></li>
        <li><a href="#tp4--aprendizaje-no-supervisado">TP4 — Aprendizaje No Supervisado</a></li>
        <li><a href="#tp5--autoencoders-y-vae">TP5 — Autoencoders y VAE</a></li>
      </ul>
    </li>
    <li><a href="#instalación">Instalación</a></li>
    <li><a href="#integrantes">Integrantes</a></li>
  </ol>
</details>

## Trabajos Prácticos

### TP1 — Métodos de Búsqueda

El TP1 implementa un motor completo para resolver el puzzle **Sokoban**, un problema de búsqueda en espacio de estados donde un jugador debe empujar cajas hacia sus destinos sin quedar atrapado, comparando algoritmos de distintas familias y heurísticas.

#### Algoritmos implementados

| Familia | Algoritmo | Óptimo | Completo |
|---------|-----------|:------:|:--------:|
| No informados | **BFS** (cola FIFO) | ✅ | ✅ |
| No informados | **DFS** (pila LIFO) | ❌ | ✅ |
| No informados | **IDDFS** (DFS con límite incremental) | ✅ | ✅ |
| Informados | **Greedy** (minimiza h(n)) | ❌ | ✅ |
| Informados | **A\*** (minimiza f(n) = g(n) + h(n)) | ✅ | ✅ |

Todos usan **graph-search** con conjunto de visitados para evitar re-explorar estados.

#### Heurísticas

Se diseñaron y compararon 5 heurísticas para guiar la búsqueda informada:

- **Manhattan** y **Euclidean** — distancia de cada caja al objetivo más cercano.
- **Dead Square** — poda inmediata de estados con cajas encerradas en esquinas irresolubles.
- **Húngaro** — asignación óptima caja↔meta minimizando la distancia total (via algoritmo húngaro).
- **Weighted Hungarian** — variante agresiva no admisible que puede encontrar soluciones más rápido sacrificando optimalidad.

Las heurísticas admiten composición: se puede combinar varias con `max(h₁, h₂, ...)` para sumar las ventajas de cada una.

#### Tableros y análisis

Se incluyen **21 tableros** `level_*.txt` de diversas dificultades, más tableros especiales para análisis dirigido:

- Tableros de variantes direccionales (`level_12_*.txt`).
- Contraejemplos que evidencian la no-admisibilidad de `weighted_hungarian`.
- Familia `box_count_traps/` para estudiar la escalabilidad de heurísticas al aumentar la cantidad de cajas.
- Showcase de `dead_square` para visualizar el efecto de poda de esquinas.

El benchmarking automático (`run_batch.py`) ejecuta **273 corridas** (21 tableros × 13 configuraciones) midiendo nodos expandidos, nodos en frontera, costo de solución y tiempo. El animador interactivo (`tp1-animate`) permite reproducir, pausar y exportar la solución como GIF o MP4.

<p align="right">(<a href="#sia-g02--sistemas-de-inteligencia-artificial">Volver</a>)</p>

### TP2 — Algoritmos Genéticos

El TP2 implementa un motor de **algoritmos genéticos** que aproxima una imagen objetivo mediante una colección de formas geométricas (triángulos o elipses) translúcidas superpuestas, evolucionando generación a generación hacia la imagen más fiel posible.

#### Modelo del individuo

Cada individuo es una lista ordenada de genes geométricos — su orden define el Z-index de renderizado. Cada gen representa una forma (triángulo con 3 vértices normalizados + color RGBA, o elipse con centro, radios, ángulo y color RGBA). La aptitud maximiza la similitud entre la imagen renderizada y el objetivo.

#### Operadores genéticos

El sistema implementa un catálogo completo y configurable de operadores:

| Componente | Implementaciones |
|-----------|-----------------|
| **Selección** | Elite, Ruleta, Universal, Boltzmann, Torneo, Torneo Probabilístico, Ranking |
| **Cruzas** | Un punto, Dos puntos, Uniforme, Anular, Spatial Z-Index, Aritmética |
| **Mutaciones** | Gen único, Multigen limitada, Multigen uniforme, Completa, Guiada por mapa de error |
| **Supervivencia** | Aditiva (élite + descendencia) y Exclusiva (reemplaza generación) |
| **Fitness** | Linear (1 - NMSE), RMSE, Inverso normalizado, Exponencial, SSIM, Edge Loss, Composite |

#### Características avanzadas

- **Modelo de Islas (IMGA)**: divide la población en islas con migración periódica entre ellas, fomentando diversidad y paralelismo real.
- **Curriculum Learning**: transición automática entre fases de operadores (e.g., cruza global → cruza local a medida que la imagen converge).
- **Mutación guiada**: `error_map_guided` concentra las mutaciones en las regiones con mayor error, acelerando la convergencia en zonas difíciles.
- **Renderizado GPU** (opcional): backend `moderngl` para renderizar individuos en GPU, con fallback automático a CPU.
- **Visualizador interactivo**: reproduce la evolución frame a frame, permite exportar GIF/MP4, y genera paneles de resumen.

<p align="right">(<a href="#sia-g02--sistemas-de-inteligencia-artificial">Volver</a>)</p>

### TP3 — Perceptrón y Redes Multicapa

El TP3 construye toda la matemática de redes neuronales supervisadas desde primitivos NumPy: propagación hacia adelante, retropropagación de gradientes, optimizadores y métricas de clasificación. Se aplica a tres problemas concretos.

#### Arquitectura común (`common/`)

El módulo compartido provee todos los bloques reutilizables:

- **Perceptrón Simple** — con funciones de activación escalón, lineal, sigmoide y ReLU.
- **MLP completo** — con soporte para early stopping, data augmentation on-the-fly y ajuste dinámico de la tasa de aprendizaje.
- **Optimizadores**: SGD, SGD con Momentum, RMSProp y Adam (con decaimiento L2 opcional).
- **Planificadores de LR**: Step Decay, Exponential Decay, Adaptive LR.
- **Inicialización de pesos**: Xavier/Glorot y He para distintas funciones de activación.
- **Ensemble**: combinación de predicciones de múltiples modelos por soft voting.

#### Ejercicios

**Ejercicio XOR** — Validación inicial de la capacidad del MLP para separar dominios no lineales. Demuestra que un perceptrón simple es incapaz de aprender XOR mientras que dos capas con backpropagation lo resuelven trivialmente.

**Ejercicio 1 — Detección de Fraude** — Modela la probabilidad de fraude (`big_model_fraud_probability`) usando variantes del Perceptrón Simple (lineal, sigmoide, ReLU). El flujo incluye análisis exploratorio, K-Fold Cross Validation y selección de umbral óptimo por análisis OOF.

**Ejercicio 2 — Clasificación de Dígitos (base)** — Entrenamiento y comparación exhaustiva de arquitecturas MLP sobre `digits.csv`. Se barren sistemáticamente arquitecturas, learning rates, optimizadores y funciones de pérdida (softmax + cross-entropy vs tanh + MSE), consolidando métricas de múltiples seeds en curvas de ensemble.

**Ejercicio 3 — Generalización Extendida** — Entrena sobre `more_digits.csv` (dataset más difícil) buscando minimizar el error de generalización en `digits_test.csv`. Se aplican técnicas avanzadas: regularización L2, data augmentation afín (rotación + escala), y ensembles heterogéneos con entrenamiento paralelo multiprocesamiento. Se incluyen mecanismos de **Saliency** y **Occlusion** para interpretabilidad visual del modelo.

<p align="right">(<a href="#sia-g02--sistemas-de-inteligencia-artificial">Volver</a>)</p>

### TP4 — Aprendizaje No Supervisado

El TP4 implementa **desde cero** (sin frameworks de deep learning) tres modelos clásicos de aprendizaje no supervisado, estudiando sus propiedades teóricas y empíricas sobre datasets reales y sintéticos.

#### Ejercicio 1.1 — Self-Organizing Map (Kohonen)

Un SOM sobre el dataset `europe.csv` (28 países europeos, 7 variables socioeconómicas) agrupa países con perfiles similares en una grilla 2D. La implementación propia incluye:

- Búsqueda de neurona ganadora (BMU) por distancia euclidiana.
- Funciones de vecindad: **Gaussiana** `h(d) = exp(-d²/2σ²)` y **Burbuja**.
- Decaimiento **exponencial** y **lineal** de radio y learning rate.
- Métricas de calidad: **Error de Cuantización** (QE) y **Error Topológico** (TE).
- Visualizaciones: U-Matrix, mapa de países, hit map y planos de componentes.

Se realizan estudios sistemáticos de hiperparámetros (barrido de radio, learning rate) para analizar el trade-off QE vs TE.

#### Ejercicio 1.2 — PCA de referencia

PCA completo con scikit-learn sobre el mismo dataset europeo: varianza explicada, biplot de los dos primeros componentes, ranking de países y boxplots pre/post estandarización. Sirve como gold standard para validar los resultados de Kohonen y Oja.

#### Ejercicio 2.1 — Regla de Oja

Neurona lineal hebbiana que converge al primer componente principal sin calcular explícitamente la matriz de covarianza. La regla `w(t+1) = w(t) + η(t)·y·(x − y·w(t))` garantiza `‖w‖ → 1` y extrae el primer autovector de la covarianza. Los resultados se comparan con sklearn PCA a través de coseno de similitud, correlación de Pearson y varianza explicada.

#### Ejercicio 2.2 — Red de Hopfield

Memoria asociativa que almacena y recupera patrones de letras 5×5 (A–Z) a partir de versiones ruidosas, usando la regla de Hebb:

```
W_ij = (1/N) · Σ_μ ξ_i^μ · ξ_j^μ,   W_ii = 0
```

Se estudian exhaustivamente:
- **Recuperación síncrona y asíncrona** — la energía `E = −½ sᵀWs` decrece monótonamente en modo asíncrono.
- **Estados espúreos** — puntos fijos distintos a los patrones almacenados y sus complementos.
- **Ortogonalidad**: se barren las C(26,4) = 14.950 combinaciones de letras para encontrar el subconjunto más ortogonal (`GRTV`), que minimiza el crosstalk.
- **Capacidad de la red**: análisis de recall accuracy vs número de patrones almacenados, con el límite teórico ≈ 0.138·N.
- **Escalado adaptativo**: con 26 letras y N=25 neuronas el ratio p/N ≈ 1.04 excede el límite; se escala la grilla a 15×15 (225 neuronas) para almacenar el abecedario completo.

<p align="right">(<a href="#sia-g02--sistemas-de-inteligencia-artificial">Volver</a>)</p>

### TP5 — Autoencoders y VAE

El TP5 lleva la implementación manual al extremo: forward, backprop y gradientes analíticos propios (sin PyTorch, sin TensorFlow), verificados sistemáticamente con **gradient-check numérico** (diferencia relativa < 1e-5).

#### Ejercicio 1 — Autoencoder determinista

Comprime los **32 glifos** de la fuente `font.h` (5×7 = 35 bits binarios cada uno) a un **espacio latente de 2 dimensiones**, y los reconstruye con a lo sumo 1 píxel de error. La arquitectura espejo (`35-25-15-8-2` encoder, `2-8-15-25-35` decoder) se define por config JSON.

Se estudian y comparan:
- **Optimizadores**: Adam (robusto, 17/20 restarts exactos) vs L-BFGS-B vía scipy (más propenso a mínimos locales, 11/20).
- **Inicializaciones**: Xavier/Glorot vs He vs Normal — Xavier más estable, Normal introduce dispersión entre runs.
- **Profundidad de red**: rampas progresivas vs cuellos abruptos (`35-20-2` falla con 3px de error mínimo).
- **Multi-restart**: se reentrenan N corridas con semillas distintas y se conserva la de menor error máximo de píxel.

El espacio latente 2D visualizado (`latent_scatter.png`) muestra cómo el autoencoder organiza los glifos: letras similares quedan juntas. La interpolación lineal en el espacio latente genera nuevos glifos plausibles (`new_letter.png`).

**Denoising Autoencoder (1b)**: se extiende el entrenamiento a `X̃ → X` (entrada corrompida, objetivo limpio). Con corrupción online por época y nivel de ruido mixto `[0, 0.3]`, el modelo alcanza ~99% de glifos perfectos a nivel 0.05 y ~85% a nivel 0.30, **sin ampliar el cuello latente de 2 dimensiones**.

#### Ejercicio 2 — Autoencoder Variacional (VAE)

Extiende el autoencoder a un modelo **generativo** sobre un dataset de **32 emojis** rasterizados (28×28 grises). Las novedades implementadas desde cero:

- **Cabezas µ/logσ²** — el encoder produce una distribución `q(z|x) = N(µ, σ²)` en lugar de un código fijo.
- **Truco de reparametrización** — `z = µ + e^{logσ²/2}·ε`, que hace el muestreo diferenciable.
- **ELBO** — pérdida = reconstrucción + β·KL(N(µ,σ) ‖ N(0,I)), con warmup lineal de β para evitar posterior collapse.
- **Generación** — muestrear `z ~ N(0,I)` y decodificar genera emojis nuevos; el manifold 2D revela la estructura del espacio latente.

El TP profundiza con un estudio empírico completo:

- **Barrido de β** — ablación del balance reconstrucción vs regularización; `β=1` (VAE canónico) es el punto óptimo para este dataset.
- **Barrido de latente** — de 2 a 32 dimensiones: la reconstrucción se aplana (~309 nats independientemente del latente), revelando que el cuello no es la dimensión sino la **capacidad del decoder MLP**.
- **MLP vs CNN** — se implementa una `ConvVAE` (convoluciones desde cero con im2col/col2im) y se compara cara a cara. El MLP gana en las tres dimensiones evaluadas (reconstrucción, generación y estructura del latente) a la escala de este dataset; la CNN solo empata con más datos (~1294 glifos).
- **Posterior agregado y GMM** — cuando el latente es alto, muestrear de `N(0,I)` produce manchones porque el posterior real no cubre el prior. Se implementa un GMM (EM desde cero) sobre los µ del dataset como prior alternativo: genera imágenes reconocibles donde `N(0,I)` fallaba, **desacoplando la calidad de generación de β y del tamaño del latente**.

<p align="right">(<a href="#sia-g02--sistemas-de-inteligencia-artificial">Volver</a>)</p>

## Instalación

Todos los TPs usan **[`uv`](https://docs.astral.sh/uv/)** como gestor de paquetes y entornos Python. `uv` descarga automáticamente la versión de Python requerida y crea el virtualenv aislado.

Clonar el repositorio:

- HTTPS:
  ```sh
  git clone https://github.com/FarDenGhi/SIA-G02.git
  ```
- SSH:
  ```sh
  git clone git@github.com:FarDenGhi/SIA-G02.git
  ```

Instalar las dependencias de un TP específico (ejecutar desde el directorio del TP):

```sh
cd tp1   # o tp2, tp3, tp4, tp5
uv sync
```

Para incluir dependencias de desarrollo (tests, scripts de análisis):

```sh
uv sync --dev
```

> **Requisito único**: [`uv`](https://docs.astral.sh/uv/). La versión de Python (3.12 en la mayoría de los TPs) se descarga automáticamente al hacer `uv sync`.

<p align="right">(<a href="#sia-g02--sistemas-de-inteligencia-artificial">Volver</a>)</p>

## Integrantes

Filipo Ardenghi (64306) - fardenghi@itba.edu.ar

Martín Alejandro Barnatán (64463) - mbarnatan@itba.edu.ar

Ignacio Pedemonte Berthoud (64908) - ipedemonteberthoud@itba.edu.ar

Ezequiel Testoni (64709) - etestoni@itba.edu.ar

<p align="right">(<a href="#sia-g02--sistemas-de-inteligencia-artificial">Volver</a>)</p>
