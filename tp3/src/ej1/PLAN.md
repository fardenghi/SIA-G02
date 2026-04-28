# Plan de Desarrollo - TP3 Ejercicio 1: Fraud Detection con Perceptrón

## Objetivo General
Implementar TinyModel (perceptrón simple) que replique el comportamiento de BigModel para predicción de probabilidades de fraude en transacciones online.

## Metodología
Iteración rápida: comparar perceptrón lineal vs no lineal, seleccionar mejor, realizar estudio de generalización.

---

## 🎯 DECISIONES CLAVE (a resolver antes de implementar)

Estas decisiones se toman AHORA y guiarán todo el desarrollo:

### D1: Manejo de Valores Faltantes (Fase 1)
**Pregunta:** ¿Si hay valores faltantes en el dataset, qué hacemos?
- **Opción A:** Eliminar filas con valores faltantes ✅
- **Opción B:** Imputar con media/mediana
- **Opción C:** Verificar primero si hay, si no hay, ignorar

**Decision:** ✅ **Opción A** - Eliminar filas con faltantes

---

### D2: Learning Rate Inicial (Fases 3 y 4)
**Pregunta:** ¿Cuál es el learning rate inicial para ambos modelos?
- **Opción A:** 0.001 (conservador, converge lento pero estable)
- **Opción B:** 0.01 (balance) ✅
- **Opción C:** 0.1 (agresivo, puede oscilar)

**Decision:** ✅ **Opción B - 0.01**

---

### D3: Número de Epochs (Fases 3 y 4)
**Pregunta:** ¿Cuántos epochs máximo entrenar?
- **Opción A:** 100-200 epochs ✅
- **Opción B:** 500 epochs
- **Opción C:** 1000 epochs (o until convergence)

**Decision:** ✅ **100-200 epochs** (comenzar con esto, ajustar según convergencia observada)

---

### D4: Función de Activación No Lineal (Fase 4)
**Pregunta:** ¿Cuál activación usamos?
- **Opción A:** tanh (recomendado por enunciado, rango [-1, 1])
- **Opción B:** sigmoid (rango [0, 1], más natural para probabilidades) ✅
- **Opción C:** ReLU (no típico para regresión, pero probar)

**Decision:** ✅ **Opción B - sigmoid**
**Justificación:** El target está en [0,1], sigmoid es el match perfecto. tanh [-1,1] sería asimétrico.

---

### D5: Proporción Train/Val (Fase 6)
**Pregunta:** ¿Cuál split de datos?
- **Opción A:** 70-30 (70% train, 30% val)
- **Opción B:** 80-20 (80% train, 20% val) ✅
- **Opción C:** Otro ratio

**Decision:** ✅ **Opción B - 80-20**

---

### D6: Estrategia de Split (Fase 6)
**Pregunta:** ¿Cómo hacemos el split?
- **Opción A:** Random shuffle simple ✅
- **Opción B:** Estratificado (mantener distribución del target)

**Decision:** ✅ **Opción A - Random split**
**Justificación:** Target es continuo (no categórico), random es simple y suficiente.

---

### D7: Métrica Principal para Umbral (Fase 7)
**Pregunta:** ¿Cuál métrica usamos para seleccionar el umbral?
- **Opción A:** Maximizar F1-score ✅
- **Opción B:** Maximizar Precision (evitar falsos positivos)
- **Opción C:** Maximizar Recall (detectar fraudes)
- **Opción D:** Balancear según cliente (a definir luego)

**Decision:** ✅ **Opción A - F1-score como métrica principal**
**Con:** Reportar también curva completa precision-recall para que cliente ajuste según su risk tolerance.
**Justificación:** En fraude, necesitamos balance. F1 lo da. Pero cliente puede ver trade-offs completos.

---

## Notas:
- Llenar las decisiones ANTES de empezar a codificar
- Las decisiones se reflejarán en `src/ej1/config.py`
- Cambios posteriores requieren actualizar config.py y comentar en git

---

## FASE 1: Exploración y Preparación de Datos

### Objetivo de la fase
Entender qué datos tenemos, en qué estado están, y qué desafíos presenta el dataset antes de tocar algoritmos.

### Tareas

1. **Cargar y inspeccionar el dataset**
   - Leer `transactions.csv`
   - Dimensiones (filas, columnas)
   - Tipos de datos
   - Valores faltantes
   - Primeras/últimas filas

2. **Análisis descriptivo de features**
   - Estadísticas básicas (min, max, mean, std) de cada feature
   - Identificar ranges y outliers
   - Detectar features que podrían ser problemáticas

3. **Análisis del target (big_model_fraud_probability)**
   - Distribución: ¿uniforme?, ¿sesgada?
   - Estadísticas (media, std, percentiles)
   - Gráficas: histograma, boxplot

4. **Análisis de correlaciones**
   - Matriz de correlación entre features
   - Identificar multicolinealidad
   - Features más correlacionados con el target

5. **Decisiones de preprocesamiento**
   - ¿Normalizar/standarizar?
   - ¿Hay features que eliminar?
   - ¿Cómo manejar outliers si los hay?

### Output esperado
- Notebook/script de exploración
- Reporte con hallazgos clave
- Dataset limpio y listo para usar

### Decisiones pendientes
- **¿Normalización?** ¿Standarización (z-score)? ¿Escalado min-max?
- **¿Qué features usan?** ¿Todos, o descartan algunos?

---

## FASE 2: Arquitectura Base del Código

### Objetivo de la fase
Preparar la estructura de código reutilizando las clases base ya implementadas (`SimplePerceptron`, datasets, etc).

### Contexto
Ya existe en `/src`:
- `SimplePerceptron` (funciona con cualquier activación)
- Funciones de activación (`step`, `linear`, etc)
- Funciones de loss
- Utilidades generales

Todo el código específico del Ej1 irá en `/src/ej1`.

### Estructura de `/src/ej1`

```
src/ej1/
├── data.py              # Cargar y normalizar transactions.csv
├── training.py          # Loop de training, reusing SimplePerceptron
├── evaluation.py        # Métricas (MSE, MAE, gráficas)
└── config.py            # Configuración de experimentos (hiperparámetros)
```

### Tareas

1. **`data.py`**
   - Función `load_fraud_dataset()` → carga transactions.csv
   - Función `normalize_features()` → z-score normalización
   - Función `train_val_split()` → divide en train/validation

2. **`training.py`**
   - Clase `FraudTrainer` que:
     - Recibe un `SimplePerceptron` ya creado
     - Entrena con SGD simple (usando `SimplePerceptron.train()`)
     - Guarda historial de loss
   - Funciones para guardar/cargar modelos (JSON config + npz pesos)

3. **`evaluation.py`**
   - Función `evaluate()` → calcula MSE, MAE en un dataset
   - Función `plot_loss_curves()` → gráfica loss training vs validation
   - Función `plot_predictions_distribution()` → histograma de predicciones

4. **`config.py`**
   - Diccionario con hiperparámetros por defecto:
     - `learning_rate`, `max_epochs`, `activation`, etc.
   - Funciones para cargar/guardar configs desde JSON

### Output esperado
- Directorio `/src/ej1` creado
- Módulos base implementados (aunque parciales)
- Sistema de configuración listo

### Decisiones tomadas
- **Optimizador:** SGD simple (SGD con momentum como opcional)
- **Loss function:** MSE
- **Datos:** Full dataset (no batches)
- **Persistencia:** JSON config + numpy npz para pesos/history

---

## FASE 3: Perceptrón Lineal

### Objetivo de la fase
Entrenar el modelo más simple: perceptrón con activación lineal (sin activación no lineal).

### Contexto
Usamos `SimplePerceptron` de `/src/perceptron.py` con `activation=linear`.

### Tareas

1. **Preparar datos**
   - Cargar dataset (usar `data.py`)
   - Normalizar features (z-score)
   - NO hacer split aún (usamos TODOS los datos para este estudio)

2. **Crear y entrenar modelo lineal**
   - `perceptron_linear = SimplePerceptron(input_size=10, activation=linear, ...)`
   - Configuración: `learning_rate=0.01`, `max_epochs=100` (valores iniciales, ajustar según necesidad)
   - Entrenar con `perceptron_linear.train(X, y)`

3. **Análisis de aprendizaje**
   - Graficar loss vs epochs
   - Observar: ¿baja el loss?, ¿converge?, ¿hay plateau?

4. **Guardar resultados**
   - Guardar modelo (config + pesos)
   - Guardar loss history para comparación posterior

### Output esperado
- Modelo lineal entrenado
- Gráfica de loss vs epochs
- Archivo de configuración y pesos guardados

### Decisiones pendientes
- **Learning rate inicial:** ¿0.01? ¿0.001? (probar y ajustar)
- **Epochs:** ¿100? ¿200? (hasta que converja)

---

## FASE 4: Perceptrón No Lineal

### Objetivo de la fase
Entrenar perceptrón con activación no lineal para agregar capacidad de aprendizaje.

### Contexto
Usamos `SimplePerceptron` de `/src/perceptron.py` con `activation=tanh` (o la que decidas).

### Tareas

1. **Seleccionar función de activación**
   - Candidatos: `tanh`, `sigmoid`, `ReLU`
   - Sugerencia: **`tanh`** (está en `/src/activation.py` y es estándar para regresión)

2. **Preparar datos**
   - Usar los MISMOS datos normalizados de Fase 3
   - Mismo: NO hacer split (TODOS los datos)

3. **Crear y entrenar modelo no lineal**
   - `perceptron_nonlinear = SimplePerceptron(input_size=10, activation=tanh, ...)`
   - Configuración: mismo learning_rate y epochs que lineal (para comparación justa)
   - Entrenar con `perceptron_nonlinear.train(X, y)`

4. **Análisis comparativo**
   - Graficar loss lineal vs no lineal en MISMO gráfico
   - Observar: ¿mejora el no lineal?, ¿cuánto?, ¿hay saturación?

5. **Guardar resultados**
   - Guardar modelo (config + pesos)
   - Guardar loss history para comparación

### Output esperado
- Modelo no lineal entrenado
- Gráficas comparativas (lineal vs no lineal)
- Observaciones iniciales sobre comportamiento

### Decisiones pendientes
- **Función de activación:** ¿tanh? ¿Probar otras luego?

---

## FASE 5: Comparación y Selección

### Objetivo de la fase
Analizar loss de ambos modelos y decidir cuál usar para el estudio de generalización.

### Tareas

1. **Análisis comparativo de loss curves**
   - Gráfica con loss lineal y no lineal superpuestos
   - Preguntas a responder:
     - ¿Observan underfitting? (loss alto al final)
     - ¿Observan saturación? (no lineal no mejora vs lineal)
     - ¿Cuál tiene mejor potencial de aprendizaje?

2. **Decisión y justificación**
   - Elegir uno (recomendación: probablemente no lineal)
   - Documentar razonamiento basado en análisis

3. **Preparación para Fase 6**
   - Guardar config del modelo elegido
   - Anotar hiperparámetros finales (learning_rate, epochs, etc)

### Output esperado
- Gráficas comparativas
- Documento breve con decisión justificada
- Modelo elegido listo para reutilizar

---

## FASE 6: Estudio de Generalización

### Objetivo de la fase
Evaluar cómo generaliza el modelo elegido en datos nunca vistos (validación).

### Tareas

1. **Realizar train/validation split**
   - Proporción: **80-20** (80% train, 20% val)
   - Estrategia: **Random split** (simple y suficiente para este dataset)
   - Usar función en `data.py` para esto

2. **Reentrenar modelo desde cero**
   - Crear nuevo `SimplePerceptron` con mismos hiperparámetros que Fase 5
   - Entrenar con **80% de los datos** (training set)
   - Monitorear loss en **ambos conjuntos** por época

3. **Seleccionar métricas de evaluación**
   - **MSE:** Error cuadrático medio (reportar en training y validation)
   - **MAE:** Error absoluto medio (más interpretable)
   - Justificación: Para problemas de regresión, son las métricas estándar

4. **Análisis de generalización**
   - Gráficar loss training vs validation por época
   - Observar: ¿overfitting?, ¿underfitting?, ¿buen balance?
   - Calcular diferencia: `loss_val - loss_train` (debe ser pequeña)

5. **Evaluación final**
   - Reportar MSE y MAE en validation set
   - Análisis: ¿qué tan bien generaliza?

### Output esperado
- Modelo entrenado con split
- Gráficas de training vs validation curves
- Tabla de métricas (MSE, MAE en training y validation)

---

## FASE 7: Umbral de Detección y Recomendaciones

### Objetivo de la fase
Determinar el mejor umbral de detección de fraude para presentar al cliente.

### Tareas

1. **Análisis de predicciones en validation set**
   - Histograma de probabilidades predichas
   - Estadísticas (media, std, percentiles)
   - Comparar distribución con target `big_model_fraud_probability`

2. **Curva Precision-Recall (o F1 vs threshold)**
   - Para cada umbral posible (0.1 a 0.9 en pasos de 0.1):
     - Calcular: TP, FP, TN, FN (usando `flagged_fraud` como ground truth)
     - Calcular: Precision, Recall, F1-score
   - Graficar: Precision y Recall vs threshold

3. **Seleccionar umbral óptimo**
   - Considerar trade-off: precisión vs recall
   - **Pregunta al cliente (o asumir):** ¿Qué es más importante?
     - Detectar fraudes (recall alto) → umbral bajo
     - Evitar falsos positivos (precisión alta) → umbral alto
   - Sugerir umbral que maximice F1 o según prioridad del cliente

4. **Recomendación final**
   - Umbral sugerido (ej: 0.5)
   - Métricas asociadas (precision, recall, F1)
   - Justificación clara

### Output esperado
- Gráficas de Precision-Recall vs threshold
- Tabla de métricas para umbrales candidatos
- Recomendación final con justificación

---

## Opcionales (implementar después de tener base sólida)

### Optimizaciones de Entrenamiento
- **SGD con momentum:** Reemplazar SGD simple con versión con momentum (parámetro γ = 0.9)
- **Adaptive learning rate:** Implementar learning rate decaying (reducir con epochs)
- **Adam optimizer:** Versión adaptativa más robusta que SGD

### Variantes de Arquitectura
- **Probar diferentes learning rates:** 0.001, 0.01, 0.1 (comparar convergencia)
- **Probar diferentes epochs:** Determinar punto óptimo de convergencia

### Feature Engineering
- **Análisis de features:** ¿Qué features contribuyen más?
- **Feature selection:** Eliminar features poco correlacionados con target
- **Feature creation:** Crear nuevos features a partir de los existentes (ej: ratio amount/quantity)

### Análisis de Calibración
- **Verificar calibración:** ¿Las probabilidades predichas son bien calibradas?
- **Ajuste de calibración:** Si es necesario, aplicar técnicas como Platt scaling o isotonic regression
- **Impacto:** Cómo afecta a la recomendación del umbral

---

## Cronograma Estimado
1. Fase 1: Exploración - 1-2 horas
2. Fase 2: Arquitectura - 2-3 horas
3. Fase 3: Lineal - 1-2 horas
4. Fase 4: No lineal - 1-2 horas
5. Fase 5: Comparación - 1 hora
6. Fase 6: Generalización - 2-3 horas
7. Fase 7: Umbral - 1-2 horas

**Total estimado (obligatorio): 9-15 horas**
**Total con opcionales: 15-25 horas**

---

## Notas Generales
- El enunciado enfatiza exploración de datos antes de modelado
- Configuración centralizada: guardar/cargar configs desde JSON
- Reutilizar `SimplePerceptron` de `/src/` (ya está probada)
- Todo código de Ej1 en `/src/ej1/`
- Gráficas y análisis en `/src/ej1/evaluation.py`
- Modelos y datos en `/experiments/` o carpeta similar para rastreabilidad
