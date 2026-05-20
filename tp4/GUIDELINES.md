# Lineamientos del proyecto

## Estructura de trabajo

- Implementar en fases; al terminar cada fase escribir tests que la cubran.
- Si los tests pasan, hacer commit. Si no, corregir antes de avanzar.
- Todos los archivos nuevos van en módulos propios (ej. `hopfield/`).
- Preferir editar archivos existentes sobre crear nuevos.

## Commits

- Mensajes concisos, sin detalles de fases del plan.
- Sin co-author en los mensajes.

## Código

- Sin comentarios salvo cuando el *porqué* no es obvio (restricción oculta, invariante sutil, workaround).
- Sin docstrings multi-párrafo ni bloques de comentarios multi-línea.
- Sin manejo de errores para escenarios imposibles; confiar en las garantías del framework.
- No agregar features, refactores ni abstracciones más allá de lo que pide la tarea.
- No diseñar para requerimientos futuros hipotéticos.

## Outputs (plots, consola)

- No incluir detalles de fase del plan en títulos, etiquetas ni nombres de archivo.
- Los plots se guardan en `output/hopfield/`.
- Nombres de archivo descriptivos del contenido, no del paso del plan.

## Tests

- Cubrir la implementación de cada fase antes del commit.
- Tests en `tests/test_hopfield.py`.
- Usar pytest.

## Análisis

- Hay libertad creativa sobre qué aspectos analizar, siempre que cubra los requerimientos del enunciado.
- Requerimientos mínimos:
  - Parte a: almacenar 4 patrones de letras 5×5, recuperar versiones ruidosas, mostrar cada paso.
  - Parte b: ingresar patrón muy ruidoso e identificar un estado espúreo.
