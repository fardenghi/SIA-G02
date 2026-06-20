# Corrida grande (overnight) — MLP-VAE vs ConvVAE en caras+animales

Guía autocontenida para lanzar y analizar la **prueba final** del Ejercicio 2: red de alta
capacidad + dataset homogéneo (caras/animales), para ver si la CNN gana **con figuras
reconocibles** (no los manchones de los experimentos chicos). Contexto completo en
[`README.md`](README.md) → sección "Iteraciones de mejora".

## TL;DR (sesión nueva)

```bash
mkdir -p out/big_run
nohup uv run python scripts/big_run.py --size 56 --latent 48 --epochs 2000 \
      > out/big_run/run.log 2>&1 &
```

- **Es un script no-interactivo: corre solo de principio a fin, NO hace preguntas.** No requiere
  intervención. `nohup ... &` lo deja corriendo aunque cierres la terminal.
- **Duración:** ~**7.4 h** la CNN (13.4 s/época × 2000) + ~media hora el MLP ≈ **~8 h**. Una noche.
- Al terminar, mirá `out/big_run/summary.txt`, `cnn_progress.png` y `mlp_progress.png`, y seguí la
  sección **"Qué analizar"** de abajo.

## Lanzar y monitorear

| Acción | Comando |
|---|---|
| Lanzar (overnight, sobrevive logout) | `mkdir -p out/big_run && nohup uv run python scripts/big_run.py --size 56 --latent 48 --epochs 2000 > out/big_run/run.log 2>&1 &` |
| Ver progreso en vivo | `tail -f out/big_run/run.log` |
| Ver la última figura | abrir `out/big_run/cnn_progress.png` (se actualiza cada 100 épocas) |
| ¿Sigue corriendo? | `pgrep -af big_run` |
| Abortar | `pkill -f big_run` |

El log imprime el **benchmark de costo** al arrancar (`s/época → horas estimadas`): si el número
no te cierra, abortá y reajustá flags antes de comprometer la noche.

## Qué produce (`out/big_run/`)

| Archivo | Qué es |
|---|---|
| `run.log` | stdout: benchmark, progreso (`recon/kl/lr` cada 100 ép) y RESUMEN final |
| `summary.txt` | **métricas finales** CNN vs MLP + gap + params (lo primero a mirar) |
| `cnn_progress.png` / `mlp_progress.png` | figura **in / recon / sample** (10 emojis), actualizada cada 100 ép |
| `cnn_ckpt.npy` / `mlp_ckpt.npy` | pesos (checkpoint); si se cuelga, conservás el último estado |

> Checkpoints/figuras viven en `out/` (no versionado). Si querés guardarlos, copialos aparte.

---

## Qué analizar — MÉTRICAS (`summary.txt` + `run.log`)

1. **El gap `recon_det/px` (CNN − MLP)** — el titular. `< 0` ⇒ **la CNN gana** en reconstrucción.
   Compará con los hitos previos para ubicar la tendencia:
   - 32 emojis (memorización): **+0.0100** (MLP gana)
   - 1294 glifos, latente 8: **+0.0060**
   - 512 glifos, latente 64: **−0.0054** (CNN empieza a ganar)
   - **Esta corrida** (220 caras/animales, red grande): ¿el gap se vuelve **más negativo**?
2. **Parámetros CNN vs MLP** — si la CNN gana con **menos o comparables** params, es la historia de
   **eficiencia** del prior convolucional. Anotalo.
3. **Trayectoria de `recon` en el log** — ¿se **aplanó** (convergió) o **seguía bajando** al final?
   Si seguía bajando con claridad ⇒ **faltaron épocas** (relanzar con más, el cosine LR ya ayuda).
4. **KL** — que **no haya colapsado a ≈0** (eso es *posterior collapse*: el latente se ignora,
   muestras idénticas → subir `--beta` o el warmup). Con `β=0.5` esperá KL moderada/alta, no nula.
5. **LR** — confirmá que el cosine bajó a **~0** al final (annealing correcto).

## Qué analizar — FIGURAS (`cnn_progress.png` vs `mlp_progress.png`)

Cada figura tiene 3 filas: **in** (entrada), **recon** (reconstrucción determinista, z=μ) y
**sample** (muestras del prior z~N(0,I)). Comparar las dos imágenes lado a lado.

1. **Reconstrucción (recon vs in):** ¿los emojis son **reconocibles**? ¿La CNN iguala o **supera**
   al MLP en nitidez y fidelidad (bordes de ojos/boca, orejas de animales)? Este es el objetivo
   visual que no logramos a baja escala.
2. **Generación (fila sample):** ¿genera **caras/animales coherentes**, no manchones? Clave para un
   modelo generativo. Mirá si la **CNN sale más suave y coherente** y el **MLP tiene ruido
   sal-y-pimienta** (su patrón típico). Si la CNN genera mejor, es la confirmación más fuerte.
3. **Comparar contra lo previo:** ¿el dataset homogéneo + red grande dio el **salto de calidad**
   esperado respecto de los manchones de `out/more_data/`?

**Señales de problema:**
- Todas las salidas casi iguales → *posterior collapse* (KL≈0): subir `--beta`/warmup.
- Ruido sal-y-pimienta fuerte → β bajo y/o el MLP; probar `--beta` un poco más alto.
- Borrosidad uniforme sin estructura → subentrenado (más épocas) o latente chico (subir `--latent`).
- Patrón de tablero (checkerboard) → no debería pasar (usamos upsample nearest); si aparece, avisar.

## Conclusiones a sacar (árbol de decisión)

- **CNN gana en número Y se ve igual/mejor** → **tesis confirmada del todo**: con datos +
  capacidad + homogeneidad, la convolución paga en **calidad y eficiencia**. → documentar en el
  README (sección de iteraciones) y cerrar el Ejercicio 2.
- **CNN gana en número pero ambas siguen borrosas** → estamos en el **techo de nitidez del VAE**
  (limitación inherente, no de tuning). Documentar la tensión "calidad vs CNN-gana"; opcional:
  subir resolución / bajar β / más épocas.
- **El MLP todavía gana** → 220 glifos homogéneos quizá siguen siendo "memorizables"; opciones:
  más glifos (usar `_EMOJI_RANGES` completo, ~1300), más épocas, o aceptar que a esta escala el
  MLP rinde.
- **`recon` seguía bajando** → relanzar con más `--epochs`.

## Ajustes (flags) y su costo

| Flag | Default | Efecto / costo |
|---|---|---|
| `--epochs` | 2000 | lineal en tiempo |
| `--size` | 56 | **cuadrático** en costo (48 ≈ 0.7×, 64 ≈ 1.3×) |
| `--channels` | `32 64 128` | capacidad de la CNN ↔ velocidad. Nº de valores = etapas de downsample; `size` debe ser divisible por `2^nº_etapas` |
| `--latent` | 48 | casi gratis; subir para más fidelidad |
| `--dense-hidden` | 256 | ancho del cuello denso |
| `--beta` | 0.5 | más bajo = más nítido (peor generación); más alto = más regularizado |
| `--batch` | 64 | tamaño de mini-batch |
| `--ckpt-every` | 100 | frecuencia de checkpoint/figura |

## Notas para retomar en sesión nueva

- Todo el contexto del Ejercicio 2 (iteraciones, color, resolución, más datos) está en el README.
- La `ConvVAE`, el mini-batch, el cosine LR y el loader `load_many_emojis(ranges=FACE_ANIMAL_RANGES)`
  están implementados y testeados (`uv run pytest -q`, 149 tests).
- Si la corrida confirma el salto de calidad, el paso natural es **volcar el resultado al README**
  como cierre del arco MLP-vs-CNN.
