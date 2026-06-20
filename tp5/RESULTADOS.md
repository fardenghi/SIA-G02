# Resultados — Iteraciones del VAE (TP5 Ej2)

Conclusiones empíricas de las dos iteraciones de mejora del VAE (latente y convoluciones).
Ambas están **completas y cerradas**. Las figuras viven en `out/` (no versionado); acá van los
números y la interpretación.

---

## Iteración 1 — Agrandar el espacio latente

**Barrido `latent_dim ∈ {2,4,8,16,32}`** (MLP, 32 emojis sin augment, 5000 épocas, β=1):

| latent_dim | recon_det | unidades activas |
|---:|---:|---:|
| 2  | 309.65 | 2 |
| 4  | 308.74 | 4 |
| 8  | 308.84 | 8 |
| 16 | 309.10 | 13 |
| 32 | 308.93 | 13 |

**Hallazgo:** la reconstrucción es **plana** (~309 ± 0.5 nats) en todo el rango — agrandar el
cuello no mejora nada. Las unidades activas saturan en **~13** (dimensionalidad intrínseca
aproximada del dataset). El encoder empaqueta más información al darle espacio (KL y activas
suben), pero el decoder no la traduce en mejores píxeles.

**Conclusión:** el cuello latente **no era el limitante** del MLP. Por el gate del roadmap
("si el salto es chico, seguir con la Iteración 2"), el limitante son las *features* → se
justifica probar convoluciones.

---

## Iteración 2 — Encoder/decoder convolucional

`ConvVAE` implementada desde cero (Conv2D im2col/col2im, Upsample2D, Flatten/Reshape), con
gradient-check de cada capa y del **ELBO completo**, reutilizando `reparameterize`/`KL`/`losses`
del VAE sin tocar `vae.py`. Conmutación de arquitectura por config (`architecture.kind:
"mlp"|"conv"`).

### Cara a cara a latente 2 (augment on, 5000 épocas)

| | recon_det | KL |
|---|---:|---:|
| MLP (`base`) | **327.6** | 7.06 |
| CNN (`conv`) | 341.8 | 6.60 |

La CNN reconstruye **~14 nats peor** y visualmente más borrosa (colapsa casi todos los emojis a
un smiley genérico; el MLP distingue anteojos/ojos-corazón).

### Barrido de latente, MLP vs CNN (32 emojis sin augment, 2000 épocas)

| latent | MLP | CNN | gap (CNN−MLP) |
|---:|---:|---:|---:|
| 2  | 315.7 | 332.4 | +16.7 |
| 4  | 309.3 | 320.6 | +11.3 |
| 8  | 309.6 | 317.4 | +7.7 |
| 16 | 309.9 | 317.3 | +7.4 |
| 32 | 310.3 | 317.9 | +7.6 |

**La CNN nunca cruza al MLP.** El gap se reduce de +16.7 (latente 2) a ~+7.5 (latente 8) y
**se clava ahí**: la CNN tiene un piso irreducible ~7.5 nats peor. Además, en latente 32 el MLP
poda a 23 dims activas mientras la CNN mantiene las 32 (KL mayor): **codifica más y reconstruye
peor → cuello en el decoder.**

### Prueba de capacidad: canales `[16,32]` → `[32,64]`

| latent | CNN `[16,32]` | CNN `[32,64]` | MLP | gap restante |
|---:|---:|---:|---:|---:|
| 8  | 317.4 | **313.1** | 309.6 | +3.5 |
| 16 | 317.3 | **313.8** | 309.9 | +3.9 |

Duplicar canales **cierra ~la mitad del gap** (+7.5 → +3.7). El cuello del decoder era **mixto**:
~mitad **capacidad** (la cierra el ancho) y ~mitad **sesgo estructural** del upsample-nearest +
pesos compartidos (que más canales no arreglan).

### Conclusión de la Iteración 2

Para **reconstrucción pura (BCE) sobre 32 emojis centrados, el MLP gana.** La CNN se le acerca
con un decoder ancho pero no lo cruza. Es coherente con la teoría: el *prior* convolucional
(pesos compartidos, invariancia a traslación, suavizado) es un **regularizador** que en un
dataset chico y centrado **cuesta más capacidad de memorización de la que aporta**. La conv
pagaría en **generalización / datos naturales / nitidez de muestras**, no en el recon BCE de este
dataset. La `ConvVAE` queda **implementada, validada (gradient-check) y caracterizada**; se elige
el MLP como mejor reconstructor para este caso.

---

## Estado

- **Iteración 1:** cerrada — latente no es el cuello del MLP (intrínseco ~13).
- **Iteración 2:** cerrada — CNN implementada/validada; MLP mejor en recon para este dataset.

Reproducción:

```bash
uv run python scripts/vae_latent_sweep.py --kind mlp  --epochs 2000   # barrido MLP
uv run python scripts/vae_latent_sweep.py --kind conv --epochs 2000   # barrido CNN
uv run python scripts/vae_latent_sweep.py --kind conv --conv-channels 32 64 --latents 8 16 --tag conv_wide
uv run autoencoder-vae --config configs/vae/conv.json                 # CNN completa (figuras)
uv run pytest -q                                                      # suite (140 tests)
```
