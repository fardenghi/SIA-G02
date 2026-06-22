"""Persistencia del modelo entrenado: guardar/cargar un VAE (y el modelo two-stage completo).

Un `VAE` queda determinado por su arquitectura (dims + activaciones + init + loss) y su vector
plano de pesos (`get_params`/`set_params`). Guardar = `np.savez(arquitectura + params)`; cargar =
reconstruir el `VAE` con esa arquitectura y `set_params`. No hace falta reentrenar: con el modelo
cargado se puede `encode`/`decode` cualquier punto del latente al instante (p. ej. para interpolar)
y generar desde el prior si se guardó también la etapa 2.

Formato (`fmt="vae-ckpt-1"`):
  - etapa 1 (siempre): enc1, latent1, activation1, out_act1, init1, loss1, params1, size.
  - etapa 2 (opcional, para generar): enc2, latent2, ..., params2, code_mean, code_std
    (las stats z-score con que se estandarizaron los códigos de la etapa 1 antes de la etapa 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .vae import VAE


def _vae_fields(vae: VAE, suffix: str) -> dict:
    """Campos de arquitectura + pesos de un `VAE`, con sufijo (`1`/`2`) para el npz."""
    return {
        f"enc{suffix}": np.asarray(vae.encoder_sizes, dtype=np.int64),
        f"latent{suffix}": np.int64(vae.latent_dim),
        f"activation{suffix}": np.str_(vae.activation),
        f"out_act{suffix}": np.str_(vae.output_activation),
        f"init{suffix}": np.str_(vae.init),
        f"loss{suffix}": np.str_(vae.loss),
        f"params{suffix}": vae.get_params(),
    }


def _build_vae(z, suffix: str) -> VAE:
    """Reconstruye un `VAE` desde los campos del npz y carga sus pesos."""
    vae = VAE(
        [int(x) for x in z[f"enc{suffix}"]],
        latent_dim=int(z[f"latent{suffix}"]),
        activation=str(z[f"activation{suffix}"]),
        output_activation=str(z[f"out_act{suffix}"]),
        init=str(z[f"init{suffix}"]),
        loss=str(z[f"loss{suffix}"]),
    )
    vae.set_params(z[f"params{suffix}"])
    return vae


def save_vae(path: str | Path, vae1: VAE, size: int, *, vae2: VAE | None = None,
             code_mean: np.ndarray | None = None, code_std: np.ndarray | None = None,
             run_args: dict | None = None) -> Path:
    """Guarda el VAE (etapa 1) y, si se pasa, el modelo two-stage completo en un `.npz`.

    `size` es el lado en píxeles de la imagen (la entrada es `size*size`), para que el que carga
    sepa cómo reshapear sin adivinar. `vae2`/`code_mean`/`code_std` habilitan generar desde el
    prior (ver `make_sampler`); para sólo interpolar/decodificar alcanza con la etapa 1.

    `run_args` (opcional): dict de la configuración de la corrida (dataset, seed, n_aug, ...),
    serializado a JSON. Permite que el modo carga reconstruya el MISMO split y los títulos sin
    que el usuario reescriba los flags (ver `load_vae` -> clave `run_args`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"fmt": np.str_("vae-ckpt-1"), "size": np.int64(size)}
    data.update(_vae_fields(vae1, "1"))
    if vae2 is not None:
        data.update(_vae_fields(vae2, "2"))
        data["code_mean"] = np.asarray(code_mean, dtype=np.float64)
        data["code_std"] = np.asarray(code_std, dtype=np.float64)
    if run_args is not None:
        data["run_args"] = np.str_(json.dumps(run_args))
    np.savez(path, **data)
    return path


def load_vae(path: str | Path) -> dict:
    """Carga un checkpoint. Devuelve dict con `vae1`, `size` y (si existe) `vae2`/`code_mean`/`code_std`.

    Uso típico (interpolar, sin reentrenar):
        ck = load_vae("model.npz"); vae = ck["vae1"]
        mu, _ = vae.encode(X[[a, b]])
        zs = np.stack([(1-t)*mu[0] + t*mu[1] for t in np.linspace(0, 1, 11)])
        imgs = vae.decode(zs)                       # (11, size*size)
    """
    z = np.load(path, allow_pickle=False)
    ck: dict = {"vae1": _build_vae(z, "1"), "size": int(z["size"])}
    if "params2" in z.files:
        ck["vae2"] = _build_vae(z, "2")
        ck["code_mean"] = z["code_mean"]
        ck["code_std"] = z["code_std"]
    if "run_args" in z.files:
        ck["run_args"] = json.loads(str(z["run_args"]))
    return ck


def make_sampler(ck: dict):
    """A partir de un checkpoint con etapa 2, devuelve `sample(n, rng) -> imágenes` (prior→imagen).

    Réplica de la generación two-stage del runner: z₂~N(0,I) → decode etapa2 → des-estandarizar
    → decode etapa1. Falla si el checkpoint no incluye la etapa 2.
    """
    if "vae2" not in ck:
        raise ValueError("el checkpoint no tiene etapa 2: no se puede generar desde el prior")
    vae1, vae2 = ck["vae1"], ck["vae2"]
    cm, cs = ck["code_mean"], ck["code_std"]

    def sample(n: int, rng: np.random.Generator) -> np.ndarray:
        z2 = rng.standard_normal((n, vae2.latent_dim))
        codes_hat = vae2.decode(z2) * cs + cm
        return vae1.decode(codes_hat)

    return sample
