"""Round-trip de la persistencia del VAE (`save_vae`/`load_vae`), incluida la config `run_args`."""

import numpy as np

from autoencoder.checkpoint import load_vae, make_sampler, save_vae
from autoencoder.vae import VAE

SIZE = 8
DIM = SIZE * SIZE


def _vae(latent=3, seed=0):
    return VAE([DIM, 16], latent_dim=latent, activation="relu",
               output_activation="sigmoid", init="he_normal", seed=seed)


def test_roundtrip_stage1_preserves_weights(tmp_path):
    vae = _vae()
    X = np.random.default_rng(1).uniform(0, 1, size=(5, DIM))
    expected = vae.decode(vae.encode(X)[0])
    ck = load_vae(save_vae(tmp_path / "m.npz", vae, SIZE))
    got = ck["vae1"].decode(ck["vae1"].encode(X)[0])
    assert np.allclose(got, expected)
    assert ck["size"] == SIZE
    assert "run_args" not in ck  # no se pasó -> no aparece


def test_roundtrip_run_args(tmp_path):
    run_args = {"dataset": "celeba", "size": SIZE, "seed": 7, "n_aug": 4,
                "val_mode": "disjoint", "latent": 3, "digits": None, "beta": 1.0}
    path = save_vae(tmp_path / "m.npz", _vae(), SIZE, run_args=run_args)
    ck = load_vae(path)
    assert ck["run_args"] == run_args  # JSON preserva tipos (str/int/float/None)


def test_roundtrip_two_stage_sampler(tmp_path):
    vae1 = _vae(latent=3)
    # la etapa 2 modela los códigos de la etapa 1: decodifica al latente de la etapa 1 (=3).
    vae2 = VAE([3, 8], latent_dim=2, activation="relu", output_activation="linear",
               init="he_normal", loss="mse", seed=0)
    cm = np.zeros(3); cs = np.ones(3)
    path = save_vae(tmp_path / "m.npz", vae1, SIZE, vae2=vae2, code_mean=cm, code_std=cs)
    sample = make_sampler(load_vae(path))
    imgs = sample(4, np.random.default_rng(0))
    assert imgs.shape == (4, DIM)
