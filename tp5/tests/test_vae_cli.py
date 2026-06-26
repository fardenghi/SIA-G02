"""Smoke de la integración CLI (Fase 2.3): dispatch mlp/conv y corrida conv end-to-end."""

import json
from pathlib import Path

import pytest

from autoencoder import emoji_data, vae_cli
from autoencoder.conv_vae import ConvVAE
from autoencoder.vae import VAE
from autoencoder.vae_config import load_vae_config

_HAS_FONT = Path(emoji_data.DEFAULT_FONT).exists()
_SKIP_NO_FONT = pytest.mark.skipif(
    not _HAS_FONT, reason=f"falta la fuente de emojis: {emoji_data.DEFAULT_FONT}")


def _write(tmp_path, obj):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(obj))
    return p


def test_build_model_dispatches_conv(tmp_path):
    obj = {
        "data": {"size": 28},
        "architecture": {"kind": "conv", "latent_dim": 2, "conv_channels": [16, 32]},
        "training": {"epochs": 10},
    }
    model = vae_cli.build_model(load_vae_config(_write(tmp_path, obj)))
    assert isinstance(model, ConvVAE)
    assert model.latent_dim == 2 and model.feat_h == 7


def test_build_model_dispatches_mlp(tmp_path):
    obj = {
        "data": {"size": 28},
        "architecture": {"encoder_layers": [784, 256, 64], "latent_dim": 2},
        "training": {"epochs": 10},
    }
    model = vae_cli.build_model(load_vae_config(_write(tmp_path, obj)))
    assert isinstance(model, VAE)


@_SKIP_NO_FONT
def test_cli_run_conv_smoke(tmp_path):
    # Corrida conv completa con pocas épocas: entrena, exporta métricas y figuras.
    obj = {
        "name": "conv_smoke",
        "data": {"size": 28, "augment": {"enabled": False}},
        "architecture": {"kind": "conv", "latent_dim": 2, "conv_channels": [16, 32],
                         "dense_hidden": 32},
        "training": {"loss": "bce", "epochs": 2, "lr": 1e-3, "beta": 1.0,
                     "log_every": 1, "seed": 0},
        "output": {"metrics_csv": str(tmp_path / "m.csv"),
                   "plots_dir": str(tmp_path / "plots")},
    }
    result = vae_cli.run(str(_write(tmp_path, obj)))
    assert isinstance(result["_vae"], ConvVAE)
    assert (tmp_path / "m.csv").exists()
    # El pipeline genera las figuras estándar (reusadas tal cual de la MLP-VAE).
    plots = tmp_path / "plots" / "conv_smoke"
    assert (plots / "reconstruction.png").exists()
    assert (plots / "samples.png").exists()
    assert (plots / "kl_per_dim.png").exists()
