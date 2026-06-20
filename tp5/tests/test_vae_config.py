"""Tests de la carga/validación del config del VAE (Fase 3)."""

import json

import pytest

from autoencoder.config import ConfigError
from autoencoder.vae_config import load_vae_config


def _write(tmp_path, obj):
    p = tmp_path / "vae.json"
    p.write_text(json.dumps(obj))
    return p


BASE = {
    "data": {"size": 28},
    "architecture": {"encoder_layers": [784, 256, 64], "latent_dim": 2},
    "training": {"loss": "bce", "epochs": 100, "lr": 1e-3},
}


def test_load_valid_config(tmp_path):
    cfg = load_vae_config(_write(tmp_path, BASE))
    assert cfg.architecture.encoder_layers == [784, 256, 64]
    assert cfg.architecture.latent_dim == 2
    assert cfg.training.loss == "bce"
    assert cfg.data.size == 28
    assert cfg.data.augment.enabled is False


def test_defaults_applied(tmp_path):
    cfg = load_vae_config(_write(tmp_path, BASE))
    assert cfg.training.beta == 1.0
    assert cfg.training.beta_warmup == 0
    assert cfg.architecture.activation == "relu"
    assert cfg.architecture.output_activation == "sigmoid"


def test_generation_defaults_to_gaussian(tmp_path):
    cfg = load_vae_config(_write(tmp_path, BASE))
    assert cfg.generation.prior == "gaussian"
    assert cfg.generation.gmm_k == 8


def test_generation_gmm_config(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["generation"] = {"prior": "gmm", "gmm_k": 12}
    cfg = load_vae_config(_write(tmp_path, obj))
    assert cfg.generation.prior == "gmm"
    assert cfg.generation.gmm_k == 12


def test_generation_invalid_prior_raises(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["generation"] = {"prior": "vampprior"}
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_generation_unknown_key_raises(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["generation"] = {"prior": "gmm", "temperature": 0.5}
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_missing_encoder_layers_raises(tmp_path):
    obj = {"data": {"size": 28}, "architecture": {"latent_dim": 2}, "training": {}}
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_encoder_input_must_match_size(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["architecture"]["encoder_layers"] = [256, 64]  # 256 != 28*28
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_invalid_activation_raises(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["architecture"]["activation"] = "selu"
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_invalid_loss_raises(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["training"]["loss"] = "hinge"
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_unknown_key_raises(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["training"]["momentum"] = 0.9
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_negative_beta_raises(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["training"]["beta"] = -1.0
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_augment_parsed(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["data"]["augment"] = {"enabled": True, "n_aug": 6, "max_rot": 10}
    cfg = load_vae_config(_write(tmp_path, obj))
    assert cfg.data.augment.enabled is True
    assert cfg.data.augment.n_aug == 6
    assert cfg.data.augment.max_rot == 10.0


def test_decoder_layers_validation(tmp_path):
    obj = json.loads(json.dumps(BASE))
    obj["architecture"]["decoder_layers"] = [2, 64, 784]
    cfg = load_vae_config(_write(tmp_path, obj))
    assert cfg.architecture.decoder_layers == [2, 64, 784]
    # Debe terminar en size*size.
    obj["architecture"]["decoder_layers"] = [2, 64, 100]
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


# ----------------------------------------------------- kind: "conv" (Iteración 2)

CONV_BASE = {
    "data": {"size": 28},
    "architecture": {"kind": "conv", "latent_dim": 2, "conv_channels": [16, 32],
                     "dense_hidden": 64},
    "training": {"loss": "bce", "epochs": 100, "lr": 1e-3},
}


def test_default_kind_is_mlp(tmp_path):
    # Retrocompatibilidad: configs sin 'kind' siguen siendo mlp.
    cfg = load_vae_config(_write(tmp_path, BASE))
    assert cfg.architecture.kind == "mlp"


def test_conv_kind_valid(tmp_path):
    cfg = load_vae_config(_write(tmp_path, CONV_BASE))
    assert cfg.architecture.kind == "conv"
    assert cfg.architecture.conv_channels == [16, 32]
    assert cfg.architecture.dense_hidden == 64


def test_conv_does_not_require_encoder_layers(tmp_path):
    # kind=conv no usa encoder_layers; debe cargar sin ellos.
    cfg = load_vae_config(_write(tmp_path, CONV_BASE))
    assert cfg.architecture.latent_dim == 2


def test_invalid_kind_raises(tmp_path):
    obj = json.loads(json.dumps(CONV_BASE))
    obj["architecture"]["kind"] = "transformer"
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_conv_incompatible_size_raises(tmp_path):
    obj = json.loads(json.dumps(CONV_BASE))
    obj["data"]["size"] = 10  # 10 → 5 → 3, 3·4 = 12 ≠ 10
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_conv_decoder_layers_forbidden(tmp_path):
    obj = json.loads(json.dumps(CONV_BASE))
    obj["architecture"]["decoder_layers"] = [2, 64, 784]
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))


def test_conv_bad_channels_raises(tmp_path):
    obj = json.loads(json.dumps(CONV_BASE))
    obj["architecture"]["conv_channels"] = [16, -1]
    with pytest.raises(ConfigError):
        load_vae_config(_write(tmp_path, obj))
