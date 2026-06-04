import json

import pytest

from autoencoder.config import ConfigError, load_config


def _write(tmp_path, obj):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(obj))
    return p


BASE = {
    "architecture": {"encoder_layers": [35, 10, 2]},
    "training": {"optimizer": "adam", "loss": "bce", "epochs": 10, "restarts": 1},
}


def test_load_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg.architecture.encoder_layers == [35, 10, 2]
    assert cfg.training.optimizer == "adam"
    assert cfg.denoising.enabled is False


def test_missing_encoder_layers(tmp_path):
    bad = {"architecture": {"activation": "tanh"}}
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_invalid_optimizer(tmp_path):
    bad = {"architecture": {"encoder_layers": [35, 2]},
           "training": {"optimizer": "powell"}}
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_invalid_init(tmp_path):
    bad = {"architecture": {"encoder_layers": [35, 2], "init": "magic"}}
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_unknown_key(tmp_path):
    bad = {"architecture": {"encoder_layers": [35, 2]}, "training": {"foo": 1}}
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_ships_all_five_named_configs():
    names = ["base_adam", "base_lbfgs", "deep", "wide_relu", "naive_init"]
    for n in names:
        cfg = load_config(f"configs/{n}.json")
        assert cfg.name == n


def test_1a2_progression_configs_load():
    import glob

    paths = sorted(glob.glob("configs/1a2/*.json"))
    assert len(paths) == 6
    for p in paths:
        cfg = load_config(p)
        assert cfg.architecture.encoder_layers[-1] == 2  # latente 2D fijo en 1a
        assert cfg.training.stop_at is None  # corren los restarts completos
