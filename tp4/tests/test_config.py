import json
import pytest
from pathlib import Path

CONFIG_PATH = Path("configs/kohonen_europe.json")
REQUIRED_KEYS = {
    "data", "grid_rows", "grid_cols",
    "lr", "lr_decay",
    "radius", "radius_decay",
    "neighborhood_fn",
    "epochs", "seed", "output_dir",
}
VALID_DECAYS = {"exponential", "linear"}
VALID_NEIGHBORHOODS = {"gaussian", "bubble"}


@pytest.fixture(scope="module")
def config():
    assert CONFIG_PATH.exists(), f"{CONFIG_PATH} not found"
    with CONFIG_PATH.open() as f:
        return json.load(f)


def test_config_is_valid_json():
    assert CONFIG_PATH.exists()
    with CONFIG_PATH.open() as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_config_has_all_required_keys(config):
    missing = REQUIRED_KEYS - config.keys()
    assert not missing, f"Missing keys: {missing}"


def test_grid_dimensions_positive_int(config):
    assert isinstance(config["grid_rows"], int) and config["grid_rows"] > 0
    assert isinstance(config["grid_cols"], int) and config["grid_cols"] > 0


def test_lr_positive_float(config):
    assert isinstance(config["lr"], (int, float)) and config["lr"] > 0


def test_lr_decay_valid(config):
    assert config["lr_decay"] in VALID_DECAYS


def test_radius_positive(config):
    assert isinstance(config["radius"], (int, float)) and config["radius"] > 0


def test_radius_decay_valid(config):
    assert config["radius_decay"] in VALID_DECAYS


def test_neighborhood_fn_valid(config):
    assert config["neighborhood_fn"] in VALID_NEIGHBORHOODS


def test_epochs_positive_int(config):
    assert isinstance(config["epochs"], int) and config["epochs"] > 0


def test_seed_is_int(config):
    assert isinstance(config["seed"], int)


def test_data_path_exists(config):
    assert Path(config["data"]).exists(), f"data path {config['data']!r} not found"


def test_config_builds_som(config):
    from kohonen.som import SOM
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(config["data"])
    X = StandardScaler().fit_transform(df.drop(columns=["Country"]))
    som = SOM(
        grid_rows=config["grid_rows"],
        grid_cols=config["grid_cols"],
        input_dim=X.shape[1],
        lr=config["lr"],
        lr_decay=config["lr_decay"],
        radius=config["radius"],
        radius_decay=config["radius_decay"],
        neighborhood_fn=config["neighborhood_fn"],
        epochs=config["epochs"],
        seed=config["seed"],
    )
    assert som.weights.shape == (config["grid_rows"], config["grid_cols"], X.shape[1])
