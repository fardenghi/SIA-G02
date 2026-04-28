import numpy as np
import pytest
from common.datasets import xor_dataset
from common.losses import mse
from common.mlp import MLP
from common.optimizers import SGD


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def xor_data():
    X, y = xor_dataset()
    return X, y.reshape(-1, 1)


def train_xor(arch, lr=0.5, epochs=10000, seed=42):
    X, y = xor_data()
    mlp = MLP(arch, activation="tanh", output_activation="tanh",
              loss="mse", seed=seed)
    mlp.fit(X, y, epochs=epochs, batch_size=4,
            optimizer=SGD(lr=lr), verbose=False)
    return mlp, X, y


# ------------------------------------------------------------------
# 5.9 Gradient check
# ------------------------------------------------------------------

def test_gradient_check():
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([[-1.0], [1.0], [1.0], [-1.0]])

    mlp = MLP([2, 3, 2, 1], activation="tanh", output_activation="tanh",
              loss="mse", seed=7)

    mlp.forward(X)
    grads = mlp.backward(y)

    eps = 1e-5
    max_rel = 0.0

    for l, (dW_ana, db_ana) in enumerate(grads):
        layer = mlp.layers[l]

        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                orig = layer.W[i, j]
                layer.W[i, j] = orig + eps
                lp = mse(y, mlp.forward(X))
                layer.W[i, j] = orig - eps
                lm = mse(y, mlp.forward(X))
                layer.W[i, j] = orig
                num = (lp - lm) / (2 * eps)
                ana = dW_ana[i, j]
                rel = abs(ana - num) / (abs(ana) + abs(num) + 1e-10)
                max_rel = max(max_rel, rel)

        for j in range(layer.b.shape[0]):
            orig = layer.b[j]
            layer.b[j] = orig + eps
            lp = mse(y, mlp.forward(X))
            layer.b[j] = orig - eps
            lm = mse(y, mlp.forward(X))
            layer.b[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = db_ana[j]
            rel = abs(ana - num) / (abs(ana) + abs(num) + 1e-10)
            max_rel = max(max_rel, rel)

    assert max_rel < 1e-5, f"Gradient check failed: max_rel_error={max_rel:.2e}"


# ------------------------------------------------------------------
# 5.10 Save / load round-trip
# ------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    mlp, X, y = train_xor([2, 2, 1], epochs=100)

    path = str(tmp_path / "model.npz")
    mlp.save(path)

    mlp2 = MLP.load(path)
    np.testing.assert_array_equal(mlp.forward(X), mlp2.forward(X))


# ------------------------------------------------------------------
# 5.2 ValueError on bad combination
# ------------------------------------------------------------------

def test_invalid_combo_raises():
    with pytest.raises(ValueError):
        MLP([2, 2, 1], activation="tanh", output_activation="softmax", loss="mse")


# ------------------------------------------------------------------
# 6.6 XOR convergence [2, 2, 1]
# ------------------------------------------------------------------

def test_xor_2_2_1():
    mlp, X, y = train_xor([2, 2, 1], lr=0.5)
    pred = mlp.forward(X)
    assert np.all(np.sign(pred.ravel()) == y.ravel()), (
        f"XOR not solved: pred={pred.ravel()}, expected={y.ravel()}"
    )


# ------------------------------------------------------------------
# 6.7 XOR convergence [2, 3, 2, 1]
# ------------------------------------------------------------------

def test_xor_2_3_2_1():
    mlp, X, y = train_xor([2, 3, 2, 1], lr=0.5)
    pred = mlp.forward(X)
    assert np.all(np.sign(pred.ravel()) == y.ravel()), (
        f"XOR [2,3,2,1] not solved: pred={pred.ravel()}"
    )


# ------------------------------------------------------------------
# 6.8 Reproducibility
# ------------------------------------------------------------------

def test_reproducibility():
    X, y = xor_data()

    def train(seed):
        mlp = MLP([2, 3, 1], activation="tanh", output_activation="tanh",
                  loss="mse", seed=seed)
        mlp.fit(X, y, epochs=50, batch_size=4,
                optimizer=SGD(lr=0.1), verbose=False)
        return [e["loss_train"] for e in mlp.history]

    h1 = train(42)
    h2 = train(42)
    assert h1 == h2, "Same seed must produce identical training history"


# ------------------------------------------------------------------
# 7.4 Metrics tracker integration
# ------------------------------------------------------------------

def test_metrics_tracker(tmp_path):
    import pandas as pd
    from common.metrics import MetricsTracker

    X, y = xor_data()
    tracker = MetricsTracker(
        run_id="test_run",
        config_meta={"arch": "2-2-1", "optimizer": "sgd", "batch_size": 4},
    )
    mlp = MLP([2, 2, 1], activation="tanh", output_activation="tanh",
              loss="mse", seed=42)
    mlp.fit(X, y, epochs=5, batch_size=4,
            optimizer=SGD(lr=0.1), verbose=False, tracker=tracker)

    csv_path = str(tmp_path / "metrics.csv")
    tracker.export_csv(csv_path)
    df = pd.read_csv(csv_path)

    assert len(df) == 5
    expected = {"epoch", "loss_train", "acc_train", "loss_val", "acc_val", "time_s", "lr"}
    assert expected.issubset(set(df.columns))
