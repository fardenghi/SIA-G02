import numpy as np
import pytest

from oja.oja import OjaNetwork


def _make_net(dim=5, lr=0.5, epochs=20, seed=42):
    return OjaNetwork(input_dim=dim, lr=lr, epochs=epochs, seed=seed)


# --- initialization ---

def test_weights_shape():
    net = _make_net(dim=7)
    assert net.weights.shape == (7,)


def test_weights_normalized_at_init():
    net = _make_net(dim=4)
    assert np.linalg.norm(net.weights) == pytest.approx(1.0)


def test_weights_reproducible_with_seed():
    n1 = _make_net(seed=11)
    n2 = _make_net(seed=11)
    np.testing.assert_array_equal(n1.weights, n2.weights)


def test_weights_differ_across_seeds():
    n1 = _make_net(seed=1)
    n2 = _make_net(seed=2)
    assert not np.allclose(n1.weights, n2.weights)


def test_history_starts_with_initial_weights():
    net = _make_net(dim=3)
    assert len(net.history) == 1
    np.testing.assert_array_equal(net.history[0], net.weights)


# --- training ---

def test_history_records_each_epoch():
    net = _make_net(dim=4, epochs=10)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    net.train(X)
    assert len(net.history) == 11  # initial + epochs


def test_train_does_not_explode():
    net = _make_net(dim=4, epochs=20)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    net.train(X)
    assert np.all(np.isfinite(net.weights))


def test_component_is_unit_norm():
    net = _make_net(dim=4, epochs=20)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    net.train(X)
    assert np.linalg.norm(net.component()) == pytest.approx(1.0)


# --- convergence vs sklearn PC1 ---

def test_converges_to_first_principal_component():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    # build data where PC1 is clearly dominant
    n = 200
    pc1 = rng.standard_normal(n) * 3
    pc2 = rng.standard_normal(n) * 0.2
    X = np.column_stack([pc1, pc2, pc1 * 0.5 + pc2 * 0.1])
    X = X - X.mean(axis=0)

    sk = PCA(n_components=1).fit(X)
    sk_w = sk.components_[0]

    net = OjaNetwork(input_dim=X.shape[1], lr=0.1, epochs=100, seed=0)
    net.train(X)
    w = net.component()
    if np.dot(w, sk_w) < 0:
        w = -w
    # cosine similarity should be close to 1
    assert float(np.dot(w, sk_w)) > 0.95


# --- predict ---

def test_predict_returns_scalar_per_sample():
    net = _make_net(dim=4, epochs=5)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    net.train(X)
    scores = net.predict(X)
    assert scores.shape == (10,)
