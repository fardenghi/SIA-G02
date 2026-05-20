import numpy as np
import pytest
from hopfield.network import HopfieldNetwork


# --- helpers ---

def two_pattern_net():
    """Example from slides: xi1=(1,1,-1,-1), xi2=(-1,-1,1,1), N=4."""
    net = HopfieldNetwork()
    patterns = np.array([
        [ 1,  1, -1, -1],
        [-1, -1,  1,  1],
    ], dtype=float)
    net.train(patterns)
    return net, patterns


# --- weight matrix ---

def test_weights_symmetric():
    net, _ = two_pattern_net()
    np.testing.assert_array_almost_equal(net.weights, net.weights.T)


def test_weights_diagonal_zero():
    net, _ = two_pattern_net()
    np.testing.assert_array_equal(np.diag(net.weights), np.zeros(4))


def test_weights_match_slides_example():
    net, _ = two_pattern_net()
    W = net.weights
    assert W[0, 1] == pytest.approx(0.5)
    assert W[0, 2] == pytest.approx(-0.5)
    assert W[0, 3] == pytest.approx(-0.5)
    assert W[2, 3] == pytest.approx(0.5)


def test_weights_shape():
    net, _ = two_pattern_net()
    assert net.weights.shape == (4, 4)


# --- prediction ---

def test_convergence_to_stored_pattern():
    net, patterns = two_pattern_net()
    query = np.array([1., -1., -1., -1.])
    result, _ = net.predict(query)
    np.testing.assert_array_equal(result, patterns[0])


def test_stored_pattern_is_fixed_point():
    net, patterns = two_pattern_net()
    for p in patterns:
        result, history = net.predict(p.copy())
        np.testing.assert_array_equal(result, p)
        assert len(history) == 1  # converges in 0 iterations (already stable)


def test_history_starts_with_query():
    net, _ = two_pattern_net()
    query = np.array([1., -1., -1., -1.])
    _, history = net.predict(query)
    np.testing.assert_array_equal(history[0], query)


def test_history_ends_at_stable_state():
    net, _ = two_pattern_net()
    query = np.array([1., -1., -1., -1.])
    result, history = net.predict(query)
    np.testing.assert_array_equal(history[-1], result)


# --- energy ---

def test_energy_of_stored_pattern_finite():
    net, patterns = two_pattern_net()
    e = net.energy(patterns[0])
    assert np.isfinite(e)


def test_energy_non_increasing_during_convergence():
    net, _ = two_pattern_net()
    query = np.array([1., -1., -1., -1.])
    _, history = net.predict(query)
    energies = [net.energy(s) for s in history]
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-10


def test_stored_patterns_are_local_minima():
    """Stored patterns have lower energy than all single-bit-flip neighbors."""
    net, patterns = two_pattern_net()
    for p in patterns:
        e_p = net.energy(p)
        for i in range(len(p)):
            neighbor = p.copy()
            neighbor[i] *= -1
            assert net.energy(neighbor) >= e_p - 1e-10


# --- n property ---

def test_n_matches_pattern_dimension():
    net, _ = two_pattern_net()
    assert net.n == 4
