import numpy as np
import pytest
from hopfield.network import HopfieldNetwork
from hopfield.patterns import LETTERS, GRID, as_vector, render, add_noise, identify


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


# --- patterns ---

def test_letter_shapes():
    for name, mat in LETTERS.items():
        assert mat.shape == (GRID, GRID), f"{name} has wrong shape"


def test_letter_values_bipolar():
    for name, mat in LETTERS.items():
        assert set(mat.flatten().tolist()) <= {1.0, -1.0}, f"{name} has non-bipolar values"


def test_as_vector_length():
    for name in LETTERS:
        v = as_vector(name)
        assert v.shape == (GRID * GRID,)


def test_render_output_has_correct_rows():
    v = as_vector("Z")
    rendered = render(v)
    assert len(rendered.split("\n")) == GRID


def test_add_noise_preserves_shape():
    rng = np.random.default_rng(0)
    v = as_vector("Z")
    noisy = add_noise(v, noise_level=0.2, rng=rng)
    assert noisy.shape == v.shape


def test_add_noise_values_bipolar():
    rng = np.random.default_rng(0)
    v = as_vector("Z")
    noisy = add_noise(v, noise_level=0.3, rng=rng)
    assert set(noisy.tolist()) <= {1.0, -1.0}


def test_add_noise_zero_level_unchanged():
    rng = np.random.default_rng(0)
    v = as_vector("E")
    noisy = add_noise(v, noise_level=0.0, rng=rng)
    np.testing.assert_array_equal(noisy, v)


def test_add_noise_full_level_inverts():
    rng = np.random.default_rng(0)
    v = as_vector("T")
    noisy = add_noise(v, noise_level=1.0, rng=rng)
    np.testing.assert_array_equal(noisy, -v)


def test_identify_returns_name_for_stored():
    stored = {name: as_vector(name) for name in LETTERS}
    for name in LETTERS:
        assert identify(as_vector(name), stored) == name


def test_identify_returns_none_for_unknown():
    stored = {name: as_vector(name) for name in LETTERS}
    unknown = np.ones(GRID * GRID)
    assert identify(unknown, stored) is None


# --- patterns as stored memories ---

def four_letter_net():
    net = HopfieldNetwork()
    patterns = np.array([as_vector(l) for l in ["Z", "E", "N", "T"]])
    net.train(patterns)
    stored = {l: as_vector(l) for l in ["Z", "E", "N", "T"]}
    return net, patterns, stored


def test_stored_letters_are_fixed_points():
    net, patterns, stored = four_letter_net()
    for p in patterns:
        result, history = net.predict(p.copy())
        np.testing.assert_array_equal(result, p)
        assert len(history) == 1


def test_low_noise_recovers_correct_letter():
    net, _, stored = four_letter_net()
    rng = np.random.default_rng(1)
    for name in ["Z", "E", "N", "T"]:
        v = as_vector(name)
        noisy = add_noise(v, noise_level=0.1, rng=rng)
        result, _ = net.predict(noisy)
        assert identify(result, stored) == name
