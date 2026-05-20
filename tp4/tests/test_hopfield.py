import numpy as np
import pytest

from hopfield.alphabet import ALPHABET, LETTERS, letter_vector, letters_in_range
from hopfield.hopfield import HopfieldNetwork, add_noise


# --- alphabet ---

def test_alphabet_has_26_letters():
    assert len(ALPHABET) == 26
    assert LETTERS == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_each_letter_is_5x5_pm1():
    for ch, m in ALPHABET.items():
        assert m.shape == (5, 5), ch
        assert set(np.unique(m).tolist()).issubset({-1, 1}), ch


def test_letter_vector_size_25():
    assert letter_vector("A").shape == (25,)


def test_letters_in_range():
    assert letters_in_range("c", "h") == ["C", "D", "E", "F", "G", "H"]
    assert letters_in_range("h", "c") == ["C", "D", "E", "F", "G", "H"]
    assert letters_in_range("a", "a") == ["A"]


# --- Hopfield ---

def _build_net(letters: list[str]) -> tuple[HopfieldNetwork, np.ndarray]:
    P = np.stack([letter_vector(c) for c in letters])
    net = HopfieldNetwork(n_units=25)
    net.store(P)
    return net, P


def test_weights_symmetric_zero_diagonal():
    net, _ = _build_net(["A", "J"])
    assert np.allclose(net.weights, net.weights.T)
    assert np.allclose(np.diag(net.weights), 0.0)


def test_stored_pattern_is_fixed_point():
    net, P = _build_net(["G", "R", "T", "V"])
    for p in P:
        final, _h, _e, conv = net.recall(p, mode="sync", max_steps=10)
        assert conv
        np.testing.assert_array_equal(final, p)


def test_recall_recovers_from_small_noise():
    rng = np.random.default_rng(0)
    net, P = _build_net(["G", "R", "T", "V"])
    successes = 0
    trials = 0
    for p in P:
        for _ in range(10):
            noisy = add_noise(p, 0.12, rng)
            final, _h, _e, _c = net.recall(noisy, mode="sync", max_steps=30)
            trials += 1
            if np.array_equal(final, p):
                successes += 1
    assert successes / trials >= 0.7


def test_async_recall_converges_for_stored_pattern():
    rng = np.random.default_rng(0)
    net, P = _build_net(["A", "B", "C"])
    final, _h, _e, conv = net.recall(P[0], mode="async", max_steps=10, rng=rng)
    assert conv
    np.testing.assert_array_equal(final, P[0])


def test_energy_decreases_monotonically_in_async():
    rng = np.random.default_rng(0)
    net, P = _build_net(["A", "B", "C"])
    noisy = add_noise(P[1], 0.2, rng)
    _f, _h, energies, _c = net.recall(noisy, mode="async", max_steps=20, rng=rng)
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1e-9


def test_is_stored_detects_stored_pattern():
    net, P = _build_net(["A", "B"])
    assert net.is_stored(P[0]) == 0
    assert net.is_stored(P[1]) == 1
    flipped = -P[0]
    assert net.is_stored(flipped) == 0  # complemento también es atractor


def test_add_noise_flips_expected_fraction():
    rng = np.random.default_rng(0)
    p = letter_vector("A")
    noisy = add_noise(p, 0.3, rng)
    flipped = int(np.sum(p != noisy))
    assert 0 < flipped < 25


# --- orthogonality ---

def test_best_k4_combo_has_low_max_dot():
    from hopfield.orthogonality import pairwise_dot_matrix, rank_combinations
    dot = pairwise_dot_matrix()
    df = rank_combinations(4, dot)
    # The most orthogonal combo should have small max|dot| (<=5 for 5x5 patterns).
    assert int(df.iloc[0]["max_abs_dot"]) <= 5
    # Worst combo should have much larger max|dot|.
    assert int(df.iloc[-1]["max_abs_dot"]) >= 15
