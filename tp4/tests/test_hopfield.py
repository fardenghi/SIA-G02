import numpy as np
import pytest

from hopfield.alphabet import (
    ALPHABET, GRID, LETTERS, letter_vector, letters_in_range,
    min_scale_factor, scale_pattern, scaled_letter_vector,
)
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


# --- regla de Hebb: ejemplo de las diapositivas (4 neuronas, 2 patrones) ---

def _slide_example_net() -> tuple[HopfieldNetwork, np.ndarray]:
    """xi1 = ( 1, 1,-1,-1), xi2 = (-1,-1, 1, 1), N=4."""
    patterns = np.array([
        [ 1,  1, -1, -1],
        [-1, -1,  1,  1],
    ], dtype=np.int8)
    net = HopfieldNetwork(n_units=4)
    net.store(patterns)
    return net, patterns


def test_slide_example_weights_symmetric():
    net, _ = _slide_example_net()
    np.testing.assert_array_almost_equal(net.weights, net.weights.T)


def test_slide_example_weights_diagonal_zero():
    net, _ = _slide_example_net()
    np.testing.assert_array_equal(np.diag(net.weights), np.zeros(4))


def test_slide_example_weights_match_textbook():
    net, _ = _slide_example_net()
    W = net.weights
    # Con N=4 y 2 patrones complementarios: w_ij = (xi1_i*xi1_j + xi2_i*xi2_j)/N
    # → w_ij = 0.5 si xi1_i*xi1_j = xi2_i*xi2_j = +1, -0.5 si = -1.
    assert W[0, 1] == pytest.approx(0.5)
    assert W[0, 2] == pytest.approx(-0.5)
    assert W[0, 3] == pytest.approx(-0.5)
    assert W[2, 3] == pytest.approx(0.5)


def test_slide_example_stored_is_fixed_point():
    net, patterns = _slide_example_net()
    for p in patterns:
        final, _h, _e, conv = net.recall(p.copy(), mode="sync", max_steps=5)
        assert conv
        np.testing.assert_array_equal(final, p)


def test_slide_example_recovers_close_query():
    net, patterns = _slide_example_net()
    query = np.array([1, -1, -1, -1], dtype=np.int8)
    final, _h, _e, _c = net.recall(query, mode="sync", max_steps=5)
    np.testing.assert_array_equal(final, patterns[0])


# --- scale_pattern / min_scale_factor ---

def test_scale_pattern_k1_identity():
    mat = ALPHABET["Z"]
    np.testing.assert_array_equal(scale_pattern(mat, 1), mat)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_scale_pattern_shape(k):
    mat = ALPHABET["A"]
    assert scale_pattern(mat, k).shape == (GRID * k, GRID * k)


def test_scale_pattern_values_bipolar():
    for k in (2, 3):
        scaled = scale_pattern(ALPHABET["E"], k)
        assert set(np.unique(scaled).tolist()).issubset({-1, 1})


def test_scale_pattern_block_structure():
    mat = ALPHABET["T"]
    k = 2
    scaled = scale_pattern(mat, k)
    for i in range(GRID):
        for j in range(GRID):
            block = scaled[i * k:(i + 1) * k, j * k:(j + 1) * k]
            assert np.all(block == mat[i, j])


@pytest.mark.parametrize("p,expected_k", [
    (1, 1), (3, 1), (4, 2), (13, 2), (14, 3), (26, 3),
])
def test_min_scale_factor(p, expected_k):
    k = min_scale_factor(p)
    assert k == expected_k
    N = (GRID * k) ** 2
    assert 0.138 * N >= p


def test_scaled_letter_vector_size():
    assert scaled_letter_vector("A", k=1).shape == (25,)
    assert scaled_letter_vector("A", k=3).shape == (225,)


def test_adaptive_alphabet_recovers_some_letters():
    """Con k=3 (N=225, capacity≈31) la red SÍ puede ser punto fijo para varios patrones,
    cosa imposible con N=25 (capacity≈3.45 << 26)."""
    k = min_scale_factor(26)
    patterns = np.stack([scaled_letter_vector(c, k) for c in LETTERS])
    net = HopfieldNetwork(n_units=patterns.shape[1])
    net.store(patterns)
    fixed_count = sum(
        np.array_equal(net.recall(p.copy(), mode="sync", max_steps=10)[0], p)
        for p in patterns
    )
    # con N=25 esto sería casi siempre 0; con N=225 esperamos >= 4.
    assert fixed_count >= 4
