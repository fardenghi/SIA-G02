import numpy as np
import pytest
from hopfield.network import HopfieldNetwork
from hopfield.patterns import (
    ALL_LETTERS, LETTERS, GRID, as_vector, render, add_noise, identify,
    min_scale_factor, scale_pattern,
)


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


# --- spurious states ---

def test_spurious_state_is_stable():
    """A spurious state found from a noisy query must be a fixed point."""
    from hopfield_letters import find_spurious
    net, _, stored = four_letter_net()
    spurious, _ = find_spurious(
        net, stored,
        noise_level=0.5,
        n_trials=500,
        rng=np.random.default_rng(44),
    )
    if spurious is None:
        pytest.skip("No spurious state found with this seed")
    result, history = net.predict(spurious.copy())
    np.testing.assert_array_equal(result, spurious)
    assert len(history) == 1


def test_spurious_state_differs_from_all_stored():
    from hopfield_letters import find_spurious
    net, _, stored = four_letter_net()
    spurious, _ = find_spurious(
        net, stored,
        noise_level=0.5,
        n_trials=500,
        rng=np.random.default_rng(44),
    )
    if spurious is None:
        pytest.skip("No spurious state found with this seed")
    assert identify(spurious, stored) is None


def test_energy_non_increasing_letter_recovery():
    net, _, stored = four_letter_net()
    rng = np.random.default_rng(3)
    for name in ["Z", "N"]:
        v = stored[name]
        noisy = add_noise(v, noise_level=0.2, rng=rng)
        _, history = net.predict(noisy)
        energies = [net.energy(s) for s in history]
        for i in range(len(energies) - 1):
            assert energies[i + 1] <= energies[i] + 1e-10


# --- ALL_LETTERS (full alphabet) ---

def test_all_letters_count():
    assert len(ALL_LETTERS) == 26


def test_all_letters_covers_alphabet():
    assert set(ALL_LETTERS.keys()) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_all_letters_shapes():
    for name, mat in ALL_LETTERS.items():
        assert mat.shape == (GRID, GRID), f"{name} has wrong shape"


def test_all_letters_values_bipolar():
    for name, mat in ALL_LETTERS.items():
        assert set(mat.flatten().tolist()) <= {1.0, -1.0}, f"{name} has non-bipolar values"


def test_all_letters_consistent_with_letters():
    """ALL_LETTERS must agree with LETTERS for the 4 original patterns."""
    for name in LETTERS:
        np.testing.assert_array_equal(
            ALL_LETTERS[name], LETTERS[name],
            err_msg=f"ALL_LETTERS['{name}'] differs from LETTERS['{name}']",
        )


def test_all_letters_network_is_deterministic():
    """With 26 patterns (p/N ≈ 1.04 >> capacity), the network is overloaded.
    Stored patterns may not be fixed points (synchronous update can produce
    period-2 cycles), but predict() must be deterministic: same input → same output.
    """
    net = HopfieldNetwork()
    patterns = np.array([mat.flatten() for mat in ALL_LETTERS.values()])
    net.train(patterns)
    stored = {name: mat.flatten() for name, mat in ALL_LETTERS.items()}
    for name, v in stored.items():
        result1, _ = net.predict(v.copy())
        result2, _ = net.predict(v.copy())
        np.testing.assert_array_equal(result1, result2, err_msg=f"predict('{name}') is not deterministic")


def test_all_letters_energy_non_increasing():
    """Energy must be non-increasing during convergence for the full-alphabet network."""
    net = HopfieldNetwork()
    patterns = np.array([mat.flatten() for mat in ALL_LETTERS.values()])
    net.train(patterns)
    rng = np.random.default_rng(7)
    for name in ["A", "M", "Z"]:
        v = ALL_LETTERS[name].flatten()
        noisy = add_noise(v, noise_level=0.2, rng=rng)
        _, history = net.predict(noisy)
        energies = [net.energy(s) for s in history]
        for i in range(len(energies) - 1):
            assert energies[i + 1] <= energies[i] + 1e-10, f"Energy increased at step {i} for '{name}'"


def test_all_letters_capacity_degradation():
    """Recovery rate must drop significantly once p >> 0.138·N (N=25 → limit ≈ 3.5)."""
    N = GRID * GRID
    names = sorted(ALL_LETTERS.keys())
    rng = np.random.default_rng(99)
    noise = 0.1
    n_trials = 50

    def recovery_rate(letter_subset: list[str]) -> float:
        net = HopfieldNetwork()
        stored = {n: ALL_LETTERS[n].flatten() for n in letter_subset}
        net.train(np.array(list(stored.values())))
        correct = total = 0
        for name in letter_subset:
            v = stored[name]
            for _ in range(n_trials):
                result, _ = net.predict(add_noise(v, noise, rng))
                if identify(result, stored) == name:
                    correct += 1
                total += 1
        return correct / total

    rate_small = recovery_rate(names[:3])   # p=3, well within capacity
    rate_large = recovery_rate(names[:26])  # p=26, far beyond capacity
    assert rate_small > rate_large, (
        f"Expected degradation: small={rate_small:.2f} should > large={rate_large:.2f}"
    )


# --- scale_pattern and min_scale_factor ---

def test_scale_pattern_k1_identity():
    mat = ALL_LETTERS["Z"]
    np.testing.assert_array_equal(scale_pattern(mat, 1), mat)


def test_scale_pattern_shape():
    mat = ALL_LETTERS["A"]
    for k in (2, 3, 4):
        scaled = scale_pattern(mat, k)
        assert scaled.shape == (GRID * k, GRID * k), f"Wrong shape for k={k}"


def test_scale_pattern_values_bipolar():
    mat = ALL_LETTERS["E"]
    for k in (2, 3):
        scaled = scale_pattern(mat, k)
        assert set(scaled.flatten().tolist()) <= {1.0, -1.0}


def test_scale_pattern_block_structure():
    """Each pixel of the original must appear as a uniform k×k block."""
    mat = ALL_LETTERS["T"]
    k = 2
    scaled = scale_pattern(mat, k)
    for i in range(GRID):
        for j in range(GRID):
            block = scaled[i*k:(i+1)*k, j*k:(j+1)*k]
            assert np.all(block == mat[i, j]), f"Block ({i},{j}) not uniform"


@pytest.mark.parametrize("p,expected_k", [
    (1,  1),   # 1 pattern: k=1 (N=25, capacity≈3.45 ≥ 1) ✓
    (3,  1),   # 3 patterns: k=1 (capacity≈3.45 ≥ 3) ✓
    (4,  2),   # 4 patterns: k=1 gives N=25, cap=3.45 < 4 → need k=2
    (13, 2),   # 13 patterns: k=2 (N=100, cap≈13.8 ≥ 13) ✓
    (14, 3),   # 14 patterns: k=2 gives cap=13.8 < 14 → need k=3
    (26, 3),   # full alphabet: k=3 (N=225, cap≈31 ≥ 26) ✓
])
def test_min_scale_factor(p, expected_k):
    k = min_scale_factor(p)
    assert k == expected_k, f"p={p}: expected k={expected_k}, got k={k}"
    N = (GRID * k) ** 2
    assert N * 0.138 >= p, f"p={p}, k={k}: N={N} still below capacity"


def test_adaptive_alphabet_recovers_better_than_fixed():
    """Network with adaptive scale (k=3) must recover more letters than fixed k=1."""
    names = sorted(ALL_LETTERS.keys())
    k = min_scale_factor(len(names))
    rng = np.random.default_rng(42)
    noise = 0.1
    n_trials = 30

    def rate(scale: int) -> float:
        net = HopfieldNetwork()
        stored = {n: scale_pattern(ALL_LETTERS[n], scale).flatten() for n in names}
        net.train(np.array(list(stored.values())))
        correct = total = 0
        for name in names:
            v = stored[name]
            for _ in range(n_trials):
                if identify(net.predict(add_noise(v, noise, rng))[0], stored) == name:
                    correct += 1
                total += 1
        return correct / total

    r_fixed = rate(1)
    r_adaptive = rate(k)
    assert r_adaptive > r_fixed, (
        f"Adaptive (k={k}) rate {r_adaptive:.2f} should exceed fixed (k=1) rate {r_fixed:.2f}"
    )
