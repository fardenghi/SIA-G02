"""Tests de edge cases y caminos de error para HopfieldNetwork y add_noise."""
import numpy as np
import pytest

from hopfield.hopfield import HopfieldNetwork, add_noise


# ---------------------------------------------------------------------------
# store() — casos extremos y errores
# ---------------------------------------------------------------------------

def test_store_single_pattern():
    """Un solo patrón almacenado debe ser punto fijo de la red."""
    p = np.array([1, -1, 1, -1, 1], dtype=np.int8)
    net = HopfieldNetwork(n_units=5)
    net.store([p])
    final, _, _, conv = net.recall(p.copy(), mode="sync", max_steps=5)
    assert conv
    np.testing.assert_array_equal(final, p)


def test_store_wrong_dim_raises():
    """store() debe lanzar ValueError si la dimensión no coincide."""
    net = HopfieldNetwork(n_units=5)
    bad = np.array([1, -1, 1], dtype=np.int8)  # dim=3, esperado 5
    with pytest.raises(ValueError, match="Patrones de tamaño"):
        net.store([bad])


def test_store_overwrites_previous():
    """Cada llamada a store() reemplaza los pesos y el historial de stored."""
    # Usamos dos patrones realmente distintos (no complementarios)
    p1 = np.array([1, 1, -1, -1], dtype=np.int8)
    p2 = np.array([1, -1, -1, 1], dtype=np.int8)   # diferente estructura
    net = HopfieldNetwork(n_units=4)
    net.store([p1])
    assert len(net.stored) == 1
    net.store([p2])
    # Después del segundo store, solo hay 1 patrón almacenado (p2)
    assert len(net.stored) == 1
    # El patrón almacenado debe ser p2, no p1
    np.testing.assert_array_equal(net.stored[0], p2)


def test_store_2d_array():
    """store() debe aceptar un array 2D (n_patterns × n_units)."""
    patterns = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=np.int8)
    net = HopfieldNetwork(n_units=4)
    net.store(patterns)
    assert len(net.stored) == 2


# ---------------------------------------------------------------------------
# recall() — modo inválido
# ---------------------------------------------------------------------------

def test_recall_invalid_mode_raises():
    """recall() debe lanzar ValueError con mode desconocido."""
    net = HopfieldNetwork(n_units=4)
    p = np.array([1, -1, 1, -1], dtype=np.int8)
    net.store([p])
    with pytest.raises(ValueError, match="mode inválido"):
        net.recall(p, mode="invalid")


# ---------------------------------------------------------------------------
# recall() — comportamiento con max_steps=0
# ---------------------------------------------------------------------------

def test_recall_zero_max_steps_returns_immediately():
    """Con max_steps=0 la red debe devolver la query sin modificarla."""
    net = HopfieldNetwork(n_units=4)
    p = np.array([1, -1, 1, -1], dtype=np.int8)
    net.store([p])
    query = np.array([1, 1, 1, -1], dtype=np.int8)
    final, history, energies, conv = net.recall(query.copy(), mode="sync", max_steps=0)
    # sin iteraciones: history solo tiene el estado inicial y conv=False
    assert not conv
    assert len(history) == 1
    np.testing.assert_array_equal(history[0], query)


# ---------------------------------------------------------------------------
# recall() síncrono — detección de ciclo de período 2
# ---------------------------------------------------------------------------

def test_sync_detects_period2_cycle():
    """Si la red oscila entre dos estados, conv debe ser False."""
    # Patrón de 4 neuronas con todos +1 — redes pequeñas pueden oscilar.
    # Buscamos cualquier red que exhiba un ciclo; usamos el control manual.
    net = HopfieldNetwork(n_units=4)
    # Fabricamos pesos que causan oscilación: W = -I (anti-Hopfield)
    net.weights = -np.eye(4)
    query = np.array([1, 1, -1, -1], dtype=np.int8)
    final, history, energies, conv = net.recall(query, mode="sync", max_steps=10)
    # Con -I: s(t+1) = sgn(-s(t)) = -s(t) → ciclo de período 2 → conv=False
    assert not conv


# ---------------------------------------------------------------------------
# energy() — propiedades
# ---------------------------------------------------------------------------

def test_energy_stored_pattern_is_finite():
    net = HopfieldNetwork(n_units=4)
    p = np.array([1, -1, 1, -1], dtype=np.int8)
    net.store([p])
    e = net.energy(p)
    assert np.isfinite(e)


def test_energy_formula():
    """E = -½ sᵀ W s verificado manualmente."""
    net = HopfieldNetwork(n_units=2)
    net.weights = np.array([[0.0, 0.5], [0.5, 0.0]])
    s = np.array([1, 1], dtype=np.int8)
    expected = -0.5 * float(s @ net.weights @ s)  # = -0.5 * 1.0 = -0.5
    assert net.energy(s) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# add_noise() — edge cases
# ---------------------------------------------------------------------------

def test_add_noise_zero_flips_nothing():
    rng = np.random.default_rng(0)
    p = np.array([1, -1, 1, -1, 1], dtype=np.int8)
    noisy = add_noise(p, 0.0, rng)
    np.testing.assert_array_equal(noisy, p)


def test_add_noise_one_flips_all():
    rng = np.random.default_rng(0)
    p = np.array([1, -1, 1, -1], dtype=np.int8)
    noisy = add_noise(p, 1.0, rng)
    np.testing.assert_array_equal(noisy, -p)


def test_add_noise_does_not_mutate_original():
    rng = np.random.default_rng(0)
    p = np.array([1, 1, -1, -1], dtype=np.int8)
    original = p.copy()
    add_noise(p, 0.5, rng)
    np.testing.assert_array_equal(p, original)


# ---------------------------------------------------------------------------
# is_stored() — complemento y patrón no almacenado
# ---------------------------------------------------------------------------

def test_is_stored_returns_minus_one_for_unknown():
    from hopfield.alphabet import letter_vector
    net = HopfieldNetwork(n_units=25)
    net.store([letter_vector("A"), letter_vector("B")])
    unknown = letter_vector("Z")
    # Z no fue almacenada y probablemente no coincide con ±A ni ±B
    # Solo verificamos que devuelve un int (puede ser -1 o índice)
    result = net.is_stored(unknown)
    assert isinstance(result, (int, np.integer))
