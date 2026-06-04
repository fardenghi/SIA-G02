import numpy as np

from autoencoder import data

FONT_PATH = "font/font.h"


def test_load_font_shape_and_values():
    X = data.load_font(FONT_PATH)
    assert X.shape == (32, 35)
    assert set(np.unique(X)).issubset({0.0, 1.0})


def test_unpack_full_byte():
    assert data.unpack_bits(0x1F).tolist() == [1, 1, 1, 1, 1]


def test_unpack_known_patterns():
    assert data.unpack_bits(0x10).tolist() == [1, 0, 0, 0, 0]
    assert data.unpack_bits(0x01).tolist() == [0, 0, 0, 0, 1]
    assert data.unpack_bits(0x0A).tolist() == [0, 1, 0, 1, 0]


def test_subset_none_returns_all():
    X = data.load_font(FONT_PATH)
    assert data.select_subset(X, None).shape == (32, 35)


def test_subset_explicit():
    X = data.load_font(FONT_PATH)
    sub = data.select_subset(X, [0, 1, 2])
    assert sub.shape == (3, 35)
    np.testing.assert_array_equal(sub, X[[0, 1, 2]])


def test_to_grid_shape():
    X = data.load_font(FONT_PATH)
    assert data.to_grid(X[0]).shape == (7, 5)


def test_noise_level_zero_is_identity():
    X = data.load_font(FONT_PATH)
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(data.add_noise(X, "salt_pepper", 0.0, rng), X)


def test_noise_changes_some_pixels():
    X = data.load_font(FONT_PATH)
    rng = np.random.default_rng(0)
    noisy = data.add_noise(X, "bit_flip", 0.2, rng)
    assert noisy.shape == X.shape
    assert not np.array_equal(noisy, X)
    assert set(np.unique(noisy)).issubset({0.0, 1.0})
