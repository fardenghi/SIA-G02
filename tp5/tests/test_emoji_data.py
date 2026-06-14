"""Tests del dataset de emojis (Fase 2).

Si la fuente de emojis no está instalada, los tests se omiten (skip) en vez de fallar, para
no romper la suite en entornos sin la fuente.
"""

from pathlib import Path

import numpy as np
import pytest

from autoencoder import emoji_data

_HAS_FONT = Path(emoji_data.DEFAULT_FONT).exists()
pytestmark = pytest.mark.skipif(
    not _HAS_FONT, reason=f"falta la fuente de emojis: {emoji_data.DEFAULT_FONT}")


def test_load_shape_and_range():
    X, labels = emoji_data.load_emojis(size=28)
    assert X.shape == (len(emoji_data.EMOJIS), 28 * 28)
    assert len(labels) == X.shape[0]
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_custom_size():
    X, _ = emoji_data.load_emojis(size=16)
    assert X.shape == (len(emoji_data.EMOJIS), 16 * 16)


def test_deterministic():
    X1, _ = emoji_data.load_emojis(size=28)
    X2, _ = emoji_data.load_emojis(size=28)
    np.testing.assert_array_equal(X1, X2)


def test_each_glyph_nonempty():
    X, _ = emoji_data.load_emojis(size=28)
    # Cada emoji deja tinta (no queda en blanco) tras el rasterizado.
    assert np.all(X.mean(axis=1) > 0.01)


def test_glyphs_are_distinct():
    X, _ = emoji_data.load_emojis(size=28)
    dups = sum(np.array_equal(X[i], X[j])
               for i in range(len(X)) for j in range(i + 1, len(X)))
    assert dups == 0


def test_subset_and_labels_aligned():
    X, labels = emoji_data.load_emojis(size=28, subset=[0, 5, 10])
    assert X.shape[0] == 3
    assert labels == [emoji_data.EMOJIS[i][1] for i in (0, 5, 10)]


def test_render_emoji_single():
    vec = emoji_data.render_emoji(emoji_data.EMOJIS[0][0], size=20)
    assert vec.shape == (20 * 20,)
    assert vec.min() >= 0.0 and vec.max() <= 1.0


def test_missing_font_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        emoji_data.load_emojis(font_path=tmp_path / "no_existe.ttf")


def test_augment_expands_and_is_deterministic():
    X, labels = emoji_data.load_emojis(size=28)
    Xa1, la1 = emoji_data.augment_dataset(
        X, labels, size=28, n_aug=2, rng=np.random.default_rng(0))
    # Original + 2 bloques aumentados.
    assert Xa1.shape == (X.shape[0] * 3, 28 * 28)
    assert len(la1) == X.shape[0] * 3
    assert Xa1.min() >= 0.0 and Xa1.max() <= 1.0
    # El primer bloque es el original sin tocar.
    np.testing.assert_array_equal(Xa1[:X.shape[0]], X)
    # Determinista dado el rng.
    Xa2, _ = emoji_data.augment_dataset(
        X, labels, size=28, n_aug=2, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(Xa1, Xa2)
