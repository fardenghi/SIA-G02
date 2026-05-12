import numpy as np
import pandas as pd
import pytest

from pca_test.pca import run_pca
from pca_test.standardize import standardize


DATA_PATH = "data/europe.csv"


@pytest.fixture(scope="module")
def europe_df():
    return pd.read_csv(DATA_PATH)


# --- standardize ---

def test_standardize_preserves_columns(europe_df):
    out = standardize(europe_df)
    assert list(out.columns) == list(europe_df.columns)


def test_standardize_preserves_row_count(europe_df):
    out = standardize(europe_df)
    assert len(out) == len(europe_df)


def test_standardize_numeric_mean_zero(europe_df):
    out = standardize(europe_df)
    numeric = out.select_dtypes(include=[np.number])
    np.testing.assert_allclose(numeric.mean(axis=0).to_numpy(), 0, atol=1e-10)


def test_standardize_numeric_std_one(europe_df):
    out = standardize(europe_df)
    numeric = out.select_dtypes(include=[np.number])
    np.testing.assert_allclose(numeric.std(axis=0, ddof=0).to_numpy(), 1, atol=1e-10)


def test_standardize_keeps_country_column(europe_df):
    out = standardize(europe_df)
    assert out["Country"].tolist() == europe_df["Country"].tolist()


# --- run_pca ---

def test_run_pca_shapes(europe_df):
    pca, components, countries = run_pca(europe_df)
    n_samples = len(europe_df)
    n_features = europe_df.select_dtypes(include=[np.number]).shape[1]
    assert components.shape == (n_samples, n_features)
    assert pca.components_.shape == (n_features, n_features)
    assert len(countries) == n_samples


def test_run_pca_explained_variance_sums_to_one(europe_df):
    pca, _, _ = run_pca(europe_df)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)


def test_run_pca_explained_variance_sorted_desc(europe_df):
    pca, _, _ = run_pca(europe_df)
    ev = pca.explained_variance_ratio_
    assert np.all(np.diff(ev) <= 1e-12)


def test_run_pca_components_are_orthonormal(europe_df):
    pca, _, _ = run_pca(europe_df)
    gram = pca.components_ @ pca.components_.T
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-10)
