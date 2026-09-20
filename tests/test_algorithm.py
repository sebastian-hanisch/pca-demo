import numpy as np
import pytest
from sklearn.decomposition import PCA

from pca_algorithm import (
    direction, fit_pca, optimal_angle, residual_along_direction, variance_along_direction,
)
from pca_scenario import generate_dataset


def _data(seed=0, n=120, d=6):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, 3))
    return latent @ rng.standard_normal((3, d)) * np.array([1, 5, 30, 0.2, 2, 100][:d]) + rng.standard_normal((n, d)) * 0.3


@pytest.mark.parametrize("standardize", [False, True])
def test_matches_sklearn_pca(standardize):
    X = _data()
    model = fit_pca(X, standardize)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1) if standardize else X
    reference = PCA().fit(Xs)
    assert np.allclose(model.explained_variance, reference.explained_variance_, rtol=1e-9)
    assert np.allclose(model.explained_variance_ratio, reference.explained_variance_ratio_, rtol=1e-9)
    for mine, theirs in zip(model.components, reference.components_):
        assert np.allclose(mine, theirs, atol=1e-8) or np.allclose(mine, -theirs, atol=1e-8)


def test_components_are_orthonormal_and_variances_sorted_and_sum_to_total():
    X = _data(1)
    model = fit_pca(X, False)
    assert np.allclose(model.components @ model.components.T, np.eye(6), atol=1e-10)
    assert (np.diff(model.explained_variance) <= 1e-12).all()
    assert abs(model.explained_variance.sum() - X.var(0, ddof=1).sum()) < 1e-8


def test_signs_are_deterministic_largest_loading_positive():
    model = fit_pca(_data(2), True)
    for comp in model.components:
        assert comp[np.abs(comp).argmax()] > 0
    assert np.array_equal(model.components, fit_pca(_data(2), True).components)


def test_reconstruction_error_equals_sum_of_discarded_eigenvalues_and_full_rank_is_exact():
    X = _data(3, n=200)
    model = fit_pca(X, False)
    n = len(X)
    for k in range(1, 7):
        expected = (n - 1) / n * model.explained_variance[k:].sum()
        assert abs(model.reconstruction_error(X, k) - expected) < 1e-8
    assert np.allclose(model.reconstruct(X, 6), X, atol=1e-8)
    assert model.reconstruction_error(X, 6) < 1e-15


def test_reconstruction_is_the_best_linear_rank_k_approximation():
    """Eckart-Young: keine zufällige k-dimensionale Projektion hat einen kleineren Fehler."""
    X = _data(4)
    model = fit_pca(X, False)
    Xc = X - X.mean(0)
    rng = np.random.default_rng(0)
    best = model.reconstruction_error(X, 2)
    for _ in range(200):
        q, _ = np.linalg.qr(rng.standard_normal((6, 2)))
        assert ((Xc - Xc @ q @ q.T) ** 2).sum(1).mean() >= best - 1e-9


def test_standardization_makes_the_result_unit_independent():
    X = _data(5)
    scaled = X * np.array([1000.0, 0.001, 5.0, 1.0, 250.0, 0.1])
    a, b = fit_pca(X, True), fit_pca(scaled, True)
    assert np.allclose(a.explained_variance_ratio, b.explained_variance_ratio)
    assert np.allclose(np.abs(a.transform(X)), np.abs(b.transform(scaled)), atol=1e-8)
    raw_a, raw_b = fit_pca(X, False), fit_pca(scaled, False)
    assert not np.allclose(raw_a.explained_variance_ratio, raw_b.explained_variance_ratio)      # ohne Standardisierung nicht


def test_transform_inverse_roundtrip_shapes():
    X = _data(6)
    model = fit_pca(X, True)
    scores = model.transform(X, 3)
    assert scores.shape == (len(X), 3) and model.inverse_transform(scores).shape == X.shape


# --- Richtungs-Abschnitt ---------------------------------------------------------------------------------------------------

def test_variance_curve_maximum_is_lambda_one_at_pc1_and_minimum_is_lambda_two_perpendicular():
    rng = np.random.default_rng(0)
    Z2 = rng.standard_normal((300, 2)) @ np.array([[2.0, 0.6], [0.0, 0.7]])
    Z2 = Z2 - Z2.mean(0)
    angle, lam1, lam2 = optimal_angle(Z2)
    angles = np.arange(0, 180, 0.25)
    variances = np.array([variance_along_direction(Z2, a) for a in angles])
    assert abs(variances.max() - lam1) < 1e-3 and abs(variances.min() - lam2) < 1e-3
    assert abs(angles[variances.argmax()] - angle) < 0.5 or abs(abs(angles[variances.argmax()] - angle) - 180) < 0.5
    assert abs(variance_along_direction(Z2, angle) - lam1) < 1e-9
    assert abs(variance_along_direction(Z2, (angle + 90) % 180) - lam2) < 1e-9


def test_variance_plus_residual_equals_total_variance_for_every_angle():
    rng = np.random.default_rng(1)
    Z2 = rng.standard_normal((200, 2)) * [3.0, 1.0]
    Z2 = Z2 - Z2.mean(0)
    total = Z2.var(0, ddof=1).sum()
    n = len(Z2)
    for a in (0, 17, 45, 90, 133, 179):
        assert abs(variance_along_direction(Z2, a) * (n - 1) / n + residual_along_direction(Z2, a) - total * (n - 1) / n) < 1e-9


def test_direction_is_a_unit_vector():
    for a in (0, 33, 90, 179):
        assert abs(np.linalg.norm(direction(a)) - 1) < 1e-12


def test_pca_on_generated_data_matches_sklearn():
    ds = generate_dataset(200, 3, 0.4, 0.3, 0, 5)
    model = fit_pca(ds.X, True)
    reference = PCA().fit((ds.X - ds.X.mean(0)) / ds.X.std(0, ddof=1))
    assert np.allclose(model.explained_variance_ratio, reference.explained_variance_ratio_)
