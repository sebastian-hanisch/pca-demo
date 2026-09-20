import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

import pca_constants as C
from pca_algorithm import fit_pca
from pca_evaluation import analyse, curvature_sweep, k_for_variance, trustworthiness, verdict
from pca_scenario import generate_dataset, loading_matrix, n_nonlinear_terms

FIXED = C.SWEEP_SEEDS


def _k90(q, kappa, noise, standardize=True, n=250, seeds=FIXED):
    return float(np.mean([
        k_for_variance(fit_pca(generate_dataset(n, q, kappa, noise, 0, s).X, standardize).explained_variance_ratio) for s in seeds
    ]))


# --- Generator ------------------------------------------------------------------------------------------------------------

def test_dataset_shapes_features_and_determinism():
    ds = generate_dataset(150, 3, 0.3, 0.2, 5, 42)
    assert ds.X.shape == (150, 12) and ds.z.shape == (150, 3) and ds.outlier.shape == (150,)
    assert len(C.FEATURES) == 12 and len(C.FEATURE_LABELS) == 12
    again = generate_dataset(150, 3, 0.3, 0.2, 5, 42)
    assert np.array_equal(ds.X, again.X) and not np.array_equal(ds.X, generate_dataset(150, 3, 0.3, 0.2, 5, 43).X)


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_linear_low_noise_data_has_exactly_q_significant_components(q):
    assert _k90(q, 0.0, 0.1) == q
    model = fit_pca(generate_dataset(300, q, 0.0, 0.05, 0, 3).X, True)
    assert model.explained_variance_ratio[:q].sum() > 0.99                            # der Rest ist Rauschen


def test_curvature_raises_k90_and_lowers_the_variance_of_the_true_components():
    ks = [_k90(2, kappa, 0.25) for kappa in (0.0, 0.5, 1.0)]
    assert ks[0] < ks[1] <= ks[2] and ks[2] >= ks[0] + 2
    ratio = lambda kappa: np.mean([fit_pca(generate_dataset(250, 2, kappa, 0.25, 0, s).X, True).explained_variance_ratio[:2].sum() for s in FIXED])
    assert ratio(0.0) > ratio(0.5) > ratio(1.0)


def test_noise_raises_k90():
    assert _k90(2, 0.0, 0.1) < _k90(2, 0.0, 0.6) <= _k90(2, 0.0, 1.0)


def test_curved_data_has_true_dimension_q_by_construction():
    """Die Merkmale sind eine glatte Funktion von q latenten Koordinaten - Nachbarn im Faktorraum bleiben im Merkmalsraum Nachbarn."""
    ds = generate_dataset(300, 2, 1.0, 0.0, 0, 9)
    assert loading_matrix(2).shape == (12, 2) and n_nonlinear_terms(2) == 7
    Zs = (ds.X - ds.X.mean(0)) / ds.X.std(0, ddof=1)
    assert trustworthiness(ds.z, Zs, 10) > 0.97


def test_raw_data_is_dominated_by_the_metre_scale_feature_and_standardizing_fixes_it():
    ds = generate_dataset(300, 2, 0.0, 0.25, 0, 7)
    raw = fit_pca(ds.X, False)
    assert raw.explained_variance_ratio[0] > 0.999 and np.abs(raw.components[0]).argmax() == 0       # Distanz in Metern
    std = fit_pca(ds.X, True)
    assert std.explained_variance_ratio[0] < 0.7 and (std.components[0] ** 2).max() < 0.3


def test_outliers_rotate_pc1_and_the_same_tours_without_outliers_are_reproduced():
    clean = generate_dataset(300, 2, 0.0, 0.25, 0, 7)
    dirty = generate_dataset(300, 2, 0.0, 0.25, 10, 7)
    assert dirty.outlier.sum() > 0 and clean.outlier.sum() == 0
    keep = ~dirty.outlier
    assert np.allclose(dirty.X[keep], clean.X[keep])                                 # gleiche Touren, nur die Sonderfahrten unterscheiden sich
    a = analyse(dirty, True, 7)
    assert a.pc1_angle_clean > 8.0


# --- Auswertung -----------------------------------------------------------------------------------------------------------

def test_k_for_variance_examples():
    assert k_for_variance(np.array([0.5, 0.3, 0.1, 0.1])) == 3
    assert k_for_variance(np.array([0.95, 0.05])) == 1
    assert k_for_variance(np.array([0.3, 0.3, 0.2, 0.2])) == 4
    assert k_for_variance(np.array([0.3, 0.3, 0.3, 0.1])) == 3                       # exakt 90 % zählt als erreicht


def test_trustworthiness_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    for embedding in (X[:, :2], rng.standard_normal((80, 2)), fit_pca(X, False).transform(X, 2)):
        assert abs(trustworthiness(X, embedding, 8) - sk_trustworthiness(X, embedding, n_neighbors=8)) < 1e-9


def test_trustworthiness_is_one_for_the_identity_embedding_and_low_for_random():
    X = np.random.default_rng(1).standard_normal((100, 4))
    assert trustworthiness(X, X, 10) > 0.999
    assert trustworthiness(X, np.random.default_rng(2).standard_normal((100, 2)), 10) < 0.7


def test_curvature_sweep_is_monotone_in_the_mean_and_deterministic():
    rows = curvature_sweep(2, 0.25, True, n_tours=200)
    assert rows == curvature_sweep(2, 0.25, True, n_tours=200)
    assert rows[0]["k90"] <= rows[-1]["k90"] - 2
    assert rows[0]["ratio_q"] > rows[-1]["ratio_q"] + 0.2 and rows[0]["trust"] > rows[-1]["trust"]
    assert C.SWEEP_SEEDS[0] >= 100_000 > C.DEFAULT_SEED


def test_verdict_cascade_codes():
    def code(q, kappa, noise, out, standardize, seed=7):
        ds = generate_dataset(300, q, kappa, noise, out, seed)
        return verdict(analyse(ds, standardize, seed), ds, standardize)[1]
    assert code(2, 0.0, 0.25, 0, True) == "linear_ok"
    assert code(2, 0.0, 0.25, 0, False) == "units"
    assert code(2, 1.0, 0.25, 0, True) == "curvature"
    assert code(2, 0.0, 0.25, 10, True) == "outliers"
    assert code(2, 0.0, 0.6, 0, True) == "noise"
    assert code(4, 0.0, 0.25, 0, True) == "linear_ok"
    assert code(2, 1.0, 0.25, 0, False) == "units"                                     # Einheiten-Falle schlägt Krümmung (Warnungen nach Vorrang)
