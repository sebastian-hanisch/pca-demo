"""Orakel-Test: die angezeigten Kennzahlen (k90, k90 ohne Krümmung, Varianzanteil, Trustworthiness, PC1-Drehwinkel, Verdict) der Presets und der
Krümmungs-Sweep über einen zweiten Rechenweg (scikit-learn-PCA und -Trustworthiness, eigene Winkelformel), dazu Trustworthiness aus der
Definition mit expliziten Rangtabellen."""

import numpy as np
import pytest

import pca_constants as C
from pca_evaluation import analyse, curvature_sweep, k_for_variance, trustworthiness, verdict
from pca_scenario import generate_dataset

decomposition = pytest.importorskip("sklearn.decomposition")
manifold = pytest.importorskip("sklearn.manifold")


def _sk(ds, std):
    X = ds.X
    Z = (X - X.mean(0)) / (X.std(0, ddof=1) if std else 1.0)
    pca = decomposition.PCA(svd_solver="full").fit(Z)
    ratio = pca.explained_variance_ratio_
    k90 = next(i + 1 for i, c in enumerate(np.cumsum(ratio)) if c >= 0.9 - 1e-12)
    trust = manifold.trustworthiness(Z, pca.transform(Z)[:, :2], n_neighbors=C.TRUST_NEIGHBORS)
    return ratio, k90, trust, pca


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_numbers_equal_the_sklearn_path(name):
    p = C.PRESETS[name]
    std = p["scaling"] == "standardized"
    ds = generate_dataset(p["n_tours"], p["q"], p["curvature"], p["noise"], p["outlier_pct"], p["seed"])
    a = analyse(ds, std, p["seed"])
    ratio, k90, trust, pca = _sk(ds, std)
    assert a.k90 == k90 and abs(a.trust_2d - trust) < 1e-9
    assert abs(a.ratio_q - ratio[:ds.q].sum()) < 1e-9 and abs(a.pc1_ratio - ratio[0]) < 1e-9
    flat = generate_dataset(ds.n, ds.q, 0.0, ds.noise, ds.outlier_pct, p["seed"])
    assert a.k90_flat == _sk(flat, std)[1]
    shares = pca.components_[0] ** 2
    assert a.dominant_feature == int(shares.argmax()) and abs(a.dominant_share - shares.max()) < 1e-9
    angle = 0.0
    if ds.outlier_pct:
        clean = _sk(generate_dataset(ds.n, ds.q, ds.curvature, ds.noise, 0, p["seed"]), std)[3]
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(pca.components_[0] @ clean.components_[0]))))))
    assert abs(a.pc1_angle_clean - angle) < 1e-6
    expected = "linear_ok"
    if not std and shares.max() >= C.UNITS_DOMINANCE:
        expected = "units"
    elif ds.curvature > 0 and k90 - a.k90_flat >= C.CURVATURE_EXTRA_COMPONENTS:
        expected = "curvature"
    elif ds.outlier_pct > 0 and angle >= C.OUTLIER_ANGLE_DEG:
        expected = "outliers"
    elif a.k90_flat - ds.q >= C.NOISE_EXTRA_COMPONENTS:
        expected = "noise"
    assert verdict(a, ds, std)[1] == expected == C.PRESET_EXPECTED_BANDS[name]["verdict"]


def test_curvature_sweep_equals_the_sklearn_path():
    rows = curvature_sweep(2, C.DEFAULT_NOISE, True, curvatures=(0.0, 0.6, 1.0), seeds=C.SWEEP_SEEDS[:3])
    for row in rows:
        k90s, rq, tr = [], [], []
        for sd in C.SWEEP_SEEDS[:3]:
            ratio, k90, trust, _ = _sk(generate_dataset(C.SWEEP_N_TOURS, 2, row["curvature"], C.DEFAULT_NOISE, 0, sd), True)
            k90s.append(k90)
            rq.append(ratio[:2].sum())
            tr.append(trust)
        assert abs(row["k90"] - np.mean(k90s)) < 1e-9 and abs(row["ratio_q"] - np.mean(rq)) < 1e-9 and abs(row["trust"] - np.mean(tr)) < 1e-9


def test_trustworthiness_equals_the_definition_with_explicit_rank_tables():
    rng = np.random.default_rng(3)
    for t in range(12):
        n, d, k = int(rng.integers(25, 70)), int(rng.integers(3, 8)), int(rng.choice([3, 5]))
        high = rng.standard_normal((n, d))
        low = high[:, :2] + (0.4 if t % 2 else 2.0) * rng.standard_normal((n, 2))
        total = 0
        for i in range(n):
            order_h = [j for j in np.argsort(np.linalg.norm(high - high[i], axis=1), kind="stable") if j != i]
            order_l = [j for j in np.argsort(np.linalg.norm(low - low[i], axis=1), kind="stable") if j != i]
            rank = {j: r + 1 for r, j in enumerate(order_h)}
            total += sum(rank[j] - k for j in order_l[:k] if rank[j] > k)
        assert abs(trustworthiness(high, low, k) - (1 - 2 / (n * k * (2 * n - 3 * k - 1)) * total)) < 1e-12


def test_k_for_variance_boundaries():
    assert k_for_variance(np.array([0.9, 0.1])) == 1
    assert k_for_variance(np.array([0.45, 0.45, 0.1])) == 2
    assert k_for_variance(np.array([1.0])) == 1
    assert k_for_variance(np.array([0.3, 0.3, 0.3, 0.1])) == 3
