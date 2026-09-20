"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert)."""

import pytest

import pca_constants as C
from pca_evaluation import analyse, verdict
from pca_scenario import generate_dataset


def _measure(p):
    dataset = generate_dataset(p["n_tours"], p["q"], p["curvature"], p["noise"], p["outlier_pct"], p["seed"])
    standardize = p["scaling"] == "standardized"
    a = analyse(dataset, standardize, p["seed"])
    code = verdict(a, dataset, standardize)[1]
    return {"verdict": code, "k90": a.k90, "k90_flat": a.k90_flat, "ratio_q": a.ratio_q, "trust": a.trust_2d, "pc1_ratio": a.pc1_ratio,
            "dominant_share": a.dominant_share, "dominant": C.FEATURE_NAMES[a.dominant_feature], "angle": a.pc1_angle_clean}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n_tours"] <= C.N_TOURS_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX
        assert C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX
        assert C.OUTLIER_PCT_MIN <= p["outlier_pct"] <= C.OUTLIER_PCT_MAX and p["scaling"] in C.SCALINGS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
