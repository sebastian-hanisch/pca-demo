"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, geschützte Regler, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pca_constants as C

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None):
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def test_default_renders_without_exception():
    at = _run()
    assert any("Dimensionen" in h.value for h in at.subheader)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    p = C.PRESETS[name]

    def setup(at):
        at.session_state["n_tours_slider"] = p["n_tours"]
        at.session_state["q_slider"] = p["q"]
        at.session_state["curvature_slider"] = p["curvature"]
        at.session_state["noise_slider"] = p["noise"]
        at.session_state["outlier_slider"] = p["outlier_pct"]
        at.session_state["scaling_radio"] = p["scaling"]
        at.session_state["seed_input"] = p["seed"]
    _run(setup)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["q_slider"] = C.Q_MIN
        at.session_state["noise_slider"] = C.NOISE_MIN
    _run(small)

    def large(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MAX
        at.session_state["q_slider"] = C.Q_MAX
        at.session_state["curvature_slider"] = C.CURVATURE_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["outlier_slider"] = C.OUTLIER_PCT_MAX
        at.session_state["scaling_radio"] = "raw"
    _run(large)


def test_angle_and_reconstruction_state_reset_when_the_data_changes():
    at = _run()
    at.session_state["angle_slider"] = 95
    at.session_state["recon_k"] = 7
    at.run()
    assert at.session_state["angle_slider"] == 95 and at.session_state["recon_k"] == 7           # bleibt bei unveränderten Daten
    at.session_state["q_slider"] = 3
    at.run()
    assert not at.exception
    assert at.session_state["angle_slider"] == 0 and at.session_state["recon_k"] == 3            # neue Daten: Zustand zurückgesetzt


def test_feature_pair_never_collapses_to_the_same_feature():
    at = _run()
    at.session_state["feat_x"] = at.session_state["feat_y"]
    at.run()
    assert not at.exception
    assert at.session_state["feat_x"] != at.session_state["feat_y"]


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "pca_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == 7
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1                       # nur das return von lock_axes selbst
