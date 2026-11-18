"""Headless smoke tests for the Streamlit app (no browser needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
HAS_MODELS = (Path(APP).parent / "results" / "models.joblib").exists()


def _fresh_app() -> "AppTest":
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_with_five_tabs():
    at = _fresh_app()
    assert [t.label for t in at.tabs] == [
        "🏠 Overview", "▶ Simulator", "🔮 Halting prediction", "📊 Experiments", "📚 Theory notes"]
    assert not at.error


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_simulator_explains_each_step():
    at = _fresh_app()  # default machine: binary incrementer on 1011
    texts = " ".join(m.value for m in at.markdown)
    assert "δ(q0, 1) = (1, R, q0)" in texts  # rule applied at step 0
    _button(at, "End ⏭").click().run()
    assert not at.exception, [e.value for e in at.exception]
    step_slider = next(s for s in at.slider if s.label == "Step through the run")
    assert step_slider.value == 8  # incrementer halts after 8 steps
    _button(at, "◀ Prev").click().run()
    assert next(s for s in at.slider if s.label == "Step through the run").value == 7


def test_final_step_explains_why_it_halted():
    at = _fresh_app()
    at.sidebar.selectbox[0].select("Unary addition 111+11 (halts)").run()
    _button(at, "End ⏭").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("halt state" in m.value for m in at.success)  # reached H

    at.sidebar.text_area[0].set_value("q0 1 1 R q0").run()  # no rule for (q0, _)
    at.sidebar.text_input[2].set_value("11").run()
    _button(at, "End ⏭").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("no rule" in m.value for m in at.success)


def test_experiment_sections_render():
    at = _fresh_app()
    section = next(r for r in at.radio if r.key == "exp_section")
    for option in section.options:
        section.set_value(option).run()
        assert not at.exception, (option, [e.value for e in at.exception])
        section = next(r for r in at.radio if r.key == "exp_section")


def test_every_preset_renders_with_correct_verdict():
    at = _fresh_app()
    expected = {
        "Binary incrementer (halts)": "Halted",
        "3-state busy beaver (halts)": "Halted",
        "4-state busy beaver (halts in 107 steps)": "Halted",
        "2-state busy beaver (halts)": "Halted",
        "Unary addition 111+11 (halts)": "Halted",
        "Binary palindrome checker (halts, writes Y/N)": "Halted",
        "Binary countdown to zero (slow halter)": "Halted",
        "Custom symbols a/b: flip & halt": "Halted",
        "Binary counter (counts forever, never halts)": "Not halted within",
        "Ping-pong (provably loops)": "Provably never halts",
        "Run-away right (never halts, but not provable by repetition)": "Not halted within",
    }
    for preset, verdict in expected.items():
        at.sidebar.selectbox[0].select(preset).run()
        assert not at.exception, (preset, [e.value for e in at.exception])
        sim_messages = [m.value for m in (*at.success, *at.info, *at.warning)]
        assert any(verdict in m for m in sim_messages), (preset, sim_messages)


def test_random_machine_option_renders():
    at = _fresh_app()
    at.sidebar.selectbox[0].select("Random machine").run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.skipif(not HAS_MODELS, reason="run experiment.py to train models first")
def test_batch_test_matches_training_population():
    at = _fresh_app()
    batch_slider = next(s for s in at.slider if s.label == "Number of machines")
    batch_slider.set_value(50).run()
    _button(at, "Run batch test").click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Random Forest accuracy" in labels
    # Default generator should match the training population -> no shift warning
    assert not any("Distribution shift" in w.value for w in at.warning)


@pytest.mark.skipif(not HAS_MODELS, reason="run experiment.py to train models first")
def test_prediction_tab_compares_every_step_limit():
    at = _fresh_app()
    at.sidebar.selectbox[0].select("4-state busy beaver (halts in 107 steps)").run()
    assert not at.exception, [e.value for e in at.exception]
    tables = [df.value for df in at.dataframe]
    per_limit = next(t for t in tables if "simulation" in getattr(t, "columns", []))
    assert list(per_limit["L"]) == [10, 25, 50, 100, 250, 500, 1000]
    # BB(4) halts at step 107: simulation must say "not halted" for L <= 100, "halts" after
    sim = dict(zip(per_limit["L"], per_limit["simulation"]))
    assert sim[100] == "not halted within L" and sim[250] == "halts within L"
