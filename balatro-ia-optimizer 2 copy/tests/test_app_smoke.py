from pathlib import Path

import pytest


streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_app_loads_without_exception():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    assert "Balatro IA Deck Optimizer" in [title.value for title in app.title]
    labels = [button.label for button in app.button]
    assert "Evaluate Next Modifier Round" in labels
    assert "Apply Recommended Modifier" in labels
    assert "Apply Hanged Man" in labels
    assert "Reset to Standard Deck" in labels
    assert "Undo Last Modifier" in labels
    assert "Run Several Rounds Automatically" in labels
    assert "Use Custom Deck" not in labels
    assert not app.tabs


def test_all_state_changing_buttons_are_wired():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()

    # Stage and apply the sensitivity recommendation.
    app.button(key="evaluate_round").click().run(timeout=45)
    assert not app.exception
    assert app.session_state["app_state"].pending is not None
    assert app.session_state["app_state"].round_number == 0
    recommendation = app.session_state["app_state"].pending.recommended
    atomic_sum = sum(
        step.prediction.trapezoidal_prediction for step in recommendation.steps
    )
    assert recommendation.predicted_change == pytest.approx(atomic_sum)
    assert recommendation.predicted_change == pytest.approx(0.205878462, abs=1e-9)
    assert app.selectbox(key="selected_modifier").value == "Hanged Man"
    assert app.button(key="apply_recommended").label == (
        "Apply Recommended Modifier (Hanged Man)"
    )
    assert app.button(key="apply_selected").label == "Apply Hanged Man"
    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert "+0.206" in rendered_markdown
    app.button(key="apply_recommended").click().run(timeout=45)
    assert app.session_state["app_state"].round_number == 1
    assert app.session_state["app_state"].deck[2] == 2
    assert app.session_state["app_state"].history[-1].predicted_change == pytest.approx(
        atomic_sum
    )

    # Undo, stage again, and override with Death.
    app.button(key="undo_modifier").click().run(timeout=45)
    assert app.session_state["app_state"].round_number == 0
    assert app.session_state["app_state"].deck[2] == 4
    app.button(key="evaluate_round").click().run(timeout=45)
    app.selectbox(key="selected_modifier").set_value("Death").run(timeout=45)
    assert app.button(key="apply_selected").label == "Apply Death"
    app.button(key="apply_selected").click().run(timeout=45)
    assert app.session_state["app_state"].history[-1].modifier == "Death"

    # Reset and then acknowledge the safety control for one automatic round.
    app.button(key="reset_standard").click().run(timeout=45)
    assert sum(app.session_state["app_state"].deck.values()) == 52
    app.checkbox(key="auto_confirm").check().run(timeout=45)
    assert not app.button(key="auto_rounds_button").disabled
    app.number_input(key="auto_round_count").set_value(1)
    app.button(key="auto_rounds_button").click().run(timeout=45)
    assert app.session_state["app_state"].round_number == 1
