import base64
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.io as pio
import pytest
from streamlit.testing.v1 import AppTest

from math_model import RANK_LABELS, RANKS, calculate_metrics, standard_deck
from formatting import format_decimal, format_error


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _luminance(hex_colour):
    channels = [int(hex_colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first, second):
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _figure(element):
    return pio.from_json(element.proto.spec)


def _values(value):
    if isinstance(value, dict) and "bdata" in value:
        raw = base64.b64decode(value["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).tolist()
    return list(value)


def test_dark_cards_rank_order_and_baseline_history_contract():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    assert not app.exception

    css = app.markdown[0].value
    assert "--panel: #151c2b" in css
    assert "--text: #f8fafc" in css
    assert "background: #f6f8fb" not in css
    assert _contrast("#f8fafc", "#151c2b") >= 4.5
    assert _contrast("#4ade80", "#101725") >= 4.5
    assert _contrast("#60a5fa", "#101725") >= 4.5

    plots = app.get("plotly_chart")
    assert len(plots) == 4
    deck_figure = _figure(plots[0])
    expected_order = [RANK_LABELS[rank] for rank in RANKS]
    assert list(deck_figure.data[0].x) == expected_order
    assert list(deck_figure.layout.xaxis.categoryarray) == expected_order
    assert deck_figure.data[0].text is not None
    assert deck_figure.data[0].textposition == "outside"

    baseline = calculate_metrics(standard_deck())
    expected_values = [
        baseline.expected_total,
        baseline.expected_straight,
        baseline.expected_full_house,
    ]
    for element, expected in zip(plots[1:], expected_values):
        figure = _figure(element)
        assert _values(figure.data[0].x) == [0]
        assert _values(figure.data[0].y) == pytest.approx([expected])
        assert "markers" in figure.data[0].mode

    assert len(app.dataframe) == 1
    baseline_row = app.dataframe[0].value.iloc[0]
    assert baseline_row["Round"] == 0
    assert baseline_row["New E"] == "1.404"


def test_evaluated_page_uses_one_three_decimal_sequential_value_everywhere():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    app.button(key="evaluate_round").click().run(timeout=45)
    assert not app.exception

    candidate = app.session_state["app_state"].pending.recommended
    atomic_sum = sum(
        step.prediction.trapezoidal_prediction for step in candidate.steps
    )
    assert candidate.predicted_change == pytest.approx(atomic_sum)

    recommendation = next(
        item.value for item in app.success if "Recommended: Hanged Man" in item.value
    )
    assert "+0.206" in recommendation

    candidate_markup = "\n".join(
        item.value
        for item in app.markdown
        if item.value.strip().startswith('<div class="candidate-card">')
    )
    assert "Sensitivity-predicted change" in candidate_markup
    assert "+0.206" in candidate_markup
    assert "margin" not in candidate_markup.lower()
    assert "confidence" not in candidate_markup.lower()

    hanged_total = next(
        item.value
        for item in app.latex
        if "Delta E" in item.value and "Hanged" in item.value
    )
    assert r"\left(+0.088\right) + \left(+0.117\right)\approx+0.206" in hanged_total

    death_step = next(
        item.value
        for item in app.latex
        if "M_{10}" in item.value and "Delta E" in item.value
    )
    assert r"\\&\quad +" in death_step

    labels = [expander.label for expander in app.expander]
    assert labels[:3] == [
        "Hanged Man — Show sensitivity working",
        "Death — Show sensitivity working",
        "Strength — Show sensitivity working",
    ]


def test_removed_panels_are_absent_and_remaining_ui_is_compact():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert len([m for m in app.markdown if m.value.strip().startswith('<div class="metric-card">')]) == 6
    assert "Model: only Full Houses and non-wheel Straights score." in [item.value for item in app.caption]
    assert "Negative favours removal; positive favours addition." not in markup
    assert "Sequential sensitivity optimization" not in markup
    assert len(app.get("download_button")) == 0
    assert [e.label for e in app.expander] == ["Reset, undo & automatic rounds"]
    assert all("Mathematical model" not in e.label for e in app.expander)
    for text in ("risk-row", "Blind", "sigma", "variance", "4 independent", "Show score counts"):
        assert text not in markup
    assert "show_score_counts" not in [checkbox.key for checkbox in app.checkbox]


def test_proposals_history_statistics_and_errors_after_multiple_rounds():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    for _ in range(3):
        app.button(key="evaluate_round").click().run()
        assert not app.exception
        state = app.session_state["app_state"]
        cards = [m.value for m in app.markdown if m.value.strip().startswith('<div class="candidate-card">')]
        assert len(cards) == 3
        for modifier, card in zip(("Hanged Man", "Death", "Strength"), cards):
            candidate = state.pending.best_by_modifier[modifier]
            assert format_decimal(candidate.predicted_change, signed=True) in card
            assert format_error(candidate.prediction_error, candidate.relative_prediction_error) in card
            for text in ("risk-row", "Blind", "σ", "Variance"):
                assert text not in card
        markup = "\n".join(m.value for m in app.markdown)
        assert "The summed sequential prediction" not in markup
        app.button(key="apply_recommended").click().run()
    state = app.session_state["app_state"]
    table = app.dataframe[0].value
    assert table["Round"].tolist() == [0, 1, 2, 3]
    for index, entry in enumerate(state.history):
        assert table.iloc[index]["New E"] == format_decimal(entry.expected_total)
        if index:
            assert table.iloc[index]["Prediction error"] == format_error(entry.prediction_error, entry.relative_prediction_error)
    assert all(not any(term in column for term in ("σ", "Blind", "hands", "variance")) for column in table.columns)


def test_formatting_retains_relative_and_tiny_validation_errors():
    assert format_error(0.01, 0.1) == "10.000000% relative"
    assert format_error(1e-15, 1e-15) == "1.000000e-13% relative"
    assert format_error(1e-12, None) == "+1.000000e-12 absolute"
    assert "e-" in format_error(1e-15, 1e-15)


def test_live_session_upgrade_keeps_investigation_and_staged_round():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    app.button(key="evaluate_round").click().run()
    app.button(key="apply_recommended").click().run()
    app.button(key="evaluate_round").click().run()
    saved = app.session_state["app_state"]
    old_entries = tuple(SimpleNamespace(**asdict(entry), risk=object()) for entry in saved.history)
    app.session_state["app_state"] = replace(saved, history=old_entries)
    app.session_state["app_version"] = "sequential-risk-v4"
    app.session_state["show_score_counts"] = True
    app.run()
    assert not app.exception
    upgraded = app.session_state["app_state"]
    assert upgraded.round_number == 1
    assert upgraded.deck == saved.deck
    assert len(upgraded.snapshots) == 1
    assert len(upgraded.history) == 2
    assert upgraded.pending is not None
    assert upgraded.pending.recommended.selection == saved.pending.recommended.selection
    assert upgraded.pending.recommended.predicted_change == saved.pending.recommended.predicted_change
    assert all(not hasattr(entry, "risk") for entry in upgraded.history)
    assert "show_score_counts" not in app.session_state
    app.button(key="apply_recommended").click().run()
    assert app.session_state["app_state"].round_number == 2
    app.button(key="undo_modifier").click().run()
    assert app.session_state["app_state"].round_number == 1
    assert app.session_state["app_state"].deck == saved.deck
