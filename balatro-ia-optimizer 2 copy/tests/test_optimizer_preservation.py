"""Before/after regression and targeted feature-removal contracts."""

import ast
import csv
import io
import json
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithm import (
    AppState, HistoryEntry, apply_recommended, evaluate_state, new_state,
    pending_is_current, reconcile_state, run_automatic_rounds, undo_last_modifier,
)
from math_model import RANKS, calculate_metrics
from modifiers import METHOD_ID, ModifierCandidate
from reporting import actions_as_csv, detailed_html_report, history_as_csv, state_as_json
from sensitivity import prediction_error


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "optimizer_18_rounds.json").read_text())
REMOVED_FIELDS = {"risk", "variance", "sigma", "second_moment", "squared_full_house",
                  "squared_straight", "small_blind", "big_blind", "boss_blind"}


def test_all_18_rounds_match_reference_captured_before_removal():
    state = new_state()
    for expected in REFERENCE["rounds"]:
        if expected["round"]:
            staged = evaluate_state(state)
            candidate = staged.pending.recommended
            assert candidate.modifier == expected["modifier"]
            assert list(candidate.selection) == expected["selection"]
            assert candidate.card_changes == expected["card_changes"]
            assert candidate.predicted_change == pytest.approx(expected["predicted_gain"], rel=1e-13, abs=1e-14)
            assert candidate.actual_change == pytest.approx(expected["actual_gain"], rel=1e-13, abs=1e-14)
            state = apply_recommended(staged)
        metrics = calculate_metrics(state.deck)
        assert state.round_number == expected["round"]
        assert [state.deck[r] for r in RANKS] == expected["counts"]
        assert metrics.deck_size == expected["N"]
        assert metrics.total_hands == expected["T"]
        assert (metrics.expected_full_house, metrics.expected_straight, metrics.expected_total) == pytest.approx(
            (expected["E_FH"], expected["E_S"], expected["E"]), rel=1e-13, abs=1e-14
        )
        entry = state.history[-1]
        assert entry.predicted_change == pytest.approx(expected["predicted_gain"], rel=1e-13, abs=1e-14)
        assert entry.actual_change == pytest.approx(expected["actual_gain"], rel=1e-13, abs=1e-14)
    automatic = run_automatic_rounds(new_state(), 18)
    assert automatic.deck == state.deck
    assert automatic.history == state.history
    assert {r: c for r, c in state.deck.items() if c} == {10: 8, 11: 4, 12: 4, 14: 8}
    assert state.history[15].predicted_change == pytest.approx(4.823608758391368, rel=1e-13)
    assert state.history[18].expected_total == pytest.approx(43.11537737624694, rel=1e-13)


@pytest.mark.parametrize("predicted,actual,expected_error,expected_relative", [
    (0.2058784621083486, 0.20596230807131066, -8.384596296207025e-05, 0.0004070937238333925),
    (0.1, 0.05, 0.05, 1.0),
    (-0.1, -0.05, -0.05, 1.0),
    (1e-8, 0, 1e-8, None),
    (0.1, 1e-12, 0.1 - 1e-12, None),
    (0.1, -1e-12, 0.1 + 1e-12, None),
    (0.1, 2e-12, 0.1 - 2e-12, (0.1 - 2e-12) / 2e-12),
])
def test_prediction_error_migration_preserves_values(predicted, actual, expected_error, expected_relative):
    error, relative = prediction_error(predicted, actual)
    assert error == pytest.approx(expected_error, abs=1e-18)
    if expected_relative is None:
        assert relative is None
    else:
        assert relative == pytest.approx(expected_relative, rel=1e-14)


def test_legacy_extra_fields_are_stripped_without_losing_history_or_pending():
    state = evaluate_state(apply_recommended(evaluate_state(new_state())))
    # Model the native objects held by a live Streamlit session during reload.
    history = tuple(SimpleNamespace(**asdict(entry), risk=object(), sigma=999) for entry in state.history)
    def old_candidate(candidate):
        if candidate is None:
            return None
        values = {field.name: getattr(candidate, field.name) for field in fields(ModifierCandidate)}
        return SimpleNamespace(**values, risk=object())
    best = {name: old_candidate(candidate) for name, candidate in state.pending.best_by_modifier.items()}
    old_pending = replace(
        state.pending,
        rankings={name: tuple(old_candidate(c) for c in ranking) for name, ranking in state.pending.rankings.items()},
        best_by_modifier=best, recommended=best[state.pending.recommended.modifier],
    )
    snapshots = tuple(replace(snapshot, history=tuple(
        SimpleNamespace(**asdict(entry), risk=object()) for entry in snapshot.history
    )) for snapshot in state.snapshots)
    old_state = replace(state, history=history, snapshots=snapshots, pending=old_pending)
    cleaned = reconcile_state(old_state)
    assert cleaned.deck == state.deck
    assert cleaned.round_number == state.round_number
    assert cleaned.history == state.history
    assert cleaned.snapshots == state.snapshots
    assert cleaned.pending == state.pending
    assert pending_is_current(cleaned)
    assert all(not hasattr(entry, "risk") and not hasattr(entry, "sigma") for entry in cleaned.history)
    assert all(not hasattr(candidate, "risk") for ranked in cleaned.pending.rankings.values() for candidate in ranked)
    assert apply_recommended(cleaned).history == apply_recommended(state).history
    assert undo_last_modifier(cleaned).history == new_state().history
    payload = json.loads(state_as_json(old_state))
    assert all("risk" not in entry and "sigma" not in entry for entry in payload["history"])
    assert payload["recommended"]["prediction_error"] == state.pending.recommended.prediction_error


def test_reconciliation_keeps_recorded_errors_and_method_identifiers():
    state = run_automatic_rounds(new_state(), 1)
    recorded = replace(state.history[-1], prediction_error=0.123, relative_prediction_error=0.456, method_id="midpoint")
    legacy = SimpleNamespace(**asdict(recorded), risk=object())
    cleaned = reconcile_state(replace(state, history=(state.history[0], legacy)))
    assert cleaned.history[-1] == recorded
    assert cleaned.history[-1].method_id == "midpoint"


def test_reports_and_new_records_have_no_removed_fields():
    staged = evaluate_state(run_automatic_rounds(new_state(), 2))
    for cls in (HistoryEntry, ModifierCandidate):
        assert not REMOVED_FIELDS.intersection(field.name for field in fields(cls))
    payload = json.loads(state_as_json(staged))
    def inspect(value):
        if isinstance(value, dict):
            assert not REMOVED_FIELDS.intersection(value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(payload)
    history = list(csv.DictReader(io.StringIO(history_as_csv(staged.history))))
    assert len(history) == 3
    assert float(history[-1]["Prediction error"]) == staged.history[-1].prediction_error
    assert float(history[-1]["Relative prediction error"]) == staged.history[-1].relative_prediction_error
    for output in (history_as_csv(staged.history), actions_as_csv(staged.pending), detailed_html_report(staged)):
        for token in ("Blind", "variance", "sigma", "second moment", "Risk", "four independent"):
            assert token not in output


def test_optimizer_runs_in_fresh_process_with_risk_import_forbidden():
    script = """
import importlib.abc
import sys
class RejectRemoved(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'risk':
            raise AssertionError('Removed module imported')
sys.meta_path.insert(0, RejectRemoved())
from algorithm import new_state, evaluate_state, apply_recommended
from reporting import state_as_json, history_as_csv, detailed_html_report, actions_as_csv
state = evaluate_state(new_state())
actions_as_csv(state.pending)
detailed_html_report(state)
state = apply_recommended(state)
state_as_json(state)
history_as_csv(state.history)
assert 'risk' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (ROOT / "risk.py").exists()
    for filename in ("app.py", "algorithm.py", "modifiers.py", "math_model.py", "reporting.py", "formatting.py", "sensitivity.py"):
        tree = ast.parse((ROOT / filename).read_text())
        assert not any(isinstance(node, ast.ImportFrom) and node.module == "risk" for node in ast.walk(tree))
