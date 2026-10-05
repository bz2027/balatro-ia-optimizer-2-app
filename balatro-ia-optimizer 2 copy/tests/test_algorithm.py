from dataclasses import replace

import pytest

from algorithm import (
    apply_recommended,
    apply_selected_modifier,
    evaluate_state,
    new_state,
    recommend_candidate,
    reset_state,
    run_automatic_rounds,
    undo_last_modifier,
    apply_candidate,
    evaluate_round,
    pending_is_current,
    reconcile_state,
)
from math_model import calculate_metrics, standard_deck
from modifiers import LEGACY_METHOD_ID, METHOD_ID, evaluate_death, evaluate_hanged_man
from types import SimpleNamespace


def test_first_recommendation_and_apply_pause_progression():
    state = evaluate_state(new_state())
    assert state.round_number == 0
    assert state.deck == standard_deck()
    assert state.pending.recommended.modifier == "Hanged Man"
    assert state.pending.recommended.selection == (2, 2)
    applied = apply_recommended(state)
    assert applied.round_number == 1
    assert applied.pending is None
    assert applied.deck[2] == 2
    assert calculate_metrics(applied.deck).deck_size == 50
    assert calculate_metrics(applied.deck).total_hands == 2_118_760
    atomic_sum = sum(
        step.prediction.trapezoidal_prediction
        for step in state.pending.recommended.steps
    )
    assert state.pending.recommended.predicted_change == pytest.approx(atomic_sum)
    assert applied.history[-1].predicted_change == pytest.approx(atomic_sum)


def test_override_undo_and_reset():
    staged = evaluate_state(new_state())
    death_state = apply_selected_modifier(staged, "Death")
    assert death_state.history[-1].modifier == "Death"
    restored = undo_last_modifier(death_state)
    assert restored.deck == standard_deck()
    assert restored.round_number == 0
    assert restored.history == new_state().history
    reset = reset_state(death_state)
    assert reset.deck == standard_deck()
    assert reset.snapshots == ()


def test_automatic_rounds_are_bounded_and_deterministic():
    first = run_automatic_rounds(new_state(), 2)
    second = run_automatic_rounds(new_state(), 2)
    assert first.round_number == 2
    assert first.deck == second.deck
    assert first.history == second.history


def test_recommendation_uses_prediction_not_actual_change():
    deck = standard_deck()
    hanged = evaluate_hanged_man(deck)[0]
    death = evaluate_death(deck)[0]
    manipulated_hanged = replace(hanged, predicted_change=10.0, actual_change=-999.0)
    manipulated_death = replace(death, predicted_change=9.0, actual_change=999.0)
    recommended, _, _ = recommend_candidate(
        [manipulated_hanged, manipulated_death]
    )
    assert recommended is manipulated_hanged


def test_cannot_apply_before_evaluation():
    with pytest.raises(ValueError):
        apply_recommended(new_state())


def test_evaluation_is_nonmutating():
    state = new_state()
    deck_before = dict(state.deck)
    staged = evaluate_state(state)
    assert state.pending is None
    assert state.deck == staged.deck == deck_before
    assert state.history == staged.history
    assert staged.snapshots == ()
    assert all(not hasattr(c, "risk") for ranked in staged.pending.rankings.values() for c in ranked)


def test_history_contains_actual_state_statistics_and_undo_reset():
    baseline = new_state()
    applied = run_automatic_rounds(baseline, 3)
    assert [e.round_number for e in applied.history] == [0, 1, 2, 3]
    for previous, entry in zip(applied.history, applied.history[1:]):
        exact = calculate_metrics(entry.deck)
        assert entry.expected_total == exact.expected_total
        assert entry.expected_full_house == exact.expected_full_house
        assert entry.expected_straight == exact.expected_straight
        assert entry.actual_change == pytest.approx(entry.expected_total - previous.expected_total)
        assert entry.prediction_error == entry.predicted_change - entry.actual_change
        assert entry.method_id == METHOD_ID
    restored = undo_last_modifier(applied)
    assert restored.history == applied.history[:-1]
    assert restored.deck == applied.history[-2].deck
    assert reset_state(applied).history == baseline.history


def test_stale_method_or_deck_cannot_be_applied():
    staged = evaluate_state(new_state())
    candidate = staged.pending.recommended
    old_method = replace(staged, pending=replace(staged.pending, method_id="whole-action-trapezoidal"))
    different_deck = dict(staged.deck)
    different_deck[2] -= 1
    changed = replace(staged, deck=different_deck)
    for stale in (old_method, changed):
        assert not pending_is_current(stale)
        with pytest.raises(ValueError):
            apply_candidate(stale, candidate)
        assert reconcile_state(stale).pending is None


def test_legacy_history_method_labels_are_preserved():
    applied = apply_recommended(evaluate_state(new_state()))
    for method in ("starting-only", "midpoint", "whole-action-trapezoidal"):
        legacy = replace(applied, history=(replace(applied.history[-1], method_id=method),))
        reconciled = reconcile_state(legacy)
        assert reconciled.history[0].round_number == 0
        assert reconciled.history[-1].method_id == method
    entry = applied.history[-1]
    old_entry = SimpleNamespace(**{name: getattr(entry, name) for name in (
        "deck", "round_number", "modifier", "card_changes", "predicted_change", "actual_change"
    )})
    unlabeled = replace(applied, history=(old_entry,))
    assert reconcile_state(unlabeled).history[-1].method_id == LEGACY_METHOD_ID


@pytest.mark.parametrize("gain", [0.0, -0.1, 1e-13])
def test_automatic_rounds_stop_without_positive_prediction(monkeypatch, gain):
    import algorithm
    baseline = new_state()
    pending = evaluate_round(baseline.deck)
    candidate = replace(pending.recommended, predicted_change=gain)
    staged = replace(baseline, pending=replace(pending, recommended=candidate))
    monkeypatch.setattr(algorithm, "evaluate_state", lambda state: staged)
    result = run_automatic_rounds(baseline, 5)
    assert result.round_number == 0
    assert result.history == baseline.history
    assert result.pending is None


def test_modifier_order_breaks_equal_prediction_ties():
    staged = evaluate_state(new_state())
    candidates = [replace(c, predicted_change=1.0) for c in staged.pending.best_by_modifier.values()]
    recommended, _, explanation = recommend_candidate(reversed(candidates))
    assert recommended.modifier == "Hanged Man"
    assert "tie" in explanation
