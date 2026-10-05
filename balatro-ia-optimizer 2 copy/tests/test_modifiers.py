import pytest

from math_model import RANKS, standard_deck
from modifiers import (
    evaluate_death,
    evaluate_hanged_man,
    evaluate_strength,
    next_rank,
    rank_candidates,
)
from dataclasses import replace
from itertools import combinations_with_replacement
from sensitivity import calculate_sensitivities


def test_initial_best_modifier_actions_and_benchmarks():
    deck = standard_deck()
    hanged = evaluate_hanged_man(deck)[0]
    death = evaluate_death(deck)[0]
    strength = evaluate_strength(deck)[0]
    assert hanged.selection == (2, 2)
    assert hanged.predicted_change == pytest.approx(0.205878462, abs=1e-9)
    assert death.selection == (2, 10)
    assert death.predicted_change == pytest.approx(0.171994952, abs=1e-9)
    assert strength.selection == (2, 2)
    assert strength.predicted_change == pytest.approx(0.122230431, abs=1e-9)


@pytest.mark.parametrize(
    "evaluator", [evaluate_hanged_man, evaluate_death, evaluate_strength]
)
def test_displayed_prediction_equals_sum_of_atomic_predictions(evaluator):
    candidate = evaluator(standard_deck())[0]
    atomic_sum = sum(
        step.prediction.trapezoidal_prediction for step in candidate.steps
    )
    assert candidate.predicted_change == pytest.approx(atomic_sum, abs=1e-15)


def test_hanged_man_sequential_state_and_recalculation():
    candidate = evaluate_hanged_man(standard_deck())[0]
    assert len(candidate.steps) == 2
    assert candidate.steps[0].deck_after[2] == 3
    assert candidate.steps[1].deck_before == candidate.steps[0].deck_after
    assert candidate.steps[1].deck_after[2] == 2
    assert sum(candidate.proposed_deck.values()) == 50
    assert (
        candidate.steps[0].prediction.end_sensitivities
        == candidate.steps[1].prediction.start_sensitivities
    )


def test_strength_sequential_state_and_ace_wrap():
    candidate = evaluate_strength(standard_deck())[0]
    assert candidate.steps[0].deck_after[2] == 3
    assert candidate.steps[0].deck_after[3] == 5
    assert candidate.steps[1].deck_before == candidate.steps[0].deck_after
    assert candidate.proposed_deck[2] == 2
    assert candidate.proposed_deck[3] == 6
    assert next_rank(14) == 2


def test_death_is_atomic_preserves_n_and_exposes_contributions():
    candidate = evaluate_death(standard_deck())[0]
    assert len(candidate.steps) == 1
    assert candidate.after_metrics.deck_size == candidate.before_metrics.deck_size
    assert candidate.removal_contribution + candidate.addition_contribution == pytest.approx(
        candidate.predicted_change
    )


def test_legal_action_generation_and_same_rank_copy_requirement():
    deck = {rank: 0 for rank in RANKS}
    deck[2] = 1
    deck[3] = 5
    deck[10] = 1
    hanged = evaluate_hanged_man(deck)
    assert all(candidate.selection != (2, 2) for candidate in hanged)
    strength = evaluate_strength(deck)
    assert all(candidate.selection != (2, 2) for candidate in strength)
    assert any(candidate.selection == (2, 3) for candidate in strength)


def test_candidate_order_is_deterministic():
    first = [candidate.selection for candidate in evaluate_strength(standard_deck())]
    second = [candidate.selection for candidate in evaluate_strength(standard_deck())]
    assert first == second


@pytest.mark.parametrize("evaluator", [evaluate_hanged_man, evaluate_strength])
@pytest.mark.parametrize("deck", [standard_deck(), {2: 1, 3: 5, 10: 1}, {2: 1, 14: 6}, {2: 7}])
def test_all_unordered_original_pairs_and_fixed_lower_rank_first(evaluator, deck):
    pairs = {
        (a, b) for a, b in combinations_with_replacement(RANKS, 2)
        if deck.get(a, 0) >= (2 if a == b else 1) and deck.get(b, 0) >= 1
    }
    candidates = evaluator(deck)
    assert {c.selection for c in candidates} == pairs
    assert len(candidates) == len(pairs)
    for c in candidates:
        a, b = c.selection
        assert a <= b
        assert c.steps[0].prediction.delta[a] == -1
        assert c.steps[1].prediction.delta[b] == -1
        assert c.steps[0].deck_after == c.steps[1].deck_before
        assert c.steps[0].prediction.end_sensitivities == c.steps[1].prediction.start_sensitivities
        for step in c.steps:
            for rank in RANKS:
                assert step.deck_after[rank] == step.deck_before[rank] + step.prediction.delta[rank]
            endpoint_average = sum(
                (step.prediction.start_sensitivities[r] + step.prediction.end_sensitivities[r])
                * step.prediction.delta[r] / 2 for r in RANKS
            )
            assert step.prediction.trapezoidal_prediction == pytest.approx(endpoint_average, abs=1e-14)


def test_strength_cannot_reselect_created_card_even_with_adjacent_ranks():
    candidates = evaluate_strength({2: 1, 10: 6})
    assert {c.selection for c in candidates} == {(2, 10), (10, 10)}
    adjacent = next(c for c in evaluate_strength({2: 1, 3: 6}) if c.selection == (2, 3))
    assert adjacent.steps[0].deck_after[3] == 7
    assert adjacent.proposed_deck[3] == 6  # Second card was one of six original 3s.
    assert all(c.selection != (2, 2) for c in candidates)


def test_death_all_ordered_existing_targets_and_atomic_endpoint_n():
    deck = {2: 1, 6: 1, 14: 5}
    candidates = evaluate_death(deck)
    assert {c.selection for c in candidates} == {(a, b) for a in deck for b in deck if a != b}
    for c in candidates:
        step = c.steps[0]
        assert sum(step.deck_before.values()) == sum(step.deck_after.values()) == 7
        endpoint = calculate_sensitivities(step.deck_after)
        assert step.prediction.end_sensitivities == {r: endpoint[r].total_sensitivity for r in RANKS}


def test_hanged_minimum_size_and_strength_constant_n():
    assert evaluate_hanged_man({2: 6}) == []
    assert evaluate_hanged_man({2: 7})[0].after_metrics.deck_size == 5
    for c in evaluate_strength(standard_deck()):
        assert all(sum(s.deck_before.values()) == sum(s.deck_after.values()) == 52 for s in c.steps)


def test_hanged_endpoint_sensitivities_and_both_atomic_benchmarks():
    c = evaluate_hanged_man(standard_deck())[0]
    assert [c.steps[0].prediction.start_sensitivities[2],
            c.steps[0].prediction.end_sensitivities[2],
            c.steps[1].prediction.end_sensitivities[2]] == pytest.approx(
                [-0.07389029045773036, -0.10305550119207345, -0.13175563137481994], abs=1e-15)
    assert [s.prediction.trapezoidal_prediction for s in c.steps] == pytest.approx([0.0884728958249019, 0.1174055662834467], abs=1e-15)
    strength = evaluate_strength(standard_deck())[0]
    assert [s.prediction.trapezoidal_prediction for s in strength.steps] == pytest.approx([0.04218161623623809, 0.0800488143975539], abs=1e-15)


def test_rank_tuple_breaks_equal_predictions_ignoring_exact_change():
    original = evaluate_strength(standard_deck())
    first = replace(original[0], selection=(2, 2), predicted_change=1, actual_change=-100)
    second = replace(original[1], selection=(2, 3), predicted_change=1, actual_change=100)
    assert rank_candidates([second, first])[0] is first
