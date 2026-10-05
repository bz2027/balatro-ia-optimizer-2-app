import pytest

from math_model import RANKS, continuous_expected_score, standard_deck
from sensitivity import (
    apply_delta,
    calculate_sensitivities,
    trapezoidal_sensitivity_prediction,
)


def test_initial_sensitivity_reference_values():
    sensitivities = calculate_sensitivities(standard_deck())
    assert sensitivities[2].total_sensitivity == pytest.approx(
        -0.0738902905, abs=1e-10
    )
    assert sensitivities[3].total_sensitivity == pytest.approx(
        -0.05040705100810436, abs=1e-14
    )
    assert sensitivities[10].total_sensitivity == pytest.approx(
        0.0710292158, abs=1e-10
    )


@pytest.mark.parametrize("rank", RANKS)
def test_analytic_sensitivity_matches_central_difference(rank):
    deck = {r: float(count) for r, count in standard_deck().items()}
    h = 1e-5
    plus = dict(deck)
    minus = dict(deck)
    plus[rank] += h
    minus[rank] -= h
    numerical = (
        continuous_expected_score(plus) - continuous_expected_score(minus)
    ) / (2 * h)
    analytic = calculate_sensitivities(standard_deck())[rank].total_sensitivity
    assert analytic == pytest.approx(numerical, rel=2e-8, abs=2e-9)


def test_apply_delta_rejects_nonexistent_cards_and_small_deck():
    deck = {rank: 0 for rank in RANKS}
    deck[2] = 5
    with pytest.raises(ValueError):
        apply_delta(deck, {3: -1})
    with pytest.raises(ValueError):
        apply_delta(deck, {2: -1})
    with pytest.raises(ValueError):
        trapezoidal_sensitivity_prediction(standard_deck(), {99: 1})
    with pytest.raises(TypeError):
        trapezoidal_sensitivity_prediction(standard_deck(), {2: True})


def test_trapezoid_uses_both_endpoint_sensitivities():
    prediction = trapezoidal_sensitivity_prediction(standard_deck(), {2: -2})
    expected = (
        0.5
        * (prediction.start_sensitivities[2] + prediction.end_sensitivities[2])
        * -2
    )
    assert prediction.trapezoidal_prediction == pytest.approx(expected)
