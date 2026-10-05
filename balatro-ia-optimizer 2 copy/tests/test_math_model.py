from math import comb

import pytest

from math_model import (
    RANKS,
    calculate_metrics,
    full_house_score,
    standard_deck,
    straight_score,
    validate_deck,
)


def test_standard_deck_exact_expected_score_decomposition():
    metrics = calculate_metrics(standard_deck())
    assert metrics.deck_size == 52
    assert metrics.total_hands == comb(52, 5) == 2_598_960
    assert metrics.weighted_full_house == 1_146_240
    assert metrics.weighted_straight == 2_502_656
    assert metrics.expected_total == pytest.approx(1.4039831317, abs=1e-10)
    assert metrics.expected_total == pytest.approx(
        metrics.expected_full_house + metrics.expected_straight,
        abs=1e-15,
    )


def test_scores_and_wheel_exclusion_boundaries():
    assert full_house_score(2, 3) == 4 * (40 + 6 + 6)
    assert straight_score(2) == 200
    assert straight_score(10) == 324
    with pytest.raises(ValueError):
        straight_score(1)


def test_invalid_decks_are_rejected():
    bad = standard_deck()
    bad[2] = -1
    with pytest.raises(ValueError):
        validate_deck(bad)
    with pytest.raises(ValueError):
        validate_deck({rank: 0 for rank in RANKS})
    bad_type = standard_deck()
    bad_type[2] = 2.5
    with pytest.raises(TypeError):
        validate_deck(bad_type)
