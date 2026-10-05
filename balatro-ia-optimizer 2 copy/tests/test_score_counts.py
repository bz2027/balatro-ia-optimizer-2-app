from dataclasses import fields
from math import comb
from random import Random

import pytest

from math_model import RANKS, calculate_metrics, score_distribution, standard_deck


def decks():
    """Retained expected-score checks for ordinary and degenerate legal decks."""
    result = [standard_deck(), {2: 5}, {2: 3, 3: 2}, {14: 3, 13: 2},
              {rank: 1 for rank in range(2, 7)}, {2: 1, 3: 1, 4: 1, 5: 1, 14: 1}]
    random = Random(412)
    for _ in range(8):
        result.append({rank: random.randrange(7) for rank in RANKS})
    return result


@pytest.mark.parametrize("deck", decks())
def test_first_moment_exact_counts_and_expected_components(deck):
    distribution = score_distribution(deck)
    metrics = calculate_metrics(deck)
    counts = dict(distribution.score_counts)
    assert all(isinstance(c, int) and c >= 0 for c in counts.values())
    assert sum(counts.values()) == distribution.total_hands == comb(sum(deck.values()), 5)
    assert sum(s * c for s, c in counts.items()) == metrics.weighted_full_house + metrics.weighted_straight
    assert sum(s * c for s, c in counts.items()) / distribution.total_hands == pytest.approx(metrics.expected_total)
    assert metrics.expected_full_house + metrics.expected_straight == metrics.expected_total
    assert set(field.name for field in fields(distribution)) == {
        "total_hands", "score_counts", "weighted_full_house", "weighted_straight"
    }


def test_expected_score_cache_uses_immutable_counts():
    deck = standard_deck()
    original = score_distribution(deck)
    assert original is score_distribution(dict(deck))
    deck[2] -= 1
    assert score_distribution(deck) is not original
    assert calculate_metrics({2: 5}).expected_total == 0
    assert calculate_metrics({2: 3, 3: 2}).expected_total == 208
