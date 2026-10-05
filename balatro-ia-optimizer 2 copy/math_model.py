"""Exact expected-score model for the simplified Balatro investigation.

The functions in this module contain no Streamlit code.  Integer deck states are
evaluated with exact combinatorial counts; the continuous polynomial extension
used for sensitivity checks is exposed separately.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import comb
from typing import Dict, Mapping, Tuple


RANKS: Tuple[int, ...] = tuple(range(2, 15))
RANK_LABELS: Dict[int, str] = {
    **{rank: str(rank) for rank in range(2, 11)},
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
CARD_VALUES: Dict[int, int] = {
    **{rank: rank for rank in range(2, 11)},
    11: 10,
    12: 10,
    13: 10,
    14: 11,
}

Deck = Dict[int, int]


@dataclass(frozen=True)
class ScoreDistribution:
    """Cached exact score counts and first-moment weighted score totals."""

    total_hands: int
    score_counts: Tuple[Tuple[int, int], ...]
    weighted_full_house: int
    weighted_straight: int


@dataclass(frozen=True)
class DeckMetrics:
    """Exact totals and expected-score components for one integer deck."""

    deck_size: int
    total_hands: int
    weighted_full_house: int
    weighted_straight: int
    expected_full_house: float
    expected_straight: float
    expected_total: float

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly representation."""

        return asdict(self)


def standard_deck() -> Deck:
    """Return the standard 52-card rank-count deck."""

    return {rank: 4 for rank in RANKS}


def normalize_deck(deck: Mapping[int, int]) -> Deck:
    """Return a complete rank dictionary and reject unknown rank keys."""

    unknown = set(deck) - set(RANKS)
    if unknown:
        raise ValueError("Unknown rank(s): {}".format(sorted(unknown)))
    return {rank: deck.get(rank, 0) for rank in RANKS}


def validate_deck(deck: Mapping[int, int], minimum_size: int = 5) -> Deck:
    """Validate non-negative integer counts and a usable five-card deck.

    Args:
        deck: Mapping from internal ranks 2 through 14 to card counts.
        minimum_size: Smallest permitted total deck size.

    Returns:
        A normalized copy containing every rank.
    """

    normalized = normalize_deck(deck)
    for rank, count in normalized.items():
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Count for rank {} must be an integer".format(rank))
        if count < 0:
            raise ValueError("Count for rank {} cannot be negative".format(rank))
    if sum(normalized.values()) < minimum_size:
        raise ValueError(
            "Deck must contain at least {} cards".format(minimum_size)
        )
    return normalized


def choose_count(n: int, k: int) -> int:
    """Return C(n, k), treating n < k as zero legal hands."""

    if n < 0:
        raise ValueError("Card counts cannot be negative")
    return comb(n, k) if n >= k else 0


def continuous_choose_2(n: float) -> float:
    """Polynomial extension of C(n, 2)."""

    return n * (n - 1.0) / 2.0


def continuous_choose_3(n: float) -> float:
    """Polynomial extension of C(n, 3)."""

    return n * (n - 1.0) * (n - 2.0) / 6.0


def continuous_choose_5(n: float) -> float:
    """Polynomial extension of C(n, 5)."""

    return n * (n - 1.0) * (n - 2.0) * (n - 3.0) * (n - 4.0) / 120.0


def full_house_score(triple_rank: int, pair_rank: int) -> int:
    """Return S_FH(x, y) for the ordered triple/pair ranks."""

    if triple_rank not in RANKS or pair_rank not in RANKS:
        raise ValueError("Full House ranks must be between 2 and 14")
    if triple_rank == pair_rank:
        raise ValueError("Full House triple and pair ranks must differ")
    return 4 * (
        40 + 3 * CARD_VALUES[triple_rank] + 2 * CARD_VALUES[pair_rank]
    )


def straight_score(start_rank: int) -> int:
    """Return S_S(a) for a valid non-wheel Straight beginning at a."""

    if start_rank not in range(2, 11):
        raise ValueError("Straight start rank must be between 2 and 10")
    return 4 * (
        30 + sum(CARD_VALUES[rank] for rank in range(start_rank, start_rank + 5))
    )


def weighted_full_house(deck: Mapping[int, int]) -> int:
    """Calculate the exact weighted Full House score W_FH."""

    counts = validate_deck(deck)
    total = 0
    for triple_rank in RANKS:
        triples = choose_count(counts[triple_rank], 3)
        if triples == 0:
            continue
        for pair_rank in RANKS:
            if pair_rank == triple_rank:
                continue
            pairs = choose_count(counts[pair_rank], 2)
            total += (
                triples
                * pairs
                * full_house_score(triple_rank, pair_rank)
            )
    return total


def weighted_straight(deck: Mapping[int, int]) -> int:
    """Calculate the exact weighted Straight score W_S."""

    counts = validate_deck(deck)
    total = 0
    for start_rank in range(2, 11):
        combinations = 1
        for rank in range(start_rank, start_rank + 5):
            combinations *= counts[rank]
        total += combinations * straight_score(start_rank)
    return total


def calculate_metrics(deck: Mapping[int, int]) -> DeckMetrics:
    """Calculate N, T, weighted totals, and all exact expectations."""

    counts = validate_deck(deck)
    deck_size = sum(counts.values())
    distribution = score_distribution(counts)
    total_hands = distribution.total_hands
    full_house_total = distribution.weighted_full_house
    straight_total = distribution.weighted_straight
    expected_full_house = full_house_total / total_hands
    expected_straight = straight_total / total_hands
    return DeckMetrics(
        deck_size=deck_size,
        total_hands=total_hands,
        weighted_full_house=full_house_total,
        weighted_straight=straight_total,
        expected_full_house=expected_full_house,
        expected_straight=expected_straight,
        expected_total=expected_full_house + expected_straight,
    )


def deck_key(deck: Mapping[int, int]) -> Tuple[int, ...]:
    """Validated immutable key in rank order for cached mathematical results."""

    counts = validate_deck(deck)
    return tuple(counts[rank] for rank in RANKS)


@lru_cache(maxsize=4096)
def _score_distribution(key: Tuple[int, ...]) -> ScoreDistribution:
    counts = dict(zip(RANKS, key))
    total_hands = comb(sum(key), 5)
    by_score: Dict[int, int] = {}
    w_fh = w_s = 0
    # Triple/pair roles and five-distinct-rank Straights are disjoint patterns.
    for x in RANKS:
        for y in RANKS:
            if x == y:
                continue
            count = choose_count(counts[x], 3) * choose_count(counts[y], 2)
            if count:
                score = full_house_score(x, y)
                by_score[score] = by_score.get(score, 0) + count
                w_fh += count * score
    for a in range(2, 11):
        count = 1
        for rank in range(a, a + 5):
            count *= counts[rank]
        if count:
            score = straight_score(a)
            by_score[score] = by_score.get(score, 0) + count
            w_s += count * score
    zero_count = total_hands - sum(by_score.values())
    if zero_count < 0:
        raise ArithmeticError("Scoring patterns exceed the total number of hands")
    by_score[0] = zero_count
    return ScoreDistribution(
        total_hands, tuple(sorted(by_score.items())), w_fh, w_s
    )


def score_distribution(deck: Mapping[int, int]) -> ScoreDistribution:
    """Aggregate exact pattern counts without enumerating physical hands."""

    return _score_distribution(deck_key(deck))


def continuous_expected_score(deck: Mapping[int, float]) -> float:
    """Evaluate the continuous polynomial extension of E.

    This function is intended for numerical derivative validation.  Candidate
    decks and displayed exact metrics always use :func:`calculate_metrics`.
    """

    counts = {rank: float(deck.get(rank, 0.0)) for rank in RANKS}
    deck_size = sum(counts.values())
    denominator = continuous_choose_5(deck_size)
    if denominator == 0:
        raise ValueError("Continuous deck denominator is zero")

    full_house_total = 0.0
    for triple_rank in RANKS:
        for pair_rank in RANKS:
            if triple_rank == pair_rank:
                continue
            full_house_total += (
                continuous_choose_3(counts[triple_rank])
                * continuous_choose_2(counts[pair_rank])
                * full_house_score(triple_rank, pair_rank)
            )

    straight_total = 0.0
    for start_rank in range(2, 11):
        combinations = 1.0
        for rank in range(start_rank, start_rank + 5):
            combinations *= counts[rank]
        straight_total += combinations * straight_score(start_rank)

    return (full_house_total + straight_total) / denominator
