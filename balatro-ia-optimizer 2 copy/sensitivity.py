"""Analytic sensitivity calculations and trapezoidal predictions."""

from dataclasses import asdict, dataclass
from math import comb
from typing import Dict, Mapping, Optional, Tuple

from math_model import (
    RANKS,
    Deck,
    calculate_metrics,
    continuous_choose_2,
    continuous_choose_3,
    full_house_score,
    straight_score,
    validate_deck,
)


@dataclass(frozen=True)
class RankSensitivity:
    """Derivative components for one rank."""

    rank: int
    count: int
    full_house_derivative: float
    straight_derivative: float
    denominator_derivative: float
    total_sensitivity: float

    @property
    def weighted_derivative(self) -> float:
        """Return d(W_FH + W_S)/dn_r."""

        return self.full_house_derivative + self.straight_derivative

    @property
    def interpretation(self) -> str:
        """Translate the sign into a plain-language local interpretation."""

        if self.total_sensitivity > 1e-12:
            return "Positive: addition is locally favoured"
        if self.total_sensitivity < -1e-12:
            return "Negative: removal is locally favoured"
        return "Near zero: locally neutral"

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly representation."""

        data = asdict(self)
        data["weighted_derivative"] = self.weighted_derivative
        data["interpretation"] = self.interpretation
        return data


@dataclass(frozen=True)
class SensitivityPrediction:
    """Working for one atomic deck change."""

    delta: Dict[int, int]
    local_prediction: float
    trapezoidal_prediction: float
    start_sensitivities: Dict[int, float]
    end_sensitivities: Dict[int, float]
    trapezoidal_contributions: Dict[int, float]
    end_deck: Deck

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly representation."""

        return {
            "delta": dict(self.delta),
            "local_prediction": self.local_prediction,
            "trapezoidal_prediction": self.trapezoidal_prediction,
            "start_sensitivities": dict(self.start_sensitivities),
            "end_sensitivities": dict(self.end_sensitivities),
            "trapezoidal_contributions": dict(
                self.trapezoidal_contributions
            ),
            "end_deck": dict(self.end_deck),
        }


def prediction_error(predicted: float, actual: float) -> Tuple[float, Optional[float]]:
    """Signed predicted-minus-actual error and relative magnitude of the gain.

    Relative error is undefined when the actual gain is within 1e-12 of zero.
    This validation helper is independent of candidate selection.
    """

    error = predicted - actual
    relative = abs(error / actual) if abs(actual) > 1e-12 else None
    return error, relative


def derivative_choose_2(n: float) -> float:
    """Return d/dn C(n, 2) under the polynomial extension."""

    return n - 0.5


def derivative_choose_3(n: float) -> float:
    """Return d/dn C(n, 3) under the polynomial extension."""

    return (3.0 * n * n - 6.0 * n + 2.0) / 6.0


def full_house_derivative(deck: Mapping[int, int], rank: int) -> float:
    """Calculate dW_FH/dn_r from the two ordered Full House roles."""

    counts = validate_deck(deck)
    if rank not in RANKS:
        raise ValueError("Rank must be between 2 and 14")

    triple_term = 0.0
    for pair_rank in RANKS:
        if pair_rank == rank:
            continue
        triple_term += (
            continuous_choose_2(counts[pair_rank])
            * derivative_choose_3(counts[rank])
            * full_house_score(rank, pair_rank)
        )

    pair_term = 0.0
    for triple_rank in RANKS:
        if triple_rank == rank:
            continue
        pair_term += (
            continuous_choose_3(counts[triple_rank])
            * derivative_choose_2(counts[rank])
            * full_house_score(triple_rank, rank)
        )
    return triple_term + pair_term


def straight_derivative(deck: Mapping[int, int], rank: int) -> float:
    """Calculate dW_S/dn_r using only Straights containing rank r."""

    counts = validate_deck(deck)
    if rank not in RANKS:
        raise ValueError("Rank must be between 2 and 14")

    total = 0.0
    for start_rank in range(2, 11):
        if not start_rank <= rank <= start_rank + 4:
            continue
        product = 1.0
        for other_rank in range(start_rank, start_rank + 5):
            if other_rank != rank:
                product *= counts[other_rank]
        total += straight_score(start_rank) * product
    return total


def denominator_derivative(deck: Mapping[int, int]) -> float:
    """Calculate dT/dn_r; it is the same for every rank at a fixed deck."""

    counts = validate_deck(deck)
    deck_size = sum(counts.values())
    total_hands = comb(deck_size, 5)
    reciprocal_sum = sum(1.0 / (deck_size - offset) for offset in range(5))
    return total_hands * reciprocal_sum


def calculate_sensitivities(
    deck: Mapping[int, int],
) -> Dict[int, RankSensitivity]:
    """Calculate all 13 complete sensitivities for the supplied deck."""

    counts = validate_deck(deck)
    metrics = calculate_metrics(counts)
    d_denominator = denominator_derivative(counts)
    result: Dict[int, RankSensitivity] = {}
    for rank in RANKS:
        d_full_house = full_house_derivative(counts, rank)
        d_straight = straight_derivative(counts, rank)
        total = (
            d_full_house
            + d_straight
            - metrics.expected_total * d_denominator
        ) / metrics.total_hands
        result[rank] = RankSensitivity(
            rank=rank,
            count=counts[rank],
            full_house_derivative=d_full_house,
            straight_derivative=d_straight,
            denominator_derivative=d_denominator,
            total_sensitivity=total,
        )
    return result


def apply_delta(deck: Mapping[int, int], delta: Mapping[int, int]) -> Deck:
    """Apply an integer change vector and reject impossible deck states."""

    counts = validate_deck(deck)
    unknown = set(delta) - set(RANKS)
    if unknown:
        raise ValueError("Unknown rank(s) in change vector: {}".format(sorted(unknown)))
    updated = dict(counts)
    for rank, change in delta.items():
        if isinstance(change, bool) or not isinstance(change, int):
            raise TypeError("Deck changes must be integers")
        updated[rank] += change
        if updated[rank] < 0:
            raise ValueError(
                "Cannot remove a card of rank {} that does not exist".format(rank)
            )
    return validate_deck(updated)


def trapezoidal_sensitivity_prediction(
    deck: Mapping[int, int], delta: Mapping[int, int]
) -> SensitivityPrediction:
    """Predict a finite atomic change with the average endpoint gradient.

    Only the analytic endpoint gradients enter this prediction. Their derived
    formula includes E; an exact difference of endpoint scores is never used.
    """

    counts = validate_deck(deck)
    unknown = set(delta) - set(RANKS)
    if unknown:
        raise ValueError(
            "Unknown rank(s) in change vector: {}".format(sorted(unknown))
        )
    for change in delta.values():
        if isinstance(change, bool) or not isinstance(change, int):
            raise TypeError("Deck changes must be integers")
    clean_delta = {rank: delta.get(rank, 0) for rank in RANKS}
    end_deck = apply_delta(counts, clean_delta)
    start_records = calculate_sensitivities(counts)
    end_records = calculate_sensitivities(end_deck)
    start = {rank: start_records[rank].total_sensitivity for rank in RANKS}
    end = {rank: end_records[rank].total_sensitivity for rank in RANKS}
    local = sum(start[rank] * clean_delta[rank] for rank in RANKS)
    contributions = {
        rank: 0.5
        * (start[rank] + end[rank])
        * clean_delta[rank]
        for rank in RANKS
    }
    predicted = sum(contributions.values())
    return SensitivityPrediction(
        delta=clean_delta,
        local_prediction=local,
        trapezoidal_prediction=predicted,
        start_sensitivities=start,
        end_sensitivities=end,
        trapezoidal_contributions=contributions,
        end_deck=end_deck,
    )
