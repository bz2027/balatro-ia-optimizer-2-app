"""Legal modifier generation and sensitivity-only candidate ranking."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from math_model import RANKS, RANK_LABELS, Deck, DeckMetrics, calculate_metrics, validate_deck
from sensitivity import SensitivityPrediction, prediction_error, trapezoidal_sensitivity_prediction


MODIFIER_ORDER: Tuple[str, ...] = ("Hanged Man", "Death", "Strength")
METHOD_ID = "sequential-trapezoidal-lower-rank-first-v2"
LEGACY_METHOD_ID = "sequential-trapezoidal-ordered-v1"
GAIN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ActionStep:
    """One sequential atomic change and its complete sensitivity working."""

    step_number: int
    description: str
    deck_before: Deck
    deck_after: Deck
    prediction: SensitivityPrediction


@dataclass(frozen=True)
class ModifierCandidate:
    """One complete legal action evaluated from a committed deck."""

    modifier: str
    selection: Tuple[int, ...]
    card_changes: str
    proposed_deck: Deck
    predicted_change: float
    actual_change: float
    before_metrics: DeckMetrics
    after_metrics: DeckMetrics
    steps: Tuple[ActionStep, ...]
    removal_contribution: Optional[float] = None
    addition_contribution: Optional[float] = None
    tie_break_explanation: str = ""
    method_id: str = METHOD_ID

    @property
    def prediction_error(self) -> float:
        return prediction_error(self.predicted_change, self.actual_change)[0]

    @property
    def relative_prediction_error(self) -> Optional[float]:
        return prediction_error(self.predicted_change, self.actual_change)[1]


def next_rank(rank: int) -> int:
    """Return the isolated Strength progression, including Ace -> 2."""

    if rank not in RANKS:
        raise ValueError("Rank must be between 2 and 14")
    return 2 if rank == 14 else rank + 1


def _single_delta(source: int, target: Optional[int] = None) -> Dict[int, int]:
    delta = {rank: 0 for rank in RANKS}
    delta[source] -= 1
    if target is not None:
        delta[target] += 1
    return delta


def _candidate_sort_key(candidate: ModifierCandidate) -> Tuple[object, ...]:
    """Sort by prediction only, then by a deterministic lower-rank tuple."""

    return (-candidate.predicted_change, candidate.selection)


def rank_candidates(
    candidates: Sequence[ModifierCandidate],
) -> List[ModifierCandidate]:
    """Return candidates ranked without consulting actual expected changes."""

    return sorted(candidates, key=_candidate_sort_key)


def evaluate_hanged_man(deck: Mapping[int, int]) -> List[ModifierCandidate]:
    """Evaluate unordered original-card pairs; lower rank is deleted first."""

    counts = validate_deck(deck)
    if sum(counts.values()) < 7:
        return []
    before_metrics = calculate_metrics(counts)
    candidates: List[ModifierCandidate] = []
    for first_rank in RANKS:
        if counts[first_rank] < 1:
            continue
        first_prediction = trapezoidal_sensitivity_prediction(
            counts, _single_delta(first_rank)
        )
        temporary = first_prediction.end_deck
        for second_rank in range(first_rank, 15):
            if not _legal_two_selected_sources(counts, first_rank, second_rank):
                continue
            second_prediction = trapezoidal_sensitivity_prediction(
                temporary, _single_delta(second_rank)
            )
            final_deck = second_prediction.end_deck
            after_metrics = calculate_metrics(final_deck)
            steps = (
                ActionStep(
                    step_number=1,
                    description="Destroy one {}".format(RANK_LABELS[first_rank]),
                    deck_before=dict(counts),
                    deck_after=dict(temporary),
                    prediction=first_prediction,
                ),
                ActionStep(
                    step_number=2,
                    description="Destroy one {}".format(RANK_LABELS[second_rank]),
                    deck_before=dict(temporary),
                    deck_after=dict(final_deck),
                    prediction=second_prediction,
                ),
            )
            candidates.append(
                ModifierCandidate(
                    modifier="Hanged Man",
                    selection=(first_rank, second_rank),
                    card_changes="Destroy {} then {}".format(
                        RANK_LABELS[first_rank], RANK_LABELS[second_rank]
                    ),
                    proposed_deck=dict(final_deck),
                    predicted_change=(
                        first_prediction.trapezoidal_prediction
                        + second_prediction.trapezoidal_prediction
                    ),
                    actual_change=(
                        after_metrics.expected_total
                        - before_metrics.expected_total
                    ),
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    steps=steps,
                )
            )
    return rank_candidates(candidates)


def evaluate_death(deck: Mapping[int, int]) -> List[ModifierCandidate]:
    """Evaluate every legal ordered source -> existing-target conversion."""

    counts = validate_deck(deck)
    before_metrics = calculate_metrics(counts)
    candidates: List[ModifierCandidate] = []
    for source in RANKS:
        if counts[source] < 1:
            continue
        for target in RANKS:
            if target == source or counts[target] < 1:
                continue
            prediction = trapezoidal_sensitivity_prediction(
                counts, _single_delta(source, target)
            )
            final_deck = prediction.end_deck
            after_metrics = calculate_metrics(final_deck)
            removal = prediction.trapezoidal_contributions[source]
            addition = prediction.trapezoidal_contributions[target]
            step = ActionStep(
                step_number=1,
                description="Convert {} into {}".format(
                    RANK_LABELS[source], RANK_LABELS[target]
                ),
                deck_before=dict(counts),
                deck_after=dict(final_deck),
                prediction=prediction,
            )
            candidates.append(
                ModifierCandidate(
                    modifier="Death",
                    selection=(source, target),
                    card_changes="Convert {} -> {}".format(
                        RANK_LABELS[source], RANK_LABELS[target]
                    ),
                    proposed_deck=dict(final_deck),
                    predicted_change=prediction.trapezoidal_prediction,
                    actual_change=(
                        after_metrics.expected_total
                        - before_metrics.expected_total
                    ),
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    steps=(step,),
                    removal_contribution=removal,
                    addition_contribution=addition,
                )
            )
    return rank_candidates(candidates)


def _legal_two_selected_sources(counts: Mapping[int, int], first: int, second: int) -> bool:
    if first == second:
        return counts[first] >= 2
    return counts[first] >= 1 and counts[second] >= 1


def evaluate_strength(deck: Mapping[int, int]) -> List[ModifierCandidate]:
    """Convert each unordered original-card pair, lower starting rank first."""

    counts = validate_deck(deck)
    before_metrics = calculate_metrics(counts)
    candidates: List[ModifierCandidate] = []
    for first_rank in RANKS:
        for second_rank in range(first_rank, 15):
            if not _legal_two_selected_sources(
                counts, first_rank, second_rank
            ):
                continue
            first_target = next_rank(first_rank)
            first_prediction = trapezoidal_sensitivity_prediction(
                counts, _single_delta(first_rank, first_target)
            )
            temporary = first_prediction.end_deck
            second_target = next_rank(second_rank)
            second_prediction = trapezoidal_sensitivity_prediction(
                temporary, _single_delta(second_rank, second_target)
            )
            final_deck = second_prediction.end_deck
            after_metrics = calculate_metrics(final_deck)
            steps = (
                ActionStep(
                    step_number=1,
                    description="Strengthen {} -> {}".format(
                        RANK_LABELS[first_rank], RANK_LABELS[first_target]
                    ),
                    deck_before=dict(counts),
                    deck_after=dict(temporary),
                    prediction=first_prediction,
                ),
                ActionStep(
                    step_number=2,
                    description="Strengthen {} -> {}".format(
                        RANK_LABELS[second_rank], RANK_LABELS[second_target]
                    ),
                    deck_before=dict(temporary),
                    deck_after=dict(final_deck),
                    prediction=second_prediction,
                ),
            )
            candidates.append(
                ModifierCandidate(
                    modifier="Strength",
                    selection=(first_rank, second_rank),
                    card_changes="Strengthen {} -> {}, then {} -> {}".format(
                        RANK_LABELS[first_rank],
                        RANK_LABELS[first_target],
                        RANK_LABELS[second_rank],
                        RANK_LABELS[second_target],
                    ),
                    proposed_deck=dict(final_deck),
                    predicted_change=(
                        first_prediction.trapezoidal_prediction
                        + second_prediction.trapezoidal_prediction
                    ),
                    actual_change=(
                        after_metrics.expected_total
                        - before_metrics.expected_total
                    ),
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    steps=steps,
                )
            )
    return rank_candidates(candidates)
