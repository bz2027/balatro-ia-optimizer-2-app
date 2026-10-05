"""JSON, CSV, and readable HTML exports for the Balatro IA application."""

import csv
import html
import io
import json
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence

from algorithm import AppState, HistoryEntry, RoundEvaluation, reconcile_state
from math_model import RANKS, RANK_LABELS, calculate_metrics
from modifiers import ModifierCandidate
from sensitivity import calculate_sensitivities
from formatting import format_decimal


def _labelled_deck(deck: Dict[int, int]) -> Dict[str, int]:
    return {RANK_LABELS[rank]: deck[rank] for rank in RANKS}


def candidate_to_dict(
    candidate: ModifierCandidate, include_working: bool = True
) -> Dict[str, object]:
    """Convert a candidate and optionally its step-by-step working."""

    data: Dict[str, object] = {
        "modifier": candidate.modifier,
        "selection": [RANK_LABELS[rank] for rank in candidate.selection],
        "card_changes": candidate.card_changes,
        "predicted_sensitivity_change": candidate.predicted_change,
        "actual_change_validation_only": candidate.actual_change,
        "proposed_deck": _labelled_deck(candidate.proposed_deck),
        "after_metrics": candidate.after_metrics.to_dict(),
        "method_id": candidate.method_id,
        "prediction_error": candidate.prediction_error,
        "relative_prediction_error": candidate.relative_prediction_error,
        "tie_break_explanation": candidate.tie_break_explanation,
        "removal_contribution": candidate.removal_contribution,
        "addition_contribution": candidate.addition_contribution,
    }
    if include_working:
        data["steps"] = [
            {
                "step_number": step.step_number,
                "description": step.description,
                "deck_before": _labelled_deck(step.deck_before),
                "deck_after": _labelled_deck(step.deck_after),
                "sensitivity_prediction": step.prediction.to_dict(),
            }
            for step in candidate.steps
        ]
    return data


def state_as_json(state: AppState) -> str:
    """Return the complete current state as formatted JSON."""

    state = reconcile_state(state)
    metrics = calculate_metrics(state.deck)
    sensitivities = calculate_sensitivities(state.deck)
    payload: Dict[str, object] = {
        "modifier_round": state.round_number,
        "deck": _labelled_deck(state.deck),
        "initial_deck": _labelled_deck(state.initial_deck),
        "metrics": metrics.to_dict(),
        "sensitivities": {
            RANK_LABELS[rank]: sensitivities[rank].to_dict()
            for rank in RANKS
        },
        "history": [asdict(entry) for entry in state.history],
    }
    if state.pending is not None:
        payload["pending_candidates"] = {
            modifier: (
                candidate_to_dict(candidate)
                if candidate is not None
                else None
            )
            for modifier, candidate in state.pending.best_by_modifier.items()
        }
        payload["recommended"] = (
            candidate_to_dict(state.pending.recommended)
            if state.pending.recommended is not None
            else None
        )
    return json.dumps(payload, indent=2, sort_keys=True)


def history_as_csv(history: Sequence[HistoryEntry]) -> str:
    """Return committed modification history as CSV."""

    output = io.StringIO()
    fieldnames = [
        "Round",
        "Modifier",
        "Card changes",
        "Predicted change",
        "Actual displayed change",
        "N",
        "T",
        "Full House E",
        "Straight E",
        "New E",
        "Prediction error", "Relative prediction error", "Method",
    ] + ["n_{}".format(RANK_LABELS[rank]) for rank in RANKS]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for entry in history:
        row = {
            "Round": entry.round_number,
            "Modifier": entry.modifier,
            "Card changes": entry.card_changes,
            "Predicted change": entry.predicted_change,
            "Actual displayed change": entry.actual_change,
            "N": entry.deck_size,
            "T": entry.total_hands,
            "Full House E": entry.expected_full_house,
            "Straight E": entry.expected_straight,
            "New E": entry.expected_total,
            "Prediction error": entry.prediction_error,
            "Relative prediction error": entry.relative_prediction_error,
            "Method": entry.method_id,
        }
        row.update(
            {
                "n_{}".format(RANK_LABELS[rank]): entry.deck[rank]
                for rank in RANKS
            }
        )
        writer.writerow(row)
    return output.getvalue()


def actions_as_csv(evaluation: Optional[RoundEvaluation]) -> str:
    """Return every legal action, including sequential working, as CSV."""

    output = io.StringIO()
    fieldnames = [
        "modifier",
        "ranked_position",
        "selection",
        "card_changes",
        "predicted_sensitivity_change",
        "actual_change_validation_only",
        "proposed_N",
        "proposed_T",
        "proposed_E_FH",
        "proposed_E_S",
        "proposed_E",
        "proposed_deck_json",
        "ordered_steps_json",
        "method_id",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    if evaluation is None:
        return output.getvalue()
    for modifier in ("Hanged Man", "Death", "Strength"):
        for position, candidate in enumerate(
            evaluation.rankings[modifier], start=1
        ):
            working = candidate_to_dict(candidate, include_working=True)
            writer.writerow(
                {
                    "modifier": modifier,
                    "ranked_position": position,
                    "selection": " -> ".join(
                        RANK_LABELS[rank] for rank in candidate.selection
                    ),
                    "card_changes": candidate.card_changes,
                    "predicted_sensitivity_change": candidate.predicted_change,
                    "actual_change_validation_only": candidate.actual_change,
                    "proposed_N": candidate.after_metrics.deck_size,
                    "proposed_T": candidate.after_metrics.total_hands,
                    "proposed_E_FH": candidate.after_metrics.expected_full_house,
                    "proposed_E_S": candidate.after_metrics.expected_straight,
                    "proposed_E": candidate.after_metrics.expected_total,
                    "proposed_deck_json": json.dumps(
                        working["proposed_deck"], sort_keys=True
                    ),
                    "ordered_steps_json": json.dumps(
                        working["steps"], sort_keys=True
                    ),
                    "method_id": candidate.method_id,
                }
            )
    return output.getvalue()


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "".join("<th>{}</th>".format(html.escape(str(item))) for item in headers)
    body = "".join(
        "<tr>{}</tr>".format(
            "".join("<td>{}</td>".format(html.escape(str(item))) for item in row)
        )
        for row in rows
    )
    return "<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(
        head, body
    )


def detailed_html_report(state: AppState) -> str:
    """Build a self-contained readable report of the current mathematics."""

    metrics = calculate_metrics(state.deck)
    sensitivities = calculate_sensitivities(state.deck)
    sensitivity_rows: List[Sequence[object]] = []
    for rank in RANKS:
        record = sensitivities[rank]
        sensitivity_rows.append(
            (
                RANK_LABELS[rank],
                state.deck[rank],
                format_decimal(record.full_house_derivative),
                format_decimal(record.straight_derivative),
                format_decimal(record.denominator_derivative),
                format_decimal(record.total_sensitivity),
                record.interpretation,
            )
        )

    candidate_sections = []
    if state.pending is not None:
        for modifier in ("Hanged Man", "Death", "Strength"):
            candidate = state.pending.best_by_modifier[modifier]
            if candidate is None:
                candidate_sections.append(
                    "<h3>{}</h3><p>No legal action.</p>".format(modifier)
                )
                continue
            step_rows = []
            for step in candidate.steps:
                changed = [
                    rank
                    for rank in RANKS
                    if step.prediction.delta[rank] != 0
                ]
                changed_working = "; ".join(
                    "{}: 0.5({:.3f} + {:.3f})({:+d}) ≈ {:.3f}".format(
                        RANK_LABELS[rank],
                        step.prediction.start_sensitivities[rank],
                        step.prediction.end_sensitivities[rank],
                        step.prediction.delta[rank],
                        step.prediction.trapezoidal_contributions[rank],
                    )
                    for rank in changed
                )
                step_rows.append(
                    (
                        step.step_number,
                        step.description,
                        json.dumps(_labelled_deck(step.deck_before)),
                        json.dumps(_labelled_deck(step.deck_after)),
                        changed_working,
                        "{:.3f}".format(
                            step.prediction.trapezoidal_prediction
                        ),
                    )
                )
            candidate_sections.append(
                "<h3>{}</h3><p><strong>{}</strong></p>".format(
                    html.escape(modifier), html.escape(candidate.card_changes)
                )
                + _table(
                    [
                        "Step",
                        "Action",
                        "Deck before",
                        "Deck after",
                        "Trapezoidal substitution",
                        "Predicted change",
                    ],
                    step_rows,
                )
                + "<p>Sequential atomic predictions sum to the ranking value: "
                "{:.3f}. Exact change (validation only): {:.3f}.</p>".format(
                    candidate.predicted_change,
                    candidate.actual_change,
                )
            )

    css = """
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 2rem auto;
           color: #172033; line-height: 1.45; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }
    th, td { border: 1px solid #ccd2dd; padding: .45rem; vertical-align: top; }
    th { background: #eef2f8; }
    code { background: #f2f4f7; padding: .15rem .3rem; }
    .notice { background: #fff5d9; border-left: 4px solid #d18b00; padding: 1rem; }
    """
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Balatro IA detailed calculations</title>
<style>{css}</style></head><body>
<h1>Balatro IA sensitivity report</h1>
<p class="notice">Modifier rankings use trapezoidal sensitivity predictions only.
Exact expected-score changes are shown only for validation.</p>
<h2>Current exact expected score</h2>
<p><code>N = {n}</code>, <code>T = C({n},5) = {t}</code>.</p>
<p><code>W_FH = {wfh}</code>, <code>W_S = {ws}</code>.</p>
<p><code>E_FH = {wfh}/{t} ≈ {efh:.3f}</code>,
<code>E_S = {ws}/{t} ≈ {es:.3f}</code>,
<code>E = (W_FH + W_S)/T ≈ {e:.3f}</code>.</p>
<h2>Formula record</h2>
<p><code>W_FH = sum[x != y] C(n_x,3) C(n_y,2) 4(40+3v_x+2v_y)</code></p>
<p><code>W_S = sum[a=2..10] (product[i=a..a+4] n_i) 4(30+sum v_i)</code></p>
<p><code>M_r = (dW_FH/dn_r + dW_S/dn_r - E dT/dn_r)/T</code></p>
<p><code>predicted Delta E = 0.5 sum_r [M_r(start)+M_r(end)] Delta n_r</code></p>
<h2>Current sensitivity substitutions</h2>
{sensitivity_table}
<h2>Staged modifier working</h2>
{candidate_sections}
</body></html>""".format(
        css=css,
        n=metrics.deck_size,
        t=metrics.total_hands,
        wfh=metrics.weighted_full_house,
        ws=metrics.weighted_straight,
        efh=metrics.expected_full_house,
        es=metrics.expected_straight,
        e=metrics.expected_total,
        sensitivity_table=_table(
            [
                "Rank",
                "n_r",
                "dW_FH/dn_r",
                "dW_S/dn_r",
                "dT/dn_r",
                "M_r",
                "Interpretation",
            ],
            sensitivity_rows,
        ),
        candidate_sections="".join(candidate_sections)
        or "<p>Evaluate the next modifier round to include candidate working.</p>",
    )
