"""Dark, presentation-ready Streamlit interface for the Balatro IA model."""

import html
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithm import (
    AppState,
    apply_recommended,
    apply_selected_modifier,
    evaluate_state,
    new_state,
    pending_is_current,
    reconcile_state,
    reset_state,
    run_automatic_rounds,
    undo_last_modifier,
)
from math_model import RANKS, RANK_LABELS, calculate_metrics
from modifiers import GAIN_TOLERANCE, MODIFIER_ORDER, ModifierCandidate, next_rank
from formatting import format_decimal, format_error, format_integer
from sensitivity import calculate_sensitivities


APP_VERSION = "sequential-only-v5"
RANK_ORDER = [RANK_LABELS[rank] for rank in RANKS]
GREEN = "#4ade80"
BLUE = "#60a5fa"


st.set_page_config(
    page_title="Balatro IA Deck Optimizer",
    page_icon="🃏",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
    --panel: #151c2b;
    --panel-soft: #1a2335;
    --border: #344158;
    --text: #f8fafc;
    --muted: #b6c1d4;
    --green: #4ade80;
    --blue: #60a5fa;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; }
.metric-card, .candidate-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 0.8rem;
    color: var(--text);
    box-shadow: 0 6px 18px rgba(0,0,0,.16);
}
.metric-card { padding: .85rem 1rem; min-height: 104px; }
.metric-label {
    color: var(--muted);
    font-size: .82rem;
    font-weight: 650;
    letter-spacing: .015em;
    margin-bottom: .35rem;
}
.metric-value {
    color: var(--text);
    font-size: 1.65rem;
    font-weight: 750;
    line-height: 1.15;
}
.candidate-card { padding: 1.05rem; margin-bottom: .6rem; }
.candidate-name { color: var(--text); font-size: 1.2rem; font-weight: 750; }
.candidate-action { color: #dbe5f5; margin: .45rem 0 .7rem; }
.result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .55rem; }
.result-box {
    background: #101725;
    border: 1px solid var(--border);
    border-radius: .55rem;
    padding: .65rem .7rem;
}
.result-box.predicted { border-left: 3px solid var(--green); }
.result-box.validation { border-left: 3px solid var(--blue); }
.result-label { color: var(--muted); font-size: .75rem; line-height: 1.25; }
.result-value { color: var(--text); font-size: 1.18rem; font-weight: 750; margin-top: .2rem; }
.result-box.predicted .result-value { color: var(--green); }
.result-box.validation .result-value { color: var(--blue); }
.candidate-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .45rem;
    margin: .7rem 0;
}
.mini-stat { background: var(--panel-soft); border-radius: .45rem; padding: .48rem; }
.mini-label { color: var(--muted); font-size: .72rem; }
.mini-value { color: var(--text); font-weight: 700; margin-top: .15rem; }
.error-line { color: var(--muted); font-size: .82rem; margin-top: .6rem; }
.change-list {
    border-top: 1px solid var(--border);
    color: #dbe5f5;
    font-size: .88rem;
    line-height: 1.6;
    margin-top: .7rem;
    padding-top: .6rem;
}
.sensitivity-wrap {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: .75rem;
    overflow: hidden;
}
.sensitivity-table { border-collapse: collapse; width: 100%; color: var(--text); }
.sensitivity-table th {
    background: #1b2538;
    color: #d9e2f0;
    font-size: .8rem;
    padding: .58rem .75rem;
    text-align: right;
}
.sensitivity-table th:first-child, .sensitivity-table td:first-child { text-align: left; }
.sensitivity-table td {
    border-top: 1px solid #2b374d;
    padding: .43rem .75rem;
    text-align: right;
}
.negative { color: #fda4af; font-weight: 650; }
.positive { color: #86efac; font-weight: 650; }
.step-table { border-collapse: collapse; width: 100%; color: var(--text); margin: .6rem 0 1rem; }
.step-table th { background: #1b2538; color: #d9e2f0; }
.step-table th, .step-table td { border: 1px solid #344158; padding: .5rem .65rem; text-align: right; }
.step-table th:first-child, .step-table td:first-child { text-align: left; }
.apply-label { color: var(--muted); font-size: .82rem; font-weight: 700; margin-bottom: .25rem; }
div[data-testid="stExpander"] { border-color: #344158; background: #111827; }
div[data-testid="stAlert"] { color: #f8fafc; }
</style>
""",
    unsafe_allow_html=True,
)


def _initialise_session() -> None:
    if "app_state" not in st.session_state:
        st.session_state.app_state = new_state()
    elif st.session_state.get("app_version") != APP_VERSION:
        st.session_state.app_state = reconcile_state(st.session_state.app_state)
        st.session_state.pop("show_score_counts", None)
    else:
        # Detect a stale staged deck even without a software-version change.
        if st.session_state.app_state.pending is not None and not pending_is_current(st.session_state.app_state):
            st.session_state.app_state = reconcile_state(st.session_state.app_state)
    st.session_state.app_version = APP_VERSION
    st.session_state.setdefault("flash_message", "")
    st.session_state.setdefault("selected_modifier", "Hanged Man")


def _set_state(state: AppState, message: str = "") -> None:
    st.session_state.app_state = state
    st.session_state.flash_message = message


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">{}</div>'
        '<div class="metric-value">{}</div></div>'.format(label, value),
        unsafe_allow_html=True,
    )


def _rank_count_frame(deck: Mapping[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Rank": RANK_ORDER, "Count": [deck[rank] for rank in RANKS]}
    )


def _deck_figure(deck: Mapping[int, int]) -> go.Figure:
    frame = _rank_count_frame(deck)
    figure = go.Figure(
        go.Bar(
            x=frame["Rank"],
            y=frame["Count"],
            text=frame["Count"],
            textposition="outside",
            cliponaxis=False,
            marker={"color": "#7185f5", "line": {"color": "#aab6ff", "width": 1}},
            hovertemplate="Rank %{x}<br>Count %{y}<extra></extra>",
        )
    )
    figure.update_xaxes(
        title=None,
        categoryorder="array",
        categoryarray=RANK_ORDER,
        tickmode="array",
        tickvals=RANK_ORDER,
    )
    maximum = max(frame["Count"].max(), 1)
    figure.update_yaxes(title="Card count", range=[0, maximum + max(1.2, maximum * 0.25)])
    figure.update_layout(
        template="plotly_dark",
        height=390,
        margin={"l": 40, "r": 15, "t": 20, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.24,
    )
    return figure


def _sensitivity_table(deck: Mapping[int, int]) -> str:
    records = calculate_sensitivities(deck)
    rows = []
    for rank in RANKS:
        value = records[rank].total_sensitivity
        css_class = "positive" if value > 0 else "negative" if value < 0 else ""
        rows.append(
            "<tr><td>{}</td><td>{}</td><td class=\"{}\">{}</td></tr>".format(
                RANK_LABELS[rank],
                deck[rank],
                css_class,
                format_decimal(value, signed=True),
            )
        )
    return (
        '<div class="sensitivity-wrap"><table class="sensitivity-table">'
        '<thead><tr><th>Rank</th><th>Count <i>n</i><sub>r</sub></th>'
        '<th>Sensitivity <i>M</i><sub>r</sub></th></tr></thead><tbody>{}'
        "</tbody></table></div>".format("".join(rows))
    )


def _timeline(state: AppState) -> pd.DataFrame:
    rows = [
        {
            "Round": entry.round_number,
            "Total": entry.expected_total,
            "Straight": entry.expected_straight,
            "Full House": entry.expected_full_house,
        }
        for entry in state.history
    ]
    return pd.DataFrame(rows)


def _history_figure(frame: pd.DataFrame, column: str, title: str, color: str) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=frame["Round"],
            y=frame[column],
            mode="lines+markers",
            marker={"size": 10, "color": color},
            line={"width": 3, "color": color},
            hovertemplate="Round %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    rounds = frame["Round"].tolist()
    values = frame[column].tolist()
    minimum = min(values)
    maximum = max(values)
    padding = max((maximum - minimum) * 0.2, max(abs(maximum), 0.1) * 0.035)
    figure.update_xaxes(
        title="Round",
        tickmode="array",
        tickvals=rounds,
        range=[min(rounds) - 0.25, max(rounds) + 0.25],
    )
    figure.update_yaxes(range=[minimum - padding, maximum + padding], tickformat=".3f")
    figure.update_layout(
        title={"text": title, "font": {"size": 15}},
        template="plotly_dark",
        height=255,
        margin={"l": 45, "r": 15, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return figure


def _action_description(candidate: ModifierCandidate) -> str:
    first = candidate.selection[0]
    if candidate.modifier == "Hanged Man":
        second = candidate.selection[1]
        if first == second:
            return "Remove two cards of rank {}".format(RANK_LABELS[first])
        return "Remove rank {}, then rank {}".format(
            RANK_LABELS[first], RANK_LABELS[second]
        )
    if candidate.modifier == "Death":
        return "Convert {} → {}".format(
            RANK_LABELS[first], RANK_LABELS[candidate.selection[1]]
        )
    second = candidate.selection[1]
    first_target = next_rank(first)
    second_target = next_rank(second)
    if first == second:
        return "Increase two cards: {} → {}".format(
            RANK_LABELS[first], RANK_LABELS[first_target]
        )
    return "Increase {} → {}, then {} → {}".format(
        RANK_LABELS[first],
        RANK_LABELS[first_target],
        RANK_LABELS[second],
        RANK_LABELS[second_target],
    )


def _candidate_delta(candidate: ModifierCandidate) -> Dict[int, int]:
    start = candidate.steps[0].deck_before
    return {
        rank: candidate.proposed_deck[rank] - start[rank]
        for rank in RANKS
        if candidate.proposed_deck[rank] != start[rank]
    }


def _delta_html(delta: Mapping[int, int]) -> str:
    return ", &nbsp;".join(
        "Δ<i>n</i><sub>{}</sub>={:+d}".format(RANK_LABELS[rank], change)
        for rank, change in delta.items()
    )


def _candidate_card(candidate: ModifierCandidate) -> None:
    before = candidate.before_metrics
    after = candidate.after_metrics
    action = html.escape(_action_description(candidate))
    delta = _delta_html(_candidate_delta(candidate))
    transition = (
        "<div><i>N</i>: {} → {}</div><div><i>T</i>: {} → {}</div>".format(
            format_integer(before.deck_size),
            format_integer(after.deck_size),
            format_integer(before.total_hands),
            format_integer(after.total_hands),
        )
        if before.deck_size != after.deck_size
        else "<div><i>N</i> = {}</div><div><i>T</i> = {}</div>".format(
            format_integer(after.deck_size), format_integer(after.total_hands)
        )
    )
    st.markdown(
        """
<div class="candidate-card">
  <div class="candidate-name">{modifier}</div>
  <div class="candidate-action">{action}</div>
  <div class="result-grid">
    <div class="result-box predicted">
      <div class="result-label">Sensitivity-predicted change</div>
      <div class="result-value">{predicted}</div>
    </div>
    <div class="result-box validation">
      <div class="result-label">Exact change — validation only</div>
      <div class="result-value">{actual}</div>
    </div>
  </div>
  <div class="candidate-metrics">
    <div class="mini-stat"><div class="mini-label"><i>E</i><sub>FH</sub></div><div class="mini-value">{efh}</div></div>
    <div class="mini-stat"><div class="mini-label"><i>E</i><sub>S</sub></div><div class="mini-value">{es}</div></div>
    <div class="mini-stat"><div class="mini-label"><i>E</i></div><div class="mini-value">{total}</div></div>
  </div>
  <div class="error-line" title="Predicted minus actual: {precise_error}">Prediction error: {error}</div>
  <div class="change-list"><div>{delta}</div>{transition}</div>
</div>
""".format(
            modifier=html.escape(candidate.modifier),
            action=action,
            predicted=format_decimal(candidate.predicted_change, signed=True),
            actual=format_decimal(candidate.actual_change, signed=True),
            efh=format_decimal(after.expected_full_house),
            es=format_decimal(after.expected_straight),
            total=format_decimal(after.expected_total),
            delta=delta,
            transition=transition,
            error=format_error(candidate.prediction_error, candidate.relative_prediction_error),
            precise_error="{:+.15g}".format(candidate.prediction_error),
        ),
        unsafe_allow_html=True,
    )


def _step_table(candidate: ModifierCandidate, step_index: int) -> str:
    step = candidate.steps[step_index]
    ranks = [rank for rank in RANKS if step.prediction.delta[rank] != 0]
    rows = []
    for rank in ranks:
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:+d}</td></tr>".format(
                RANK_LABELS[rank],
                format_decimal(step.prediction.start_sensitivities[rank], signed=True),
                format_decimal(step.prediction.end_sensitivities[rank], signed=True),
                step.prediction.delta[rank],
            )
        )
    return (
        '<table class="step-table"><thead><tr><th>Rank</th>'
        '<th><i>M</i><sub>start</sub></th><th><i>M</i><sub>end</sub></th>'
        '<th>Δ<i>n</i><sub>r</sub></th></tr></thead><tbody>{}'
        "</tbody></table>".format("".join(rows))
    )


def _step_equation(candidate: ModifierCandidate, step_index: int) -> str:
    step = candidate.steps[step_index]
    ranks = [rank for rank in RANKS if step.prediction.delta[rank] != 0]
    lines = [r"\begin{aligned}"]
    for index, rank in enumerate(ranks):
        prefix = r"\widehat{\Delta E}_{%d} &=" % (step_index + 1) if index == 0 else r"&\quad +"
        lines.append(
            prefix
            + r"\frac12\left(M_{%s}^{\mathrm{start}}+M_{%s}^{\mathrm{end}}\right)(%+d)\\"
            % (RANK_LABELS[rank], RANK_LABELS[rank], step.prediction.delta[rank])
        )
    for index, rank in enumerate(ranks):
        prefix = r"&\approx" if index == 0 else r"&\quad +"
        lines.append(
            prefix
            + r"\frac12\left(%s%s\right)(%+d)\\"
            % (
                format_decimal(step.prediction.start_sensitivities[rank], signed=True),
                format_decimal(step.prediction.end_sensitivities[rank], signed=True),
                step.prediction.delta[rank],
            )
        )
    lines.append(
        r"&\approx%s\end{aligned}"
        % format_decimal(step.prediction.trapezoidal_prediction, signed=True)
    )
    return "".join(lines)


def _step_update_equation(candidate: ModifierCandidate, step_index: int) -> str:
    step = candidate.steps[step_index]
    changed = [rank for rank in RANKS if step.deck_before[rank] != step.deck_after[rank]]
    before_metrics = calculate_metrics(step.deck_before)
    after_metrics = calculate_metrics(step.deck_after)
    lines = [r"\begin{aligned}"]
    for rank in changed:
        lines.append(
            r"n_{%s}&:%d\rightarrow%d\\"
            % (RANK_LABELS[rank], step.deck_before[rank], step.deck_after[rank])
        )
    lines.append(r"N&:%d\rightarrow%d\\" % (before_metrics.deck_size, after_metrics.deck_size))
    if before_metrics.deck_size != after_metrics.deck_size:
        lines.append(
            r"T&:\binom{%d}{5}\rightarrow\binom{%d}{5}\\"
            % (before_metrics.deck_size, after_metrics.deck_size)
        )
    else:
        latex_total = format_integer(after_metrics.total_hands).replace(",", "{,}")
        lines.append(r"T&=%s\\" % latex_total)
    lines.append(r"\end{aligned}")
    return "".join(lines)


def _candidate_working(candidate: ModifierCandidate) -> None:
    for index, step in enumerate(candidate.steps):
        st.markdown("#### Step {} — {}".format(index + 1, step.description))
        st.latex(_step_equation(candidate, index))
        st.markdown(_step_table(candidate, index), unsafe_allow_html=True)
        st.latex(_step_update_equation(candidate, index))
    atomic_values = [step.prediction.trapezoidal_prediction for step in candidate.steps]
    total_expression = " + ".join(
        r"\left({}\right)".format(format_decimal(value, signed=True))
        for value in atomic_values
    )
    st.latex(
        r"\widehat{\Delta E}_{\mathrm{%s}}\approx%s\approx%s"
        % (
            candidate.modifier.replace(" ", r"\ "),
            total_expression,
            format_decimal(candidate.predicted_change, signed=True),
        )
    )


def _history_frame(state: AppState) -> pd.DataFrame:
    rows = [
        {
            "Round": entry.round_number,
            "Modifier": entry.modifier,
            "Card changes": entry.card_changes,
            "Predicted change": format_decimal(entry.predicted_change, signed=True) if entry.round_number else "—",
            "Actual change": format_decimal(entry.actual_change, signed=True) if entry.round_number else "—",
            "Prediction error": format_error(entry.prediction_error, entry.relative_prediction_error) if entry.round_number else "—",
            "N": format_integer(entry.deck_size),
            "T": format_integer(entry.total_hands),
            "New E": format_decimal(entry.expected_total),
            "Method": entry.method_id,
        }
        for entry in state.history
    ]
    return pd.DataFrame(rows)


_initialise_session()
state: AppState = st.session_state.app_state
metrics = calculate_metrics(state.deck)

st.title("Balatro IA Deck Optimizer")
st.caption(
    "Model: only Full Houses and non-wheel Straights score."
)

metric_columns = st.columns(6)
metric_values = [
    ("Modifier round", str(state.round_number)),
    ("Deck size <i>N</i>", format_integer(metrics.deck_size)),
    ("Five-card hands <i>T</i>", format_integer(metrics.total_hands)),
    ("Full House <i>E</i><sub>FH</sub>", format_decimal(metrics.expected_full_house)),
    ("Straight <i>E</i><sub>S</sub>", format_decimal(metrics.expected_straight)),
    ("Total expected <i>E</i>", format_decimal(metrics.expected_total)),
]
for column, (label, value) in zip(metric_columns, metric_values):
    with column:
        _metric_card(label, value)

st.subheader("Current Deck")
deck_column, sensitivity_column = st.columns([1.7, 1])
with deck_column:
    st.plotly_chart(
        _deck_figure(state.deck),
        width="stretch",
        config={"displayModeBar": False},
        key="deck_rank_chart",
    )
with sensitivity_column:
    st.markdown("**Current rank sensitivities**")
    st.markdown(_sensitivity_table(state.deck), unsafe_allow_html=True)

st.subheader("Historical expected scores")
timeline = _timeline(state)
chart_columns = st.columns(3)
history_specs: Sequence[Tuple[str, str, str]] = (
    ("Total", "Total expected score E", "#a78bfa"),
    ("Straight", "Straight contribution E_S", GREEN),
    ("Full House", "Full House contribution E_FH", BLUE),
)
for column, (field, title, color) in zip(chart_columns, history_specs):
    with column:
        st.plotly_chart(
            _history_figure(timeline, field, title, color),
            width="stretch",
            config={"displayModeBar": False},
            key="history_{}".format(field.lower().replace(" ", "_")),
        )

st.subheader("Next modifier round")
can_apply = state.pending is not None and state.pending.recommended is not None
recommended_name = (
    state.pending.recommended.modifier
    if state.pending is not None and state.pending.recommended is not None
    else None
)
recommended_label = "Apply Recommended Modifier"
if recommended_name:
    recommended_label += " ({})".format(recommended_name)

control_evaluate, control_recommended, control_manual = st.columns([1, 1.2, 1.2])
with control_evaluate:
    if st.button(
        "Evaluate Next Modifier Round",
        key="evaluate_round",
        type="primary",
        width="stretch",
    ):
        with st.spinner("Evaluating candidates..."):
            evaluated = evaluate_state(state)
        if evaluated.pending is not None and evaluated.pending.recommended is not None:
            st.session_state.selected_modifier = evaluated.pending.recommended.modifier
        _set_state(evaluated, "Round evaluated. The deck is unchanged.")
        st.rerun()
with control_recommended:
    if st.button(
        recommended_label,
        key="apply_recommended",
        disabled=not can_apply,
        width="stretch",
    ):
        _set_state(apply_recommended(state), "Applied the recommended modifier.")
        st.rerun()
with control_manual:
    st.markdown('<div class="apply-label">Apply:</div>', unsafe_allow_html=True)
    legal_modifiers = (
        [
            name
            for name in MODIFIER_ORDER
            if state.pending is not None
            and state.pending.best_by_modifier.get(name) is not None
        ]
        if state.pending is not None
        else list(MODIFIER_ORDER)
    )
    if st.session_state.get("selected_modifier") not in legal_modifiers:
        st.session_state.selected_modifier = recommended_name or legal_modifiers[0]
    selected_modifier = st.selectbox(
        "Apply:",
        legal_modifiers,
        label_visibility="collapsed",
        key="selected_modifier",
        disabled=state.pending is None,
    )
    if st.button(
        "Apply {}".format(selected_modifier),
        key="apply_selected",
        disabled=state.pending is None,
        width="stretch",
    ):
        _set_state(
            apply_selected_modifier(state, selected_modifier),
            "Applied {}.".format(selected_modifier),
        )
        st.rerun()

if st.session_state.flash_message:
    st.success(st.session_state.flash_message)
    st.session_state.flash_message = ""

if state.pending is not None and state.pending.recommended is not None:
    recommendation = state.pending.recommended
    message = (
        "**Recommended: {}**  \n{}  \n"
        r"$\widehat{{\Delta E}}_{{\mathrm{{sens}}}}\approx{}$".format(
            recommendation.modifier,
            _action_description(recommendation),
            format_decimal(recommendation.predicted_change, signed=True),
        )
    )
    if recommendation.predicted_change > GAIN_TOLERANCE:
        st.success(message)
    else:
        st.warning("No positive predicted gain. Automatic rounds stop here.")
    if state.pending.tie_break_explanation:
        st.caption(state.pending.tie_break_explanation)
    if recommendation.predicted_change * recommendation.actual_change < 0:
        st.warning("Prediction and validation have opposite signs.")
    st.markdown("### Modifier candidates")
    for modifier in MODIFIER_ORDER:
        candidate = state.pending.best_by_modifier[modifier]
        if candidate is None:
            st.warning("No legal {} action.".format(modifier))
            continue
        _candidate_card(candidate)
        with st.expander(
            "{} — Show sensitivity working".format(modifier), expanded=False
        ):
            _candidate_working(candidate)

st.subheader("Modification history")
st.dataframe(_history_frame(state), hide_index=True, width="stretch")

with st.expander("Reset, undo & automatic rounds", expanded=False):
    reset_column, undo_column, count_column, confirmation_column, run_column = st.columns(
        [1, 1, .75, 1.5, 1.2]
    )
    with reset_column:
        if st.button(
            "Reset to Standard Deck", key="reset_standard", width="stretch"
        ):
            _set_state(reset_state(state), "Reset to the standard 52-card deck.")
            st.rerun()
    with undo_column:
        if st.button(
            "Undo Last Modifier",
            key="undo_modifier",
            disabled=not bool(state.snapshots),
            width="stretch",
        ):
            _set_state(undo_last_modifier(state), "Undid the last modifier.")
            st.rerun()
    with count_column:
        auto_rounds = st.number_input(
            "Rounds",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key="auto_round_count",
        )
    with confirmation_column:
        auto_confirm = st.checkbox(
            "Confirm automatic application",
            key="auto_confirm",
            help="Each round recalculates all actions and commits the recommendation.",
        )
    with run_column:
        if st.button(
            "Run Several Rounds Automatically",
            key="auto_rounds_button",
            disabled=not auto_confirm,
            width="stretch",
        ):
            with st.spinner("Evaluating and committing sequential rounds..."):
                updated = run_automatic_rounds(state, int(auto_rounds))
            _set_state(
                updated,
                "Automatically applied {} round(s).".format(
                    updated.round_number - state.round_number
                ),
            )
            st.rerun()
