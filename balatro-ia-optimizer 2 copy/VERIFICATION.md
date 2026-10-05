# Balatro IA optimizer verification — 4 October 2026

Standard deviation, second moments, variance, Blind/Ante success probabilities,
their UI panels, export/history fields, calculations, formatting and dedicated
tests were removed. The shared `prediction_error` helper moved from the deleted
`risk.py` into `sensitivity.py`; relative validation error percentages remain.

The exact scoring model, analytic sensitivities, sequential endpoint averaging,
unordered original-card pairs, lower-rank-first convention, deterministic
recommendation, Ace-to-2 progression, validation results, dark styling, vertical
modifier panels and workflow controls are preserved. First-moment score-count
caching remains; squared-score work was removed. The README reflects this scope.

## Numerical preservation

The baseline and 18 recommended rounds were captured before editing in
`tests/fixtures/optimizer_18_rounds.json`. Afterwards, every action, selected pair,
full rank-count vector, N, T, predicted gain, actual gain and expected-score
component matched. The maximum absolute numerical difference was **0.0**.

| Round | Action | N | T | E_FH | E_S | E |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Baseline | 52 | 2,598,960 | 0.441 | 0.963 | 1.404 |
| 1 | Remove two 2s | 50 | 2,118,760 | 0.477 | 1.133 | 1.610 |
| 14 | Finish removing ranks 2–8 | 24 | 42,504 | 6.098 | 15.419 | 21.517 |
| 15 | Two 9s → 10s | 24 | 42,504 | 8.910 | 17.418 | 26.329 |
| 16 | Remaining 9s → 10s | 24 | 42,504 | 17.746 | 15.612 | 33.357 |
| 17 | Two Kings → Aces | 24 | 42,504 | 24.026 | 11.709 | 35.735 |
| 18 | Remaining Kings → Aces | 24 | 42,504 | 43.115 | 0.000 | 43.115 |

The first predicted gain is 0.2058784621083486, its exact validation gain is
0.20596230807131066, and the final expected score is 43.11537737624694.
Final nonzero counts: eight 10s, four Jacks, four Queens and eight Aces.
These values are calculated, not forced by benchmark logic.

## Tests and local startup

**90 tests passed** (`python -m pytest -q tests`) under Python 3.12.14 with
Streamlit 1.64.0, pandas 2.3.3, Plotly 6.9.0 and pytest 8.4.2.
The suite covers analytic sensitivities, sequential calculations, legal selection,
the 18-round reference, first-moment caching, validation errors, nonmutating
evaluation, apply/undo/reset, automatic stopping, stale proposals, legacy state
migration, retained reports and Streamlit interaction contracts.

Legacy objects with obsolete fields are rebuilt using the retained schema.
Counts, rounds, scores, gains, recorded errors, method identifiers, undo snapshots
and compatible staged proposals are preserved. A fresh-process test forbids
importing the removed module while running the optimizer and reports.

The app starts locally at http://localhost:8501; `/` and `/_stcore/health` return
HTTP 200, with health body `ok`. It is bound to the loopback address.
No public deployment or publication was performed.

Chromium screenshot inspection was blocked by the macOS sandbox denying its
MachPortRendezvous service. Automated UI tests verify six remaining metrics,
four charts, baseline history, rank order, retained calculations, validation-error
precision, feature absence, controls and session migration. Actual rendered
appearance, empty gaps and formula clipping remain visually unchecked.
