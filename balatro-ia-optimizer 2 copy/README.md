# Balatro IA Deck Optimizer

A local, browser-based Python application for the simplified IB Mathematics AAHL
Balatro investigation.  Python performs all calculations; Streamlit provides the
interface.  Modifier choices are ranked by analytic sensitivity predictions,
while exact expected-score changes are reported separately for validation only.

## Project structure

- `app.py` - Streamlit interface and session-state controls.
- `math_model.py` - exact Full House/Straight combinatorics.
- `formatting.py` - three-decimal scores and precise validation-error displays.
- `sensitivity.py` - analytic partial derivatives and trapezoidal predictions.
- `modifiers.py` - legal Hanged Man, Death, and Strength enumeration.
- `algorithm.py` - round recommendation, apply, undo, reset, and auto-run state.
- `reporting.py` - JSON, CSV, and standalone HTML exports.
- `tests/` - mathematical, action, state, reporting, and UI smoke tests.

## 1. Create a virtual environment

Use Python 3.12. The dependencies are pinned to verified versions.

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The calculations need no network connection, account, API, or database after the
packages have been installed.

## 3. Run the tests

From the project folder:

```bash
python -m pytest -q tests
```

## 4. Start the Streamlit app

```bash
python -m streamlit run app.py
```

Streamlit prints a local address, normally `http://localhost:8501`.  Open that
address in a normal browser.  Stop the server with `Ctrl+C` in the terminal.

## How a modifier round works

1. **Evaluate Next Modifier Round** enumerates every legal action for all
   three modifiers from the same committed deck.  It stages three candidates and
   does not change the deck.
2. **Apply Recommended Modifier** commits the candidate with the largest
   sensitivity prediction.
3. **Apply:** and **Apply [modifier name]** override the cross-modifier recommendation while
   still selecting the best sensitivity-ranked action within the chosen modifier.
4. Undo, reset, and confirmed automatic-round controls are available in the
   compact **Reset, undo & automatic rounds** expander.

Hanged Man and Strength are evaluated sequentially. The first atomic card change
is applied, every sensitivity is recalculated, and the second atomic change is
then evaluated from that temporary deck. The two atomic predictions are added,
and that exact sum is used everywhere: ranking, recommendation, summary, and
history.

Two-card selections are unordered original-card rank pairs `(a,b)` with `a <= b`.
The lower starting rank is processed first; the computational order is fixed,
not optimized. Same-rank selections require two original cards. Strength cannot
select a newly created card again, and Ace wraps to 2. Death is one constant-size
atomic conversion to an existing target. Automatic rounds stop when the greatest
predicted gain is at most `1e-12`. Manual application remains available.

## Model and validation

Only Full Houses and non-wheel Straights score. Rank counts determine the exact
weighted score sums and the denominator `T = choose(N,5)`. The first-moment
score-count cache is retained; the displayed current score is always recalculated
from the actual deck.

Prediction error is signed `predicted_change - actual_change`. Relative error
is its absolute magnitude divided by the actual improvement, unless the actual
gain is within `1e-12` of zero; then the interface displays absolute error.
Scores and sensitivities display to three decimal places, while tiny validation
errors retain scientific notation. JSON and CSV functions keep full precision.

Round-zero and committed history retain rank counts, exact expected-score
components, predicted/actual gains, validation errors and calculation-method
identifiers. Undo/reset restores these snapshots. Compatible staged proposals
survive session updates; proposals with an obsolete deck or method are invalidated.
Legacy method identifiers are preserved. The interface has no download section.

The test fixture `tests/fixtures/optimizer_18_rounds.json` was captured before
the feature removal. Regression tests compare all 18 actions, counts, deck sizes,
hand counts, gains and expected-score components to that reference.

## 5. Deploy publicly with Streamlit Community Cloud

1. Create a GitHub repository and push this project, including
   `requirements.txt`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Select **Create app**, choose the repository and branch, and set the entrypoint
   file to `app.py`.
4. Click **Deploy**.  No secrets are required.  When the build finishes, Streamlit
   provides a public URL that can be shared with another person.

Community Cloud needs internet access to install packages during deployment, but
the running application's mathematical calculations do not call any external
service.
