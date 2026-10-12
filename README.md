# CUET Forecast Studio 🎓

An interactive **CUET UG cutoff explorer** for 2027–2030, built with FastAPI and a trained scikit-learn `HistGradientBoostingRegressor`. Features a responsive dark-themed dashboard, category filters, prediction cards, and interactive charts.

**Live site:** https://cuet-ug-cutoff-predictor.onrender.com

> **Important:** This project is an experimental decision-support tool, **not** an official CUET/NTA or university service. It cannot guarantee admission, and its future-year accuracy has not been independently validated.

## Features

- University, programme, college, category, seat type, and admission-round selectors
- Predicted cutoffs for 2027, 2028, 2029, and 2030 from the serialized model
- Comparison chart and four annual summary cards
- **Transparent what-if scenarios**: optional user-controlled annual percentage adjustment, default **0%**
- Model readiness endpoint, input validation, informative errors and responsive layout

## Why can 2027–2030 have identical predictions?

The uploaded model is a `HistGradientBoostingRegressor`, a tree-based algorithm. Trees generally do **not extrapolate smooth trends** beyond the observed training-year range. If future years follow the same terminal year split, the predicted score may be identical for 2027, 2028, 2029, and 2030. **This is a model limitation, not a broken chart.**

The dashboard now provides *two distinct outputs*:

1. **Raw ML predictions:** the model's actual per-year outputs, which might be flat.
2. **User-defined scenario:** `scenario(year) = ML_prediction(year) × (1 + annual_change_pct / 100)^(year - 2027)`. At the default **0%**, this exactly equals the raw predictions. A +1% assumption causes the displayed scenario to rise by 1% each year relative to each year's ML estimate; −1% decreases it. These percentages are **not estimated by the trained model**, and should never be presented as data-driven forecasts.

For genuinely model-derived year-specific trends, collect verified historical cutoff datasets with years, institutions and seat/category details, retrain with a forecasting-oriented method, and evaluate using *time-based holdouts* (e.g. train on earlier years, test on later years). Do not rely on random train/test accuracy for extrapolation.

## Project structure

```text
CUET-UG-Cutoff-Predictor/
├── app.py                         # FastAPI server and prediction logic
├── static/
│   └── index.html                 # HTML/CSS/JS dashboard + Chart.js
├── model_deployment_package.pkl   # Trained model, target encodings, metadata
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

Recommended: Python **3.11 or 3.12** and scikit-learn **1.6.1** (the model's serialization version).

```bash
git clone https://github.com/Armaan-30/CUET-UG-Cutoff-Predictor.git
cd CUET-UG-Cutoff-Predictor
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
python app.py
```

Visit **http://localhost:8000**.

### Important model notes

- `model_deployment_package.pkl` must be in the same directory as `app.py`.
- Only load **trusted** pickle files. Unpickling arbitrary files can execute code.
- The model package stores a regressor, target-encoding maps and a global fallback mean. Previously unseen feature combinations use the fallback, which may make predictions less dependable.
- `source_page` is a numeric training feature. The UI defaults to 0, but this is not independently validated as a meaningful input for every source.
- The forecast is in the same score units as the training target; verify the interpretation and maximum score before drawing admissions conclusions.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/api/health` | Check model file presence |
| GET | `/api/options` | Allowed dropdown values |
| POST | `/api/predict` | ML forecasts and optional scenario |

POST request example (replace placeholders with **exact keys** from `/api/options`):

```json
{
  "university_id": "<university ID>",
  "programme_id": "<matching programme ID>",
  "college": "<college>",
  "category": "<category>",
  "seat_type": "<seat type>",
  "round": "<round>",
  "source_page": 0,
  "annual_change_pct": 1
}
```

Response includes `forecasts` (raw model), `scenario_forecasts` (assumption-adjusted), `annual_change_pct`, `model_forecasts_flat`, `uses_fallback_encoding`, and `warnings`. Unsupported dropdown values and annual adjustments outside **−15% to +15%** return HTTP 422.

## Deploy to Render

1. Open [Render](https://dashboard.render.com/) and create a **Web Service** connected to this GitHub repository.
2. Select **Python** with root directory blank.
3. Set **Build Command**: `pip install -r requirements.txt`.
4. Set **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
5. Set `PYTHON_VERSION` to `3.11.11` if available and build with the dependencies in `requirements.txt`.
6. Deploy. On Render Free, the service can sleep during inactivity, causing slow first requests.

After each new commit to `main`, confirm Render's deploy has completed and test `/api/health`, `/api/options`, then a valid `/api/predict` request.

## Limitations and roadmap

- Future cutoff values are experimental and may remain flat.
- Scenario percentages are **hypothetical user assumptions**, not confidence intervals or learned growth rates.
- Real forecasting requires historical data across years, longitudinal validation, and appropriate calibration.
- Future improvements: university-specific historical trends, uncertainty estimates, comparable score scales, and backtested performance metrics.

## Technology

Python · FastAPI · NumPy · pandas · scikit-learn · HTML/CSS/JavaScript · Chart.js · Render

## Disclaimer

Independent educational project. Not affiliated with NTA, CUET, participating universities, or official counselling bodies. Check official cutoffs and admission notices before applying.
