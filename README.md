# CUET Forecast Studio

A responsive CUET UG cutoff prediction dashboard built with FastAPI and scikit-learn, generating predictions for 2027–2030 from a trained HistGradientBoostingRegressor.

## Local setup

Use Python 3.10–3.12 (the model was serialized with scikit-learn 1.6.1).

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000.

The application requires `model_deployment_package.pkl` at the repository root. Do not execute untrusted pickle files.

## Forecasting caveats

The package uses nine target encodings, and `source_page` is a numeric model feature whose default of 0 has not been independently validated. The model cannot necessarily extrapolate trends beyond the years in its training data. It may predict identical cutoffs for 2027–2030; such results are shown without invented variations.
