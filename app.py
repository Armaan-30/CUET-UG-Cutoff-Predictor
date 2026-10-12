"""CUET UG forecast dashboard. Only load models from a trusted source."""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
FEATURES = ['year','source_page','university_id_encoded','programme_id_encoded','college_encoded','category_encoded','seat_type_encoded','round_encoded','college_prog_round_encoded','cat_seat_uni_encoded','full_combo_encoded']

@lru_cache(maxsize=1)
def load_package():
    model_path = ROOT / 'model_deployment_package.pkl'
    if not model_path.is_file():
        raise FileNotFoundError('Trained model is missing. Upload model_deployment_package.pkl to the repository root.')
    with model_path.open('rb') as f:
        pkg = pickle.load(f)
    for field in ('model', 'target_mappings', 'global_mean'):
        if field not in pkg:
            raise RuntimeError(f'Missing model key: {field}')
    return pkg

app = FastAPI(title='CUET Forecast Studio', version='1.0')
app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')

class PredictionRequest(BaseModel):
    university_id: str
    programme_id: str
    college: str
    category: str
    seat_type: str
    round: str
    source_page: float = Field(default=0, ge=-10000, le=10000)
    annual_change_pct: float = Field(default=0, ge=-15, le=15, description='User-defined annual scenario adjustment, not a learned model forecast')

def encoded_frame(r: PredictionRequest, year: int, pkg: dict):
    values = r.model_dump()
    mappings = pkg['target_mappings']
    global_mean = float(pkg['global_mean'])
    values.update({
        'college_prog_round': f"{r.college}_{r.programme_id}_{r.round}",
        'cat_seat_uni': f"{r.category}_{r.seat_type}_{r.university_id}",
        'full_combo': f"{r.college}_{r.programme_id}_{r.category}_{r.seat_type}_{r.round}",
    })
    row = {'year': year, 'source_page': r.source_page}
    unknown = []
    for field, mapping in mappings.items():
        raw = values[field]
        if raw not in mapping:
            unknown.append(field)
        row[f'{field}_encoded'] = float(mapping.get(raw, global_mean))
    return pd.DataFrame([row], columns=FEATURES), unknown

@app.get('/')
def home():
    return FileResponse(ROOT / 'static' / 'index.html')

@app.get('/api/health')
def health():
    model_present = (ROOT / 'model_deployment_package.pkl').is_file()
    return {'status': 'ready' if model_present else 'model_missing', 'model_available': model_present}


def require_package():
    try:
        return load_package()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail='Unable to load the trained model. Check its format and dependencies.') from exc


@app.get('/api/options')
def options():
    pkg = require_package()
    mapping = pkg['target_mappings']
    return {
        'universities': list(mapping['university_id'].keys()),
        'programmes': list(mapping['programme_id'].keys()),
        'colleges': list(mapping['college'].keys()),
        'categories': list(mapping['category'].keys()),
        'seat_types': list(mapping['seat_type'].keys()),
        'rounds': list(mapping['round'].keys()),
        'global_mean': round(float(pkg['global_mean']), 2),
        'model': type(pkg['model']).__name__,
    }

@app.post('/api/predict')
def predict(r: PredictionRequest):
    pkg = require_package()
    for field in ['university_id','programme_id','college','category','seat_type','round']:
        if getattr(r, field) not in pkg['target_mappings'][field]:
            raise HTTPException(status_code=422, detail=f'Unsupported {field} choice')
    if not r.programme_id.startswith(r.university_id + '-'):
        raise HTTPException(status_code=422, detail='Programme does not belong to selected university')
    predictions = []
    unknown = set()
    for year in range(2027, 2031):
        frame, missing = encoded_frame(r, year, pkg)
        unknown.update(missing)
        score = float(pkg['model'].predict(frame)[0])
        if not np.isfinite(score):
            raise HTTPException(status_code=500, detail='Model returned an invalid prediction')
        predictions.append({'year':year,'cutoff':round(score, 2)})
    # What-if projection: adjust each raw model prediction relative to the
    # 2027 baseline. This rate is an explicit user assumption, NOT learned
    # from historical cutoff data. At 0%, the raw model remains unchanged.
    rate = r.annual_change_pct / 100.0
    scenario = [
        {'year': item['year'], 'cutoff': round(item['cutoff'] * (1 + rate) ** (item['year'] - 2027), 2)}
        for item in predictions
    ]
    model_flat = len({item['cutoff'] for item in predictions}) == 1
    return {
        'forecasts': predictions,
        'scenario_forecasts': scenario,
        'annual_change_pct': r.annual_change_pct,
        'model_forecasts_flat': model_flat,
        'scenario_change': round(scenario[-1]['cutoff'] - scenario[0]['cutoff'], 2),
        'change': round(predictions[-1]['cutoff'] - predictions[0]['cutoff'],2),
        'unknown_combinations': sorted(unknown),
        'uses_fallback_encoding': bool(unknown),
        'warnings': [
            'These are extrapolated model estimates, not official cutoffs or admission guarantees.',
            'The training dataset and year range are not included in this package; future-year accuracy is unverified.',
            'Scenario projections use a user-specified annual percentage and are not predictions made by the trained model.',
        ]
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
