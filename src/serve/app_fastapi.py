from fastapi import FastAPI
import joblib, pandas as pd, numpy as np, os, json
from src.utils.io import read_parquet

app = FastAPI(title='FraudGraphX API')
_model, _feat_cols, _default_threshold = None, None, 0.5

def load_model(model_path='models/xgb_final.joblib',
               data_example='data/processed/transactions_with_features.parquet'):
    global _model, _feat_cols, _default_threshold
    if _model is None:
        _model = joblib.load(model_path)

        # exact training columns (created by train_xgb.py)
        feat_json = 'models/feature_cols.json'
        if os.path.exists(feat_json):
            with open(feat_json, 'r') as f:
                _feat_cols = json.load(f)
        else:
            df = read_parquet(data_example)
            drop = ['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
            _feat_cols = [c for c in df.columns if c not in drop]

        # default threshold if present (see step 2)
        thr_json = 'models/threshold.json'
        if os.path.exists(thr_json):
            _default_threshold = json.load(open(thr_json))['threshold']
    return _model, _feat_cols, _default_threshold

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/score')
def score(txn: dict, threshold: float | None = None):
    model, feat_cols, default_thr = load_model()
    thr = float(threshold) if threshold is not None else float(default_thr)
    x = np.array([[txn.get(c, 0.0) for c in feat_cols]], dtype=float)
    proba = float(model.predict_proba(x)[0, 1])
    return {'score': proba, 'threshold': thr, 'is_fraud': proba >= thr}
