# FraudGraphX 

**Dataset:** `data/raw/MOCK_DATA.csv`

## Quick Start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.data.preprocess --input data/raw/MOCK_DATA.csv --out data/processed/transactions.parquet
python -m src.features.build_all --input data/processed/transactions.parquet --out data/processed/transactions_with_features.parquet
python -m src.models.autoencoder --input data/processed/transactions_with_features.parquet --out data/processed/transactions_with_features.parquet
python -m src.models.train_xgb --input data/processed/transactions_with_features.parquet --model_path models/xgb_final.joblib
python -m src.explain.shap_report --model models/xgb_final.joblib --data data/processed/transactions_with_features.parquet --out reports/
python -m src.visual.heatmap --data data/processed/transactions_with_features.parquet --out reports/
streamlit run src/visual/dashboard_streamlit.py
# optional:
uvicorn src.serve.app_fastapi:app --reload
```


## Quick Start

```bash
# 0) Environment # Windows: .venv\Scripts\activate
python -m venv .venv && source .venv/bin/activate     
pip install --upgrade pip
pip install -r requirements.txt

# 1) Preprocess raw CSV -> Parquet
python -m src.data.preprocess \
  --input data/raw/MOCK_DATA.csv \
  --out data/processed/transactions.parquet

# 2) Build features (Parquet -> Parquet)
python -m src.features.build_all \
  --input data/processed/transactions.parquet \
  --out data/processed/transactions_with_features.parquet

# 3) (Optional) Autoencoder step (keeps same file and adds columns)
python -m src.models.autoencoder \
  --input data/processed/transactions_with_features.parquet \
  --out data/processed/transactions_with_features.parquet

# 4) Train XGBoost with flexible thresholding (F1 floor = 2%)
python -m src.models.train_xgb \
  --input data/processed/transactions_with_features.parquet \
  --target fraud \
  --model-out artifacts/model.xgb \
  --metrics-out artifacts/metrics.json \
  --threshold auto \
  --min-f1 0.02 \
  --auto-scale-pos-weight

# 5) Explainability / Visuals (adjust paths if your scripts expect model/joblib)
python -m src.explain.shap_report \
  --model artifacts/model.xgb \
  --data data/processed/transactions_with_features.parquet \
  --out reports/

python -m src.visual.heatmap \
  --data data/processed/transactions_with_features.parquet \
  --out reports/

# 6) Streamlit dashboard
streamlit run src/visual/dashboard_streamlit.py

# optional API
uvicorn src.serve.app_fastapi:app --reload
