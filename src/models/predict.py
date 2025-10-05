import argparse
import json
import numpy as np
import xgboost as xgb
import pandas as pd
# Load the trained model, score the input features,
# apply the saved threshold, print a summary, and optionally save prob/pred.
def main(args):
    booster = xgb.Booster()
    booster.load_model(args.model)

    with open(args.metrics, "r") as f:
        meta = json.load(f)
    threshold = float(meta.get("threshold", 0.5))
    features = meta.get("features", None)

    df = pd.read_parquet(args.input)
    if features is not None and all(f in df.columns for f in features):
        X = df[features]
    else:
        drop_cols = [c for c in ['fraud','transaction_id','timestamp','card_id','merchant'] if c in df.columns]
        X = df.drop(columns=drop_cols, errors='ignore')

    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
    probs = booster.predict(dtest)
    preds = (probs >= threshold).astype(int)

    out = {
        "threshold": threshold,
        "positives": int(preds.sum()),
        "total": int(len(preds)),
    }
    print(json.dumps(out, indent=2))

    if args.out is not None:
        pd.DataFrame({"prob": probs, "pred": preds}).to_parquet(args.out, index=False)
# Parse command-line arguments and run main().
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained model")
    ap.add_argument("--metrics", required=True, help="Path to training metrics JSON with threshold")
    ap.add_argument("--input", required=True, help="Parquet file with features")
    ap.add_argument("--out", default=None, help="Optional parquet output with prob/pred")
    args = ap.parse_args()
    main(args)
