
import os
import json
import argparse
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb

# Optional project-specific parquet reader
try:
    from ..utils.io import read_parquet  # type: ignore
except Exception:
    import pandas as pd
    def read_parquet(path):
        return pd.read_parquet(path)

def choose_threshold(y_true, y_prob, min_f1=0.02, prefer="f1"):
    """
    Pick a decision threshold on a validation set.
    Strategy: sweep thresholds produced by precision_recall_curve, compute metrics,
    and choose the index that maximizes the preferred metric (default F1).
    Returns a dict with threshold, precision, recall, f1, and whether min_f1 is satisfied.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Align sizes: thresholds has len = len(precisions) - 1
    thresholds = np.concatenate([thresholds, [1.0]])
    f1s = np.zeros_like(precisions)
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        denom = (p + r)
        f1s[i] = 0.0 if denom == 0 else (2 * p * r / denom)

    if prefer == "precision":
        idx = int(np.argmax(precisions))
    elif prefer == "recall":
        idx = int(np.argmax(recalls))
    else:
        idx = int(np.argmax(f1s))

    chosen = {
        "threshold": float(thresholds[idx]),
        "precision": float(precisions[idx]),
        "recall": float(recalls[idx]),
        "f1": float(f1s[idx]),
        "min_f1_satisfied": bool(f1s[idx] >= min_f1),
    }
    return chosen

def main(args):
    df = read_parquet(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in columns: {list(df.columns)[:20]}")

    y = df[args.target].astype(int).values

    # Drop common non-feature columns if present
    drop_cols = set([args.target])
    for c in ['transaction_id', 'timestamp', 'card_id', 'merchant']:
        if c in df.columns:
            drop_cols.add(c)
    X = df.drop(columns=list(drop_cols), errors='ignore')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X.columns))

    # Handle imbalance
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
   # spw = max((neg / max(pos, 1)), 1.0) if args.auto_scale_pos_weight else 1.0
    spw = max(1.0, (neg / max(pos, 1)) ** 0.5) if args.auto_scale_pos_weight else 1.0

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "scale_pos_weight": spw,
        "verbosity": 1,
        "nthread": args.nthread,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False
    )

    y_val_prob = booster.predict(dval)

    # === Choose threshold ===
    if args.threshold != "auto":
        # Fixed, user-specified threshold
        threshold = float(args.threshold)
        y_pred = (y_val_prob >= threshold).astype(int)
    else:
        # Auto: best F1 among thresholds <= 0.5 that also meet a precision floor
        prec, rec, thr = precision_recall_curve(y_val, y_val_prob)  # thr aligns with prec[1:], rec[1:]
        f1 = (2 * prec[1:] * rec[1:]) / (prec[1:] + rec[1:] + 1e-9)

        PRECISION_FLOOR = 0.32  # try 0.32–0.35 if needed
        mask = (thr <= 0.5) & (prec[1:] >= PRECISION_FLOOR)

        if mask.any():
            i = np.argmax(f1[mask])
            threshold = float(thr[mask][i])
        else:
            threshold = 0.5  # safest strict fallback

        y_pred = (y_val_prob >= threshold).astype(int)

    # Compute metrics at the chosen threshold
    p = float(precision_score(y_val, y_pred, zero_division=0))
    r = float(recall_score(y_val, y_pred, zero_division=0))
    f = float(f1_score(y_val, y_pred, zero_division=0))

    chosen = {
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f,
        "min_f1_satisfied": (f >= args.min_f1),
    }

    booster.save_model(args.model_out)

    meta = {
        "threshold": chosen["threshold"],
        "val_metrics": {
            "precision": chosen["precision"],
            "recall": chosen["recall"],
            "f1": chosen["f1"],
            "min_f1_satisfied": chosen["min_f1_satisfied"],
        },
        "params": params,
        "best_iteration": int(getattr(booster, 'best_iteration', 0) or getattr(booster, 'best_ntree_limit', 0)),
        "features": list(X.columns),
        "notes": "Threshold chosen on validation set with flexible strategy."
    }
    os.makedirs("models", exist_ok=True)
    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": meta["threshold"]}, f)

    with open(args.metrics_out, "w") as f:
        json.dump(meta, f, indent=2)

    if not meta["val_metrics"]["min_f1_satisfied"]:
        print(f"[WARN] Best validation F1 ({meta['val_metrics']['f1']:.4f}) did not reach the desired minimum ({args.min_f1}).")
    else:
        print(f"[OK] Validation F1 {meta['val_metrics']['f1']:.4f} meets/exceeds minimum {args.min_f1}.")

    print("Validation report at chosen threshold:")
    y_pred = (y_val_prob >= meta["threshold"]).astype(int)
    print(classification_report(y_val, y_pred, digits=4, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model with flexible threshold selection.")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--target", type=str, default="fraud", help="Target column name (default: fraud)")
    parser.add_argument("--model-out", type=str, default="artifacts/model.xgb", help="Where to save model")
    parser.add_argument("--metrics-out", type=str, default="artifacts/metrics.json", help="Where to save metrics/metadata JSON")
    parser.add_argument("--threshold", type=str, default="auto", help="'auto' to pick best on val set, or a float like 0.5")
    parser.add_argument("--min-f1", type=float, default=0.02, help="Desired minimum F1 (default: 0.02 = 2%)")
    parser.add_argument("--prefer-metric", type=str, choices=["f1","precision","recall"], default="f1",
                        help="Metric to optimize when choosing threshold (default: f1)")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-boost-round", type=int, default=400, help="Boosting rounds")
    parser.add_argument("--early-stopping-rounds", type=int, default=30, help="Early stopping on validation")
    parser.add_argument("--eta", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth")
    parser.add_argument("--subsample", type=float, default=0.9, help="Row subsample")
    parser.add_argument("--colsample-bytree", type=float, default=0.9, help="Feature subsample")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="Min child weight")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization")
    parser.add_argument("--auto-scale-pos-weight", action="store_true", help="Auto set scale_pos_weight = neg/pos")
    parser.add_argument("--nthread", type=int, default=0, help="Threads; 0 lets XGBoost choose")
    args = parser.parse_args()
    main(args)
