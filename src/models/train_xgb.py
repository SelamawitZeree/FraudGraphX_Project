# src/models/train_xgb.py
import os, json, argparse, joblib, numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, classification_report

from ..utils.io import read_parquet

DROP_IDS = ['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
# Pick the operating threshold from the PR curve, honoring a precision floor/max threshold.
def pick_threshold(y_true, y_prob, precision_floor=0.0, max_thr=1.0, prefer="f1"):
    """Choose threshold from PR curve; optionally enforce a precision floor and a max threshold."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)      # thr len = len(prec)-1
    # Align arrays (ignore the first prec/rec point with no threshold)
    prec, rec, thr = prec[1:], rec[1:], thr
    f1 = (2*prec*rec) / (prec+rec+1e-9)

    mask = np.ones_like(thr, dtype=bool)
    if precision_floor > 0:
        mask &= (prec >= precision_floor)
    if max_thr < 1.0:
        mask &= (thr <= max_thr)

    if mask.any():
        i = int(np.argmax(f1[mask]))
        t = float(thr[mask][i])
    else:
        i = int(np.argmax(f1))
        t = float(thr[i])

    # choose metric to report
    chosen = {"threshold": t, "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
    return chosen
# Train/validate XGBoost, save model + feature list,
# choose threshold on validation, and print metrics.
def main(args):
    df = read_parquet(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found")

    y = df[args.target].astype(int).values
    X = df.drop(columns=[c for c in DROP_IDS if c in df.columns], errors='ignore')
    # ensure numeric-only for XGB
    X = X.select_dtypes(include=['number']).fillna(0.0)

    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )

    # class imbalance handling
    pos = max(int((ytr == 1).sum()), 1)
    neg = int((ytr == 0).sum())
    spw = neg / pos if args.auto_scale_pos_weight else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=args.eta,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        scale_pos_weight=spw,
        n_estimators=args.num_boost_round,
        n_jobs=args.nthread if args.nthread != 0 else None,
        random_state=args.seed,
        tree_method="hist",
        verbosity=1
    )

    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)

    # save exact training columns (API uses these)
    os.makedirs('models', exist_ok=True)
    with open('models/feature_cols.json', 'w') as f:
        json.dump(list(X.columns), f)

    joblib.dump(model, args.model_path)

    # choose threshold on validation
    yv_prob = model.predict_proba(Xv)[:, 1]
    chosen = pick_threshold(
        yv, yv_prob,
        precision_floor=args.precision_floor,
        max_thr=args.max_thr,
        prefer=args.prefer_metric
    )

    with open('models/threshold.json', 'w') as f:
        json.dump({"threshold": chosen["threshold"]}, f)

    # report metrics at chosen threshold
    yv_pred = (yv_prob >= chosen["threshold"]).astype(int)
    p = precision_score(yv, yv_pred, zero_division=0)
    r = recall_score(yv, yv_pred, zero_division=0)
    f = f1_score(yv, yv_pred, zero_division=0)
    print(f"[VAL] thr={chosen['threshold']:.4f}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")
    print(classification_report(yv, yv_pred, digits=4, zero_division=0))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--target', default='fraud')
    p.add_argument('--model_path', default='models/xgb_final.joblib')
    p.add_argument('--val_size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_boost_round', type=int, default=400)
    p.add_argument('--early_stopping_rounds', type=int, default=0)  # not used in XGBClassifier here
    p.add_argument('--eta', type=float, default=0.1)
    p.add_argument('--max_depth', type=int, default=6)
    p.add_argument('--subsample', type=float, default=0.9)
    p.add_argument('--colsample_bytree', type=float, default=0.9)
    p.add_argument('--min_child_weight', type=float, default=1.0)
    p.add_argument('--reg_lambda', type=float, default=1.0)
    p.add_argument('--reg_alpha', type=float, default=0.0)
    p.add_argument('--auto_scale_pos_weight', action='store_true')
    p.add_argument('--nthread', type=int, default=0)
    p.add_argument('--precision_floor', type=float, default=0.0,
                   help='Enforce minimum precision when picking threshold (e.g., 0.3)')
    p.add_argument('--max_thr', type=float, default=0.5,
                   help='Do not pick a threshold above this (e.g., 0.5 to keep recall healthy)')
    p.add_argument('--prefer_metric', choices=['f1','precision','recall'], default='f1')
    args = p.parse_args()
    main(args)
