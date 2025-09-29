import argparse, os, joblib, shap
from ..utils.io import read_parquet
import matplotlib.pyplot as plt
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); a=ap.parse_args()
    os.makedirs(a.out, exist_ok=True); model=joblib.load(a.model); df=read_parquet(a.data)
    drop=['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
    X=df.drop(columns=[c for c in drop if c in df.columns]).fillna(0.0)
    expl=shap.TreeExplainer(model); sv=expl.shap_values(X)
    plt.figure(); shap.summary_plot(sv, X, show=False); plt.tight_layout(); plt.savefig(os.path.join(a.out,'shap_summary.png'))
    print('Saved SHAP summary to', os.path.join(a.out,'shap_summary.png'))
