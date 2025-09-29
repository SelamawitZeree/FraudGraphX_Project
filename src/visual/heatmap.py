import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.io import read_parquet

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    df = read_parquet(a.data).copy()

    # hour for rows
    df['hour'] = df['timestamp'].dt.hour.astype(int)

    # --- robust amount binning ---
    nuniq = int(df['amount'].nunique())
    bins_q = max(1, min(10, nuniq))  # cannot exceed number of unique values
    try:
        df['amount_bin'] = pd.qcut(df['amount'], q=bins_q, duplicates='drop')
    except Exception:
        # fallback: equal-width bins
        a_min, a_max = float(df['amount'].min()), float(df['amount'].max())
        if a_min == a_max:  # constant series -> widen slightly
            a_min -= 1e-6
            a_max += 1e-6
        bins_c = min(10, max(2, nuniq or 2))
        df['amount_bin'] = pd.cut(df['amount'], bins=bins_c, include_lowest=True)

    # --- heatmap 1: fraud rate by hour vs amount bin ---
    pv = df.pivot_table(
        index='hour', columns='amount_bin', values='fraud',
        aggfunc='mean', observed=False  # silence pandas future warning
    )

    plt.figure()
    if pv.size == 0 or pv.dropna(how='all').empty:
        # Fallback: show fraud rate by hour only (1 x 24 heatmap)
        hourly = df.groupby('hour')['fraud'].mean().reindex(range(24), fill_value=0)
        pv_fallback = pd.DataFrame([hourly.values], index=['fraud_rate'], columns=list(range(24)))
        sns.heatmap(pv_fallback, cbar=True)
        plt.title('Fraud Rate by Hour (fallback)')
    else:
        sns.heatmap(pv, annot=False)
        plt.title('Fraud Rate by Hour vs Amount Bin')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, 'fraud_heatmap_hour_amount.png'))

    # --- heatmap 2: feature correlation (robust) ---
    num = df.select_dtypes(include=['number'])
    plt.figure()
    if num.shape[1] >= 2:
        sns.heatmap(num.corr(), annot=False)
        plt.title('Feature Correlation Heatmap')
    else:
        plt.text(0.5, 0.5, 'Not enough numeric features for correlation.',
                 ha='center', va='center')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, 'feature_correlation_heatmap.png'))