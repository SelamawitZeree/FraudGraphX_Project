import pandas as pd
import numpy as np
from datetime import datetime, timedelta

LABELS = ['fraud', 'is_fraud', 'class', 'label', 'target']
AMTS   = ['amount', 'amt', 'transaction_amount', 'value', 'price']
TIMES  = ['timestamp', 'time', 'datetime', 'date', 'transaction_time']

POSITIVE_TOKENS = {'1', 'true', 't', 'yes', 'y', 'fraud', 'positive'}

def infer_cols(df: pd.DataFrame):
    cols = {'label': None, 'amount': None, 'timestamp': None}
    lower_map = {c: c.lower() for c in df.columns}

    def find(cands):
        for want in cands:
            for orig, low in lower_map.items():
                if low == want:
                    return orig
        return None

    cols['label'] = find(LABELS)
    cols['amount'] = find(AMTS)
    cols['timestamp'] = find(TIMES)
    return cols

def _coerce_label_series(s: pd.Series) -> pd.Series:
    """Map arbitrary label series to {0,1}."""
    if s.dtype.kind in ('i', 'u', 'f'):
        # numeric-like
        y = pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
        return (y > 0).astype(int)
    # string-like
    sl = s.astype(str).str.strip().str.lower()
    return sl.isin(POSITIVE_TOKENS).astype(int)

def synthesize(df: pd.DataFrame):
    cols = infer_cols(df)
    out = df.copy()

    # --- amount ---
    if cols['amount']:
        out['amount'] = pd.to_numeric(out[cols['amount']], errors='coerce').fillna(0.0)
    else:
        # make a Series, not a scalar, so later ops (quantile) work consistently
        out['amount'] = pd.Series(10.0, index=out.index)

    # --- timestamp ---
    if cols['timestamp']:
        out['timestamp'] = pd.to_datetime(out[cols['timestamp']], errors='coerce')
        mask = out['timestamp'].isna()
        if mask.any():
            base = datetime(2024, 1, 1)
            out.loc[mask, 'timestamp'] = [base + timedelta(minutes=i) for i in range(mask.sum())]
    else:
        base = datetime(2024, 1, 1)
        out['timestamp'] = [base + timedelta(minutes=i) for i in range(len(out))]

    # --- ids (ensure presence as strings) ---
    if 'transaction_id' not in out.columns:
        out['transaction_id'] = range(len(out))
    for col, pref, mod in [('card_id','card_',1000),
                           ('merchant_id','m_',50),
                           ('device_id','d_',200),
                           ('ip','10.0.0.',255)]:
        if col not in out.columns:
            if col == 'ip':
                out[col] = ['10.0.0.' + str(i % 255) for i in range(len(out))]
            else:
                out[col] = [f"{pref}{i % mod}" for i in range(len(out))]
        out[col] = out[col].astype(str)

    # --- labels ---
    if cols['label']:
        y = _coerce_label_series(out[cols['label']])
        out['fraud'] = y
        # If label exists but has no positives, create synthetic fraud for training robustness
        if out['fraud'].sum() == 0:
            out['fraud_orig'] = out['fraud']
            k = max(5, int(0.02 * len(out)))  # top 2% or at least 5 rows
            idx = out['amount'].nlargest(k).index
            out['fraud'] = 0
            out.loc[idx, 'fraud'] = 1
    else:
        # No label provided -> synthesize positives using relative spend + night hours
        out['hour'] = out['timestamp'].dt.hour
        # z-score per merchant
        m_stats = out.groupby('merchant_id')['amount'].agg(['mean', 'std']).rename(
            columns={'mean': 'm_mean', 'std': 'm_std'})
        out = out.merge(m_stats, left_on='merchant_id', right_index=True, how='left')
        out['m_std'] = out['m_std'].replace(0, 1e-6)
        out['amount_z_m'] = (out['amount'] - out['m_mean']) / out['m_std']

        # fraud if unusual spend at night OR very unusual in general
        cond = ((out['amount_z_m'] > 3.0) & (out['hour'].isin([0, 1, 2, 3, 4, 5]))) | (out['amount_z_m'] > 4.0)

        # ensure at least a few positives (2% or 5)
        out['fraud'] = cond.astype(int)
        if out['fraud'].sum() < max(5, int(0.02 * len(out))):
            k = max(5, int(0.02 * len(out)))
            extra = out.loc[~cond].nlargest(k, 'amount_z_m').index
            out.loc[extra, 'fraud'] = 1

    # Sort by time for downstream rolling features
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out
