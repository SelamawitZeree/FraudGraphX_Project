import pandas as pd
# For each transaction, attach the merchant’s average amount, max amount, and total transaction count.
def merchant_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('merchant_id')['amount'].agg(['mean','max','count']).reset_index()
    agg.columns = ['merchant_id','m_amt_mean','m_amt_max','m_txn_count']
    feats = df[['transaction_id','merchant_id']].merge(agg, on='merchant_id', how='left')
    # we don't need merchant_id in the final feature set
    return feats.drop(columns=['merchant_id'])
