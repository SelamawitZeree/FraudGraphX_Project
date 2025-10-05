import pandas as pd
# For each transaction, add historical fraud rates for its merchant and device.
def neighbor_fraud_ratio(df: pd.DataFrame)->pd.DataFrame:
    m=df.groupby('merchant_id')['fraud'].mean().rename('merchant_fraud_rate').reset_index()
    d=df.groupby('device_id')['fraud'].mean().rename('device_fraud_rate').reset_index()
    out=df[['transaction_id','merchant_id','device_id']].merge(m, on='merchant_id', how='left').merge(d,on='device_id', how='left')
    return out[['transaction_id','merchant_fraud_rate','device_fraud_rate']].fillna(0.0)
