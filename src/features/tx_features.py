import numpy as np, pandas as pd
def tx_basic(df: pd.DataFrame)->pd.DataFrame:
    x=df.copy()
    x['amount_log']=np.log1p(x['amount'])
    x['hour']=x['timestamp'].dt.hour
    x['weekday']=x['timestamp'].dt.weekday
    x['is_weekend']=(x['weekday']>=5).astype(int)
    keep=['transaction_id','card_id','merchant_id','device_id','ip','timestamp','amount','amount_log','hour','weekday','is_weekend','fraud']
    return x[keep]
