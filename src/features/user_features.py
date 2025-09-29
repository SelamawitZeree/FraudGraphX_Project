import pandas as pd
def user_roll(df: pd.DataFrame)->pd.DataFrame:
    outs=[]
    for cid,g in df.sort_values('timestamp').groupby('card_id'):
        g=g.set_index('timestamp')
        r=g['amount'].rolling('24h').agg(['count','sum','mean','max'])
        r.columns=['user_txn_count_24h','user_amt_sum_24h','user_amt_mean_24h','user_amt_max_24h']
        outs.append(g.join(r).reset_index())
    o=pd.concat(outs, ignore_index=True)
    return o[['transaction_id','user_txn_count_24h','user_amt_sum_24h','user_amt_mean_24h','user_amt_max_24h']]
