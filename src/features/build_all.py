import argparse
from ..utils.io import read_parquet, save_parquet
from .tx_features import tx_basic
from .user_features import user_roll
from .merchant_features import merchant_agg
from .graph_features import neighbor_fraud_ratio
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--out', required=True)
    a=ap.parse_args()
    df=read_parquet(a.input)
    tx=tx_basic(df); ur=user_roll(df); ma=merchant_agg(df); gr=neighbor_fraud_ratio(df)
    out=tx.merge(ur,on='transaction_id',how='left').merge(ma,on='transaction_id',how='left').merge(gr,on='transaction_id',how='left').fillna(0.0)
    save_parquet(out,a.out)
