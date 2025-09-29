import argparse, pandas as pd
from .schema_utils import synthesize
from ..utils.io import save_parquet
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    a=ap.parse_args()
    df=pd.read_csv(a.input)
    df=synthesize(df).sort_values('timestamp')
    save_parquet(df,a.out)
