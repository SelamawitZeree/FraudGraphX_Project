# import pandas as pd
# def read_parquet(path: str) -> pd.DataFrame: return pd.read_parquet(path)
# def save_parquet(df: pd.DataFrame, path: str): df.to_parquet(path, index=False)

import os
import pandas as pd

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_parquet(df: pd.DataFrame, path: str):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    df.to_parquet(path, index=False)