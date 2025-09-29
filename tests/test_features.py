import pandas as pd
from src.features.tx_features import tx_basic
def test_tx_basic():
    df = pd.DataFrame({'transaction_id':[1], 'timestamp': pd.to_datetime(['2024-01-01']), 'amount':[10.0],
                       'card_id':['a'], 'merchant_id':['m1'], 'device_id':['d1'], 'ip':['1.1.1.1'], 'fraud':[0]})
    out = tx_basic(df); assert 'amount_log' in out.columns
