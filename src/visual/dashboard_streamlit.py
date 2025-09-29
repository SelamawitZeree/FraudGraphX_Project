# import streamlit as st, joblib, matplotlib.pyplot as plt
# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)
# from src.utils.io import read_parquet
# st.set_page_config(page_title='FraudGraphX Dashboard', layout='wide')
# st.title('FraudGraphX — Presentation Dashboard')
# data_path = st.text_input('Processed data path', 'data/processed/transactions_with_features.parquet')
# model_path = st.text_input('Model path', 'models/xgb_final.joblib')
# if st.button('Load'):
#     df=read_parquet(data_path); st.success(f'Loaded {len(df):,} rows')
#     model=joblib.load(model_path)
#     drop=['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
#     X=df.drop(columns=[c for c in drop if c in df.columns]).fillna(0.0)
#     proba=model.predict_proba(X)[:,1]; df['score']=proba
#     st.subheader('Top Alerts'); st.dataframe(df.sort_values('score', ascending=False).head(20))
#     fig=plt.figure(); plt.hist(df['score'], bins=50); plt.title('Score Distribution'); st.pyplot(fig)
#     tx_id = st.number_input('Transaction ID to inspect', int(df['transaction_id'].min()), int(df['transaction_id'].max()), int(df['transaction_id'].iloc[0]))
#     st.subheader('Transaction Details'); st.json(df[df['transaction_id']==tx_id].iloc[0].to_dict())
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import os, sys

# Ensure project root is in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.io import read_parquet

# Page setup
st.set_page_config(page_title='FraudGraphX Dashboard', layout='wide')
st.title('FraudGraphX — Presentation Dashboard')

# Input paths
data_path = st.text_input('Processed data path', 'data/processed/transactions_with_features.parquet')
model_path = st.text_input('Model path', 'models/xgb_final.joblib')

# Initialize session_state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None

# Load button
if st.button('Load'):
    df = read_parquet(data_path)
    st.session_state.df = df
    st.session_state.model = joblib.load(model_path)

    # Compute scores once
    drop = ['fraud', 'transaction_id', 'timestamp', 'card_id', 'merchant_id', 'device_id', 'ip']
    X = df.drop(columns=[c for c in drop if c in df.columns]).fillna(0.0)
    proba = st.session_state.model.predict_proba(X)[:, 1]
    st.session_state.df['score'] = proba

    st.success(f'Loaded {len(df):,} rows')

# If data is loaded, show dashboard
if st.session_state.df is not None:
    df = st.session_state.df

    # Top alerts
    st.subheader('Top Alerts')
    st.dataframe(df.sort_values('score', ascending=False).head(20))

    # Score distribution
    fig = plt.figure()
    plt.hist(df['score'], bins=50)
    plt.title('Score Distribution')
    st.pyplot(fig)

    # Transaction details
    st.subheader('Inspect Transaction')
    valid_ids = df['transaction_id'].tolist()

    # Persist selected tx_id in session state
    if "selected_tx" not in st.session_state:
        st.session_state.selected_tx = valid_ids[0]

    tx_id = st.selectbox(
        'Transaction ID to inspect',
        valid_ids,
        index=valid_ids.index(st.session_state.selected_tx) if st.session_state.selected_tx in valid_ids else 0,
        key="selected_tx"
    )

    st.subheader('Transaction Details')
    try:
        details = df[df['transaction_id'] == tx_id].iloc[0].to_dict()
        st.json(details)
    except IndexError:
        st.error(f"Transaction ID {tx_id} not found in dataset")

