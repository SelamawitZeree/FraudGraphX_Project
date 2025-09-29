import argparse, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
from ..utils.io import read_parquet, save_parquet
class AE(nn.Module):
    def __init__(self,d): 
        super().__init__(); self.e=nn.Sequential(nn.Linear(d,max(8,d//2)),nn.ReLU(),nn.Linear(max(8,d//2),max(4,d//4)))
        self.d=nn.Sequential(nn.Linear(max(4,d//4),max(8,d//2)),nn.ReLU(),nn.Linear(max(8,d//2),d))
    def forward(self,x): z=self.e(x); return self.d(z)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input', required=True); ap.add_argument('--out', required=True); a=ap.parse_args()
    df=read_parquet(a.input)
    drop=['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
    Z = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore').copy()
    # keep only numeric columns for the autoencoder
    Z = Z.select_dtypes(include=['number']).fillna(0.0)
    X = Z.astype(float).values
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    d=Xs.shape[1]; model=AE(d); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss=nn.MSELoss()
    X_t=torch.tensor(Xs,dtype=torch.float32)
    for _ in range(20):
        opt.zero_grad(); xh=model(X_t); l=loss(xh,X_t); l.backward(); opt.step()
    with torch.no_grad(): err=((model(X_t)-X_t)**2).mean(dim=1).numpy()
    df['ae_anomaly']=(err-err.mean())/(err.std()+1e-6); save_parquet(df,a.out)
