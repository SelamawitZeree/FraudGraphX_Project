import torch
from src.models.autoencoder import AE
def test_shapes():
    m=AE(16); x=torch.randn(4,16); y=m(x); assert y.shape==x.shape
