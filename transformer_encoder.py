import torch.nn as nn
from transformer_encoder_layer import TransformerEncoderLayer
from settings import max_seq_length, embed_dim

class TransformerEncoder(nn.Module):
  
  def __init__(self, layers):
    super(TransformerEncoder, self).__init__()
    transformer_layers = []
    for i in range(layers):
      transformer_layers.append(TransformerEncoderLayer(max_seq_length, embed_dim, n_heads=8))
    self.layers = transformer_layers
  
  def __call__(self, X):
    for layer in self.layers:
      X = layer(X)
    
    return X