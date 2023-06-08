import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from settings import max_seq_length, embed_dim, device

class TransformerEncoderLayer(nn.Module):

  def __init__(self, max_seq_len, embedding_dim, n_heads=8):
    super(TransformerEncoderLayer, self).__init__()
    self.multi_head_attention = MultiHeadAttention(max_seq_length, embed_dim, n_heads=8)
    self.norm1 = nn.LayerNorm(embedding_dim).to(device)
    self.norm2 = nn.LayerNorm(embedding_dim).to(device)
    self.fc1 = nn.Linear(embedding_dim, 2048).to(device)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(2048, embedding_dim).to(device)

  def __call__(self, X):
    att_X = self.multi_head_attention(X)
    X = X + att_X
    X = self.norm1(X)
    ff_X = self.relu(self.fc1(X))
    ff_X = self.fc2(ff_X)
    out = self.norm2(X + ff_X)
    return out