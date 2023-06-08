import math
import torch
from settings import embed_dim, device

class PositionalEmbedding():
  
  def __init__(self, max_seq_length, embed_model_dim):
    self.max_seq_length = max_seq_length
    self.embed_model_dim = embed_model_dim
  
  def get_position_embedding(self, pos, i):
    if i % 2 == 0:
      val = math.sin(pos / (10_000 ** (2 * i / self.embed_model_dim)))
    else:
      val = math.cos(pos / (10_000 ** (2 * i / self.embed_model_dim)))
    return val
  
  def generate_positional_embedding(self):
    indices = torch.arange(embed_dim, dtype=torch.float32)
    indices = indices.repeat(self.max_seq_length, 1)
    pos = torch.arange(self.max_seq_length, dtype=torch.float32).view(-1, 1)
    pos = pos.repeat(1, 512)
    pos[:, 0::2] = torch.sin(pos[:, 0::2] / (10_000 ** (2*indices[:, 0::2] / self.embed_model_dim)))
    pos[:, 1::2] = torch.cos(pos[:, 1::2] / (10_000 ** (2*indices[:, 1::2] / self.embed_model_dim)))
    return pos
  
  def generate_positional_embedding1(self):
    X = torch.zeros((self.max_seq_length, self.embed_model_dim))
    for pos, word in enumerate(X):
      for i, _ in enumerate(word):
        X[pos][i] = self.get_position_embedding(pos, i)
    return X

  def __call__(self, X):
    # r1 = self.generate_positional_embedding1()
    pe =  self.generate_positional_embedding()
    pe = torch.unsqueeze(pe, dim=0).to(device)
    return X + pe
    