import torch
import torch.nn as nn
from self_attention_head import SelfAttentionHead
from settings import embed_dim

class MultiHeadAttention(nn.Module):
  
  def __init__(self, max_seq_length, embedding_dim, n_heads=1):
    super(MultiHeadAttention, self).__init__()
    self.n_heads = n_heads
    self.attention_heads = []
    for i in range(n_heads):
      attention_head = SelfAttentionHead(embed_dim)
      self.attention_heads.append(attention_head)
    
  def __call__(self, X):
    out = []
    for i in range(self.n_heads):
      out.append(self.attention_heads[i](X))
    return torch.concat(out, dim=-1)