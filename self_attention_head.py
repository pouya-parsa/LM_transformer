import torch
import torch.nn as nn
from settings import device 

class SelfAttentionHead(nn.Module):

  def __init__(self, embedding_dim):
    super(SelfAttentionHead, self).__init__()
    internal_dim = 64
    self.embedding_dim = torch.tensor(embedding_dim).to(device)
    self.K_fc = nn.Linear(embedding_dim, internal_dim).to(device)
    self.Q_fc = nn.Linear(embedding_dim, internal_dim).to(device)
    self.V_fc = nn.Linear(embedding_dim, internal_dim).to(device)
  
  def __call__(self, X):
    K, Q, V = self.K_fc(X), self.Q_fc(X), self.V_fc(X) # K has the shape (b, max_seq_length, embedding_dim / n_heads)
    K = K.transpose(-2, -1)
    QK = torch.softmax(torch.matmul(Q, K) / torch.sqrt(self.embedding_dim), dim=-1)
    QKV = torch.matmul(QK, V)
    return QKV