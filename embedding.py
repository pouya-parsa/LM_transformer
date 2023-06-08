import torch.nn as nn

class EmbeddingClass(nn.Module):
  
  def __init__(self, vocab_size, embedding_dim):
    super(EmbeddingClass, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
  
  def __call__(self, X):
    out = self.embedding_layer(X) # output shape [b, max_seq_length, embedding_dim]
    return out