import torch.nn as nn
from embedding import EmbeddingClass
from positional_embedding import PositionalEmbedding
from transformer_encoder import TransformerEncoder

class Predictor(nn.Module):

  def __init__(self, max_seq_length, vocab_size, embed_dim, encoder_layers):
    super(Predictor, self).__init__()
    self.embedding = EmbeddingClass(vocab_size, embed_dim)
    self.positional_embedding = PositionalEmbedding(max_seq_length, embed_dim)
    self.encoder = TransformerEncoder(encoder_layers)
    self.fc = nn.Linear(embed_dim, vocab_size)
    self.softmax = nn.LogSoftmax(dim=-1)
  
  def __call__(self, X):
    X = self.embedding(X)
    X = self.positional_embedding(X)
    encoded = self.encoder(X)
    encoded = encoded[:, 0, :]
    out = self.fc(encoded)
    out = self.softmax(out)
    return out