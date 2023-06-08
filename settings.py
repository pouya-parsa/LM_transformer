import torch

embed_dim = 512
n_heads = 8
max_seq_length = 100
batch_size = 128
train_dataset_size = 36718 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20
lr = 5