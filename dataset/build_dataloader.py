import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.datasets import WikiText2
from torch.utils.data.dataset import Subset
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from dataset.utils import pad_or_truncate, get_label
from settings import device, max_seq_length, batch_size

class DataloaderBulder():
  
  def __init__(self):
    print("building vocab")
    self.build_vocab()
    
  def build_vocab(self):
    tokenizer = get_tokenizer('basic_english')
    data_iter = WikiText2(split='train')

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>", "<mask>"])
    vocab.set_default_index(vocab["<unk>"])
    self.vocab = vocab
    self.mask_index = vocab['<mask>']
    self.text_pipeline = lambda x: pad_or_truncate(vocab(tokenizer(x)))
    self.vocab_size = len(vocab)
    
  def collate_batch(self, batch):
    text_tesnor, label_tensor = torch.zeros(len(batch), max_seq_length, dtype=torch.int), torch.zeros(len(batch), dtype=torch.long) 
    for i, _text in enumerate(batch):
      token_ids = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
      token_ids = pad_or_truncate(token_ids)
      token_ids, label = get_label(token_ids, self.vocab_size, self.mask_index)
      text_tesnor[i] = token_ids
      label_tensor[i] = label

    return text_tesnor.to(device), label_tensor.to(device)

  def get_loaders(self):
    train_iter = WikiText2(split='train')
    val_iter = WikiText2(split='valid')
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, shuffle=False, collate_fn=self.collate_batch)
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, shuffle=False, collate_fn=self.collate_batch)
    return train_dataloader, val_dataloader