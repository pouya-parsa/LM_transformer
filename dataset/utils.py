import torch
import random
from settings import max_seq_length, device

def pad_or_truncate(input_ids):
  if len(input_ids) > max_seq_length:
    return input_ids[:max_seq_length]
  elif len(input_ids) < max_seq_length:
    to_pad = max_seq_length - len(input_ids)
    for i in range(to_pad):
      input_ids.append(0)
    return input_ids
  return input_ids

def get_label(input_ids, vocab_size, mask_index):
  random_number = random.randint(0, len(input_ids) - 1)
  label = random_number
  input_ids[random_number] = mask_index
  return input_ids, label