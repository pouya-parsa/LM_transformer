import time
import torch
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from dataset.build_dataloader import DataloaderBulder
from predictor import Predictor
from train import train
from evaluate import evaluate
from settings import max_seq_length, embed_dim, device, epochs, lr

dataloader_builder = DataloaderBulder()
vocab_size = dataloader_builder.vocab_size
train_dataloader, val_dataloader = dataloader_builder.get_loaders()

model = Predictor(max_seq_length, vocab_size, embed_dim, 6)
model.to(device)
# input_data = torch.randint(low=0, high=10, size=(8, 100))
# input_data = input_data.to(device)
# out_dist = model(input_data)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

total_accu = None
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(1, epochs + 1):
  train_epoch_acc, train_epoch_loss = train(model, optimizer, criterion, train_dataloader)
  train_loss_list.append(train_epoch_loss)
  train_acc_list.append(train_epoch_acc)
  epoch_start_time = time.time()
  accu_val, loss_val = evaluate(model, optimizer, criterion, val_dataloader)
  val_loss_list.append(loss_val)
  val_acc_list.append(accu_val)
  if total_accu is not None and train_epoch_acc > accu_val:
    scheduler.step()
  else:
    total_accu = accu_val
  print('-' * 59)
  print('| end of epoch {:3d} | time: {:5.2f}s | '
        'validation accuracy {:8.3f} | loss: {:5.2f}s '.format(epoch,
                                          time.time() - epoch_start_time,
                                          accu_val, loss_val))
  print('-' * 59)
  print(train_loss_list)
  print(train_acc_list)
  print(val_loss_list)
  print(val_acc_list)