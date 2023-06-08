import time
import torch
from tqdm.auto import tqdm

def train(model, optimizer, criterion, train_dataloader):
  model.train()
  total_acc, total_count = 0, 0
  log_interval = 50
  start_time = time.time()
  train_loss = 0

  for idx, (text, label) in tqdm(enumerate(train_dataloader)):
    optimizer.zero_grad()
    predicted_label = model(text)
    # print(predicted_label.argmax(1))
    # print(label)
    loss = criterion(predicted_label, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    total_acc += (predicted_label.argmax(1) == label).sum().item()
    total_count += label.size(0)
    train_loss += loss.item()
      
  elapsed = time.time() - start_time
  print('| accuracy {:8.3f} | loss {:8.3f}'.format(
                                    total_acc/total_count, loss.item()))
  return (total_acc / total_count), (train_loss / total_count)