import time
import torch
from tqdm.auto import tqdm

def evaluate(model, optimizer, criterion, val_dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label) in tqdm(enumerate(val_dataloader)):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count, loss.item()