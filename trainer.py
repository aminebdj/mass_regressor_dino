import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import ABO_DATASET
from model import DINORegressor
from tqdm import tqdm
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, targets in dataloader:
            images = data['image']
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss = (preds-targets.squeeze().repeat_interleave(images.shape[1], dim=0)).abs().sum()
            total_loss += loss.item()
    return total_loss / len(dataloader.dataset)


def train(device='cuda', batch_size=8, save_best_model_in='./logs', num_epochs=100, overfit=False):

    model = DINORegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    os.makedirs(save_best_model_in, exist_ok=True)
    log_path = os.path.join(save_best_model_in, 'log.txt')

    train_dataset = ABO_DATASET(split='train', overfit=overfit)
    val_dataset = ABO_DATASET(split='val', overfit=overfit)
    # train_dataset[0]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    best_val_loss = float('inf')

    with open(log_path, 'w') as log_file:
        log_file.write("Epoch,TrainLoss,ValLoss\n")  # CSV header
    step = 0
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for data, targets in tqdm(train_dataloader):
            images = data['image']
            # continue
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss = criterion(preds, targets.squeeze().repeat_interleave(images.shape[1], dim=0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            running_train_loss += loss.item() * images.size(0)
            # print(f"Step {step+1}: Train Loss = {loss.item() * images.size(0):.4f}")
        avg_train_loss = running_train_loss / len(train_dataloader.dataset)
        val_loss = evaluate(model, val_dataloader, device)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Append losses to log file
        with open(log_path, 'a') as log_file:
            log_file.write(f"{epoch+1},{avg_train_loss:.4f},{val_loss:.4f}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_best_model_in, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")