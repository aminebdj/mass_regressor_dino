import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import ABO_DATASET
from model import Regressor
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as T
import multiprocessing
num_workers = multiprocessing.cpu_count()  # Or set a specific number like 4 or 8
def evaluate(model, dataloader, device, num_images=3):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        num_images = 0
        for data, targets in dataloader:
            images = data['image']
            num_images += len(images)
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss = (preds-targets.squeeze().repeat_interleave(images.shape[1], dim=0)).abs().sum()
            total_loss += loss.item()
    return total_loss / num_images

def train(data_path,gt_path,val_path,device='cuda', batch_size=8, save_best_model_in='./logs', num_epochs=100, overfit=False, backbone='clip', tune_blocks=[]):

    model = Regressor(feature_extractor = backbone, tune_blocks=tune_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_best_model_in = f"{save_best_model_in}/{timestamp}_{backbone}_epochs_{num_epochs}"
    os.makedirs(save_best_model_in, exist_ok=True)
    log_path = os.path.join(save_best_model_in, 'log.txt')
    transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    T.RandomRotation(degrees=10),
])

    train_dataset = ABO_DATASET(split='train', transform=transform, overfit=overfit, path_to_dataset=data_path, val_path=val_path, path_to_annotations=gt_path)
    val_dataset = ABO_DATASET(split='val', overfit=overfit, path_to_dataset=data_path, val_path=val_path, path_to_annotations=gt_path)
    # train_dataset[0]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3*(num_workers//4),pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_val_loss = float('inf')

    with open(log_path, 'w') as log_file:
        log_file.write("Epoch,TrainLoss,ValLoss\n")
    step = 0
    print(f"[Loading] with {num_workers} cores")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        num_images = 0
        for data, targets in tqdm(train_dataloader):
            images = data['image']
            num_images += len(images)

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
        avg_train_loss = running_train_loss / num_images
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