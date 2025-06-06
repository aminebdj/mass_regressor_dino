import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.ABO import ABO_DATASET
from model import Regressor, MaPLe
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
def soft_cross_entropy(logits, target_probs, reduction='mean'):
    """
    logits: (batch_size, num_classes)
    target_probs: (batch_size, num_classes) - soft label distribution
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def evaluate(maple_trainer, dataloader, device, num_images=3):
    maple_trainer.model.eval()
    total_loss = 0.0
    b_size = 100

    with torch.no_grad():
        num_images = 0
        for data, targets in dataloader:
            images = data['image'].to(device)
            targets = targets.to(device)
            all_preds = []
            # Process large image batches in sub-batches to avoid OOM
            for i in range(0, images.shape[1], b_size):
                batch = images[:, i:i + b_size]  # sub-batch
                pred_logits = maple_trainer.model(batch)
                preds = F.softmax(pred_logits, dim=1)  # still on GPU
                all_preds.append(preds)
            preds = torch.cat(all_preds)
            # Extend targets to match number of images (move to CPU as well)
            tragets_ext = targets.repeat_interleave(images.shape[1], dim=0)

            # Compute loss using mass weighting
            min_mass = dataloader.dataset.min_w
            max_mass = dataloader.dataset.max_w
            pred_mass = preds[:, 0] * min_mass + (1 - preds[:, 0]) * max_mass
            target_mass = tragets_ext[:, 0] * min_mass + (1 - tragets_ext[:, 0]) * max_mass

            loss = (pred_mass - target_mass).abs().sum()
            total_loss += loss.item()
            num_images += preds.shape[0]

    return total_loss / num_images
def train(data_path,gt_path,val_path,device='cuda', batch_size=8, save_best_model_in='./logs', num_epochs=100, overfit=False, backbone='clip', tune_blocks=[]):

    # model = Regressor(feature_extractor = backbone, tune_blocks=tune_blocks).to(device)
    maple_trainer = MaPLe()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # criterion = nn.MSELoss()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_best_model_in = f"{save_best_model_in}/{timestamp}_{backbone}_epochs_{num_epochs}"
    os.makedirs(save_best_model_in, exist_ok=True)
    log_path = os.path.join(save_best_model_in, 'log.txt')


    train_dataset = ABO_DATASET(split='train', overfit=overfit, path_to_dataset=data_path, val_path=val_path, path_to_annotations=gt_path)
    val_dataset = ABO_DATASET(split='val', overfit=overfit, path_to_dataset=data_path, val_path=val_path, path_to_annotations=gt_path)
    # train_dataset[0]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # best_val_loss = float('inf')

    with open(log_path, 'w') as log_file:
        log_file.write("Epoch,TrainLoss,ADE\n")
    step = 0
    # print(f"[Loading] with {num_workers} cores")
    for epoch in range(num_epochs):
        maple_trainer.model.train()
        running_train_loss = 0.0
        num_images = 0
        for data, targets in tqdm(train_dataloader):
            images = data['image']
            num_images += len(images)

            # continue
            images = images.to(device)
            targets = targets.to(device)

            logits = maple_trainer.model(images)
            tragets_ext = targets.repeat_interleave(images.shape[1], dim=0)
            loss = 50*soft_cross_entropy(logits, tragets_ext)

            maple_trainer.optim.zero_grad()
            loss.backward()
            maple_trainer.optim.step()


            # maple_trainer.optim.update_lr()
            running_train_loss += loss.item()
            print(f"Step {step+1}: Train Loss = {loss.item():.4f}")
        avg_train_loss = running_train_loss / num_images
        val_loss = evaluate(maple_trainer, val_dataloader, device)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss/len(val_dataloader.dataset):.4f}, ADE = {val_loss:.4f}")

        # Append losses to log file
        with open(log_path, 'a') as log_file:
            log_file.write(f"{epoch+1},{avg_train_loss:.4f},{val_loss:.4f}\n")

        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_path = os.path.join(save_best_model_in, 'best_model.pt')
        #     torch.save(maple_trainer.model.state_dict(), save_path)
        #     print(f"Saved new best model to {save_path}")