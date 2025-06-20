import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.ABO import ABO_DATASET
from datasets.collate_function import collate_fn
from model import Regressor, MaPLe, Classifier
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
num_workers = max(os.cpu_count() - 1, 1)  # Ensure at least 1 worker
import MinkowskiEngine as ME

import numpy as np
def save_preds_gt(validation_gt, validation_preds, epoch, save_figres_in):
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to NumPy arrays
    validation_gt = np.array(validation_gt)
    validation_preds = np.array(validation_preds)

    # Avoid division or log errors by clipping small values
    epsilon = 1e-8
    validation_gt = np.clip(validation_gt, epsilon, None)
    validation_preds = np.clip(validation_preds, epsilon, None)

    # Sort by descending ground truth
    sorted_indices = np.argsort(-validation_gt)
    gt_sorted = validation_gt[sorted_indices]
    pred_sorted = validation_preds[sorted_indices]

    # Compute per-sample metrics
    ade = np.abs(gt_sorted - pred_sorted)
    alde = np.abs(np.log(gt_sorted) - np.log(pred_sorted))
    ape = np.abs(gt_sorted - pred_sorted) / gt_sorted
    mnre = np.minimum(gt_sorted / pred_sorted, pred_sorted / gt_sorted)

    # Plot all 4 metrics
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(ade, label='ADE', color='blue')
    plt.title('Absolute Difference Error (ADE)')
    plt.ylim([0, np.max(ade)*1.1])
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(alde, label='ALDE', color='orange')
    plt.title('Absolute Log Difference Error (ALDE)')
    plt.ylim([0, np.max(alde)*1.1])
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(ape, label='APE', color='green')
    plt.ylim([0, 1.1])
    plt.title('Absolute Percentage Error (APE)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(mnre, label='MnRE', color='red')
    plt.ylim([0, 1.1])
    plt.title('Min Ratio Error (MnRE)')
    plt.grid(True)

    plt.suptitle(f'Error Metrics per Sample (Sorted by GT) - Epoch {epoch+1}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_figres_in.replace('.png', '_metrics.png'))
    plt.close()

    # Plot predictions vs ground truth
    plt.figure(figsize=(10, 6))
    plt.plot(gt_sorted, label='Ground Truth', color='blue')
    plt.plot(pred_sorted, label='Predictions', color='red')
    plt.ylim([0, max([np.max(pred_sorted), np.max(gt_sorted)]) * 1.1])
    plt.title(f'Normal Predictions vs Ground Truth - Epoch {epoch+1}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_figres_in.replace('.png', '_sorted.png'))
    plt.close()

    # Return dictionary of mean metrics
    return {
        'ADE': float(np.mean(ade)),
        'ALDE': float(np.mean(alde)),
        'APE': float(np.mean(ape)),
        'MnRE': float(np.mean(mnre)),
    }
    
# def soft_cross_entropy(logits, target_probs, reduction='mean'):
#     """
#     logits: (batch_size, num_classes)
#     target_probs: (batch_size, num_classes) - soft label distribution
#     """
#     log_probs = F.log_softmax(logits, dim=1)
#     loss = -(target_probs * log_probs).sum(dim=1)

#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     else:
#         return loss

def soft_cross_entropy(logits, target_probs, reduction='mean'):
    """
    logits: (batch_size, num_classes)
    target_probs: (batch_size, num_classes) - soft label distribution
    """
    log_probs = F.log_softmax(logits, dim=1)
    
    # KL divergence: KL(target || predicted)
    loss = F.kl_div(log_probs, target_probs, reduction='mean')
    return loss

    # if reduction == 'mean':
    #     return loss.mean()
    # elif reduction == 'sum':
    #     return loss.sum()
    # else:
    #     return loss

def evaluate(maple_trainer, dataloader, device, num_images=3):
    maple_trainer.model.eval()
    total_loss = 0.0
    b_size = 3
    validation_preds = []
    validation_gt = []
    with torch.no_grad():
        num_images = 0
        mass_mapping = dataloader.dataset.corr_property_values
        mass_res = dataloader.dataset.corr_property_values[1]-dataloader.dataset.corr_property_values[0]
        print('Evaluating ... ')
        for voxels, features, data, targets, mass_targets, fg_probs_gt in dataloader:
            images = data['image'].to(device)
            mass_targets = mass_targets.cpu()
            # all_preds = []
            # Process large image batches in sub-batches to avoid OOM
            # for i in range(0, images.shape[1], b_size):
            # num_images = len(images)
            # step_size = num_images//b_size if b_size < num_images else 1
            # batch = images[:, ::step_size]  # sub-batch
            # print(len(batch))
            # print(f"Features shape: {features.shape}")
            # print(f"IMage shape: {batch.shape}")
            sparse_input = ME.SparseTensor(coordinates=voxels.to(device), features=features.to(device))
            pred_logits, fg_prob = maple_trainer.model(images, sparse_input)
            # preds = pred_logits
            preds = F.softmax(pred_logits, dim=1)  # still on GPU
            # exit()
            
            # all_preds.append(preds)
            # preds = torch.cat(all_preds)
            # Extend targets to match number of images (move to CPU as well)
            # tragets_ext = targets.repeat_interleave(images.shape[1], dim=0)
            
            
            

            # Compute loss using mass weighting
            # min_mass = dataloader.dataset.min_w
            # max_mass = dataloader.dataset.max_w
            # print(preds.argmax(dim=-1).cpu().numpy().shape)
            # print(preds.shape)
            # print(preds.argmax(dim=-1).cpu().numpy())
            # exit()
            pred_mass = torch.tensor([mass_mapping[p_idx] for p_idx in preds.argmax(dim=-1).long().cpu().numpy()])+fg_prob.squeeze().item()*mass_res
            
            loss = (pred_mass - mass_targets).abs().sum()
            # print(preds)
            # print(tragets_ext)
            # exit()
            validation_gt += mass_targets.cpu().tolist() 
            validation_preds += pred_mass.cpu().tolist() 
            total_loss += loss.item()
            # num_images += preds.shape[0]

    return total_loss / len(dataloader), validation_gt, validation_preds
def train(data_path,gt_path,val_path, path_to_3d_samples,device='cuda', batch_size=8, fuse=False, save_best_model_in='./logs', num_epochs=100, overfit=False, backbone='clip', tune_blocks=[], mass_step = 2000):
    classnames = [
            f"An object with weight {w}g"
            for w in range(9, 453138+mass_step, mass_step)
        ]
        
    corr_property_values = np.array([
            float(w/1000)
            for w in range(9, 453138+mass_step, mass_step)
        ])
    # model = Regressor(feature_extractor = backbone, tune_blocks=tune_blocks).to(device)
    maple_trainer = MaPLe(fuse=fuse, corr_property_values=corr_property_values, classnames= classnames)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # criterion = nn.MSELoss()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_best_model_in = f"{save_best_model_in}/{timestamp}_{backbone}_epochs_{num_epochs}"
    os.makedirs(save_best_model_in, exist_ok=True)
    log_path = os.path.join(save_best_model_in, 'log.txt')


    train_dataset = ABO_DATASET(split='train', overfit=overfit, path_to_3d_samples=path_to_3d_samples, path_to_dataset=data_path, val_path=val_path, path_to_annotations=gt_path, corr_property_values=corr_property_values)
    val_dataset = ABO_DATASET(split='val', overfit=overfit, path_to_3d_samples=path_to_3d_samples, path_to_dataset=data_path, val_path=val_path, path_to_annotations=gt_path, corr_property_values=corr_property_values)
    # train_dataset[0]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=11, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=11, shuffle=False, collate_fn=collate_fn)
    # print(len(val_dataloader))
    # exit()
    # best_val_loss = float('inf')

    with open(log_path, 'w') as log_file:
        log_file.write("Epoch,TrainLoss, ADE*,ADE,ALDE,APE,MnRE\n")
    step = 0
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(os.path.dirname(log_path), f'best_model_Fuse_{fuse}_BS_{batch_size}.pth')
    # print(f"[Loading] with {num_workers} cores")
    for epoch in range(num_epochs):
        maple_trainer.model.train()
        running_train_loss = 0.0
        num_images = 0
        # val_loss, validation_gt, validation_preds = evaluate(maple_trainer, val_dataloader, device)
        
        for voxels, features, data, targets, mass_targets, fg_probs_gt in tqdm(train_dataloader):
            images = data['image']
            num_images += len(images)

            # continue
            images = images.to(device)
            targets = targets.to(device)
            fg_probs_gt = fg_probs_gt.to(device)
            sparse_input = ME.SparseTensor(coordinates=voxels.to(device), features=features.to(device))
            logits, fg_prob = maple_trainer.model(images, sparse_input)

            # preds = F.softmax(logits, dim=1)  # still on GPU
            # tragets_ext = targets
            # tragets_ext = targets.repeat_interleave(images.shape[1], dim=0)
            # loss = soft_cross_entropy(logits, tragets_ext.float())
            # loss = 0.001*(logits[:, 0]- tragets_ext[:, 0].float()).abs().mean()
            # print(len(targets))
            # print(targets)
            # exit()
            # fg_prob = fg_prob.squeeze(-1)
            # fg_prob = fg_prob / fg_prob.sum()
            # fg_probs_gt = fg_probs_gt / fg_probs_gt.sum()
            # log_fg_probs_gt = torch.log(fg_probs_gt.float() + 1e-10)  # Add epsilon for numerical stability

            # Compute KL divergence (element-wise or batch-wise)
            mse_loss = F.mse_loss(fg_probs_gt.float(), fg_prob.squeeze(-1), reduction='mean') # or 'none', 'sum', 'mean'
            # print(kl_div_loss)
            loss = F.cross_entropy(logits, targets)+0.1*mse_loss
            maple_trainer.optim.zero_grad()
            loss.backward()
            maple_trainer.optim.step()


            # maple_trainer.optim.update_lr()
            running_train_loss += loss.item()
            # print(f"Step {step+1}: Train Loss = {loss.item():.4f}")
        if epoch % 5 == 0:
            avg_train_loss = running_train_loss / num_images
            val_loss, validation_gt, validation_preds = evaluate(maple_trainer, val_dataloader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': maple_trainer.model.state_dict(),
                    'optimizer_state_dict': maple_trainer.optim.state_dict(),
                    'val_loss': val_loss,
                }, best_checkpoint_path)
                print(f"Best model saved at epoch {epoch+1} with ADE {val_loss:.4f}")

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss/len(val_dataloader.dataset):.4f}, ADE = {val_loss:.4f}")
            save_figres_in = '/'.join(log_path.split('/')[:-1]+['per_sample_preds', f'{epoch}.png'])
            os.makedirs('/'.join(save_figres_in.split('/')[:-1]), exist_ok=True)
            metrics = save_preds_gt(validation_gt, validation_preds, epoch, save_figres_in)
            # Append losses to log file
            with open(log_path, 'a') as log_file: 
                log_file.write(f"{epoch+1},{avg_train_loss:.4f},{val_loss:.4f},{metrics['ADE']:.4f},{metrics['ALDE']:.4f},{metrics['APE']:.4f},{metrics['MnRE']:.4f}\n")


        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_path = os.path.join(save_best_model_in, 'best_model.pt')
        #     torch.save(maple_trainer.model.state_dict(), save_path)
        #     print(f"Saved new best model to {save_path}")