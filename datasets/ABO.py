import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import random
import torch.nn.functional as F

def resize_to_patch_multiple(images, patch_size=14):
    H, W = images.shape[-2], images.shape[-1]
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size
    if H != new_H or W != new_W:
        images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return images
# import exr  # You may need to install a library like pyexr or openexr
def load_json(filepath):
    """
    Load and return the contents of a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def expand_array(input_array, k):
    n_needed = k - input_array.shape[0]  # number of additional slices
    indices = torch.randint(0, input_array.shape[0], (n_needed,))  # randomly pick indices to repeat

    repeated = input_array[indices]  # shape (n_needed, 5, 6)
    return repeated
import torchvision.transforms as T

class ABO_DATASET(Dataset):
    def __init__(self, split = 'train', overfit=False, path_to_dataset='/cluster/umoja/aminebdj/datasets/ABO/abo-benchmark-material', val_path="/cluster/umoja/aminebdj/datasets/ABO/abo_500/scenes", path_to_annotations='/cluster/umoja/aminebdj/datasets/ABO/abo_500/filtered_product_weights.json', return_probs = True):
        """
        Args:
            base_path (string): Path to the directory containing the subfolders with data.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        VAL_SPLIT_PATH = [f.split('_')[0] for f in os.listdir(val_path)]
        self.return_probs = return_probs
        self.base_path = path_to_dataset
        self.sample_to_mass = load_json(path_to_annotations)
        masses_list = list(self.sample_to_mass.values())
        self.max_w = max(masses_list)
        self.min_w = min(masses_list)
        probs = (np.array(masses_list)-self.min_w)/(self.max_w-self.min_w)
        self.sample_to_prob = dict(zip(list(self.sample_to_mass.keys()), list(probs)))
        # exit()
        self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                T.RandomRotation(degrees=10),
            ])
        self.transform_in  = False if overfit else True
        self.split = split
        self.num_images = 3 if split=='train' else -1 
        
        # Verify all required subdirectories exist
        self.required_folders = {
            'images': 'render',
            'normals': 'normals',
            'metallic_roughness': 'metallic_roughness',
            'render': 'render',
            'depth': 'depth',
            'masks': 'segmentation'
        }
        all_files = os.listdir(path_to_dataset)
        split_to_file = {'val':[], 'train':[]}
        for filename in all_files:
            if filename not in self.sample_to_mass:
                continue
            if filename in VAL_SPLIT_PATH:
                split_to_file['val'].append(filename)
                continue
            
            split_to_file['train'].append(filename)

        # Get list of files (assuming all folders have same files in same order)
        self.file_list = split_to_file[split]
        self.file_list = split_to_file['val'][:2] if overfit else self.file_list 
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        base_filename = os.path.splitext(self.file_list[idx])[0]
        # Load PNG images
        def load_png(subfolder, masks = None, frame_indices=None):
            path = os.path.join(self.base_path, base_filename, self.required_folders[subfolder])
            path_to_data = [os.path.join(path, root, fname) for root, _, fnames in os.walk(path) for fname in fnames if fname.endswith('.jpg') or fname.endswith('.png') ]
            # print(len(path_to_data))

            sorted_indices = sorted(range(len(path_to_data)), key=lambda i: int(path_to_data[i].split("_")[1].split('.')[0]))
            path_to_data = [path_to_data[i] for i in sorted_indices]

            num_samples = len(path_to_data) if self.num_images == -1 else self.num_images
            frame_indices = random.sample(range(len(path_to_data)), num_samples) if frame_indices is None else frame_indices
            path_to_data = [path for path in path_to_data if int(os.path.basename(path).split('.')[0].split('_')[-1]) in frame_indices]
            imgs = np.stack([np.array(Image.open(path)) for path in  path_to_data])
            

            if masks is not None:
                expen = len(imgs)//len(masks)
                expanded_masks = np.repeat(masks, expen, axis=0)[:len(imgs)]
                imgs = imgs*expanded_masks[..., None]
                if len(imgs) != num_samples*3:
                    imgs_additional = expand_array(imgs, num_samples*3)
                    imgs = np.concatenate([imgs, imgs_additional])
            return imgs, frame_indices
        
        # # Load EXR depth
        # def load_exr():
        #     path = os.path.join(self.base_path, self.required_folders['depth'], f"{base_filename}.exr")
        #     depths = np.stack([exr.read(os.path.join(path, fname) for fname in os.listdir(path))])
        #     return depths  # Adjust based on your EXR library
        
        # Load all data
        masks, frame_indices = load_png('masks')
        masks = masks > 125
        image, _ = load_png('images', masks, frame_indices=frame_indices)
        # normal = load_png('normals', masks)
        # metallic_roughness = load_png('metallic_roughness', masks)
        # render = load_png('render', masks)
        # depth = load_exr()
        
        # Convert to tensors
        img_tensor = resize_to_patch_multiple(torch.from_numpy(image).float().permute(0,3, 1, 2) / 255.0)
        if self.split == 'train' and self.transform_in:
            img_tensor = self.transform(img_tensor)
        sample = {
            'image': img_tensor,
            # 'masks': masks,
            # 'normal': torch.from_numpy(normal).float().permute(0,3, 1, 2) / 255.0,
            # 'metallic_roughness': torch.from_numpy(metallic_roughness).float().permute(0,3, 1, 2) / 255.0,
            # 'render': torch.from_numpy(render).float().permute(0,3, 1, 2) / 255.0,
            # 'depth': torch.from_numpy(depth).float().unsqueeze(1)  # Add channel dimension
        }
        
        
        
        target_label = self.sample_to_prob[self.file_list[idx]] if self.return_probs else self.sample_to_mass[self.file_list[idx]]
        return sample, torch.tensor([target_label,1-target_label ])