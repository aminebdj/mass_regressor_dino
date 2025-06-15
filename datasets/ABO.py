import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import random
import torch.nn.functional as F
from pathlib import Path
import volumentations as V
import albumentations as A
import MinkowskiEngine as ME

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
def fast_image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB').copy()
    return np.array(img)
def expand_array(input_array, k):
    n_needed = k - input_array.shape[0]  # number of additional slices
    indices = torch.randint(0, input_array.shape[0], (n_needed,))  # randomly pick indices to repeat

    repeated = input_array[indices]  # shape (n_needed, 5, 6)
    return repeated
import torchvision.transforms as T

class ABO_DATASET(Dataset):
    def __init__(self, split = 'train', overfit=False,path_to_3d_samples = '/mnt/ssda/datasets/ABO/raw/3dmodels/preprocessed', path_to_dataset='/cluster/umoja/aminebdj/datasets/ABO/abo-benchmark-material', val_path="/cluster/umoja/aminebdj/datasets/ABO/abo_500/scenes", path_to_annotations='/cluster/umoja/aminebdj/datasets/ABO/abo_500/filtered_product_weights.json', return_probs = True):
        """
        Args:
            base_path (string): Path to the directory containing the subfolders with data.
            transform (callable, optional): Optional transform to be applied to the images.
        """
            # Load volume augmentations
        volume_augmentations_path =  './configs/augmentations/volum_augm.yaml'
        image_augmentations_path = './configs/augmentations/album_aug.yaml'
        self.volume_augmentations = V.NoOp()
        self.path_to_3d_samples = path_to_3d_samples
        if (volume_augmentations_path is not None) and (volume_augmentations_path.lower() != "none"):
            self.volume_augmentations = V.load(Path(volume_augmentations_path), data_format="yaml")

        # Load image augmentations
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (image_augmentations_path.lower() != "none"):
            self.image_augmentations = A.load(Path(image_augmentations_path), data_format="yaml")
        VAL_SPLIT_PATH = [f.split('_')[0] for f in os.listdir(val_path)]
        self.return_probs = return_probs
        self.base_path = path_to_dataset
        self.sample_to_mass = load_json(path_to_annotations)
        masses_list = list(self.sample_to_mass.values())
        # self.max_w = max(masses_list)
        # self.min_w = min(masses_list)
        self.max_w = 20
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
        self.split = 'overfit' if overfit else split
        
        self.num_images = 3 if split=='train' else -1 
        self.voxel_size = 0.02
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
        self.file_list = split_to_file['val'][:1] if overfit else self.file_list 
        self.corr_property_values = np.array([
            w/1000
            for w in range(100, 400_001, 2000)
        ])
        self.sample_to_paths = {}
        for sample_id in self.file_list:
            sample_dir = os.path.join(self.base_path, sample_id)
            img_dir = os.path.join(sample_dir, self.required_folders['images'])
            img_paths = sorted([
                os.path.join(dp, f)
                for dp, _, fn in os.walk(img_dir)
                for f in fn if f.endswith(('.png', '.jpg'))
            ], key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            self.sample_to_paths[sample_id] = img_paths
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        sample_id = self.file_list[idx]
        data_3d = np.load(os.path.join(self.path_to_3d_samples, sample_id+'.npy'))
        points, color = data_3d[:,:3], data_3d[:,6:9]
        if self.split == 'train':
            # Apply volume augmentations
            aug = self.volume_augmentations(
                points=points,
                features=color,
            )
            points, color= aug["points"], aug["features"]
            
            # Apply image augmentations
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])
        feats = color/255.
        quantized_coords, feats, unique_indices, inverse_maps = ME.utils.sparse_quantize(
            coordinates=points,
            features=feats,
            return_inverse=True,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        img_paths = self.sample_to_paths[sample_id]
        num_samples = self.num_images if self.num_images != -1 else len(img_paths)
        frame_indices = random.sample(range(len(img_paths)), num_samples)
        selected_paths = [img_paths[i] for i in frame_indices]
        
        imgs = np.stack([fast_image_loader(p) for p in selected_paths])
        
        img_tensor = torch.from_numpy(imgs).float().permute(0, 3, 1, 2) / 255.0
        img_tensor = resize_to_patch_multiple(img_tensor)
        
        if self.split == 'train' and self.transform_in:
            img_tensor = self.transform(img_tensor)

        # prob = self.sample_to_prob[sample_id] if self.return_probs else self.sample_to_mass[sample_id]
        target_mass = self.sample_to_mass[sample_id]
        class_idx = np.argmin(np.abs(self.corr_property_values-target_mass))
        # prob = self.sample_to_mass[sample_id]
        # print(prob)
        # exit()
        return quantized_coords, feats, {'image': img_tensor}, torch.tensor([class_idx]),  torch.tensor([target_mass])