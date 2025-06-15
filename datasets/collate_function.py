import torch
import MinkowskiEngine as ME

def collate_fn(batch):
    """
    Collate function for ScanNetPP Dataset.
    
    Args:
        batch: A list of samples, where each sample is a tuple
               (quantized_coords, feats, voxelized_masks, inverse_maps).
    
    Returns:
        A dictionary with batched sparse coordinates, features, masks, and inverse maps.
    """
    all_coords = []
    all_feats = []
    all_images = []
    all_targets = []
    for b_id, b_item in enumerate(batch):
        quantized_coords, feats, images, target_labels = b_item
        all_coords.append(quantized_coords)
        all_feats.append(feats)
        all_images.append(images['image'])
        all_targets.append(target_labels)

    coordinates, features = ME.utils.sparse_collate(coords = all_coords, feats=all_feats)

    return coordinates, features.float(), {'image': torch.stack(all_images)}, torch.stack(all_targets)
