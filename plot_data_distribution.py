import json
import matplotlib.pyplot as plt
import os

# Paths
file_path = '/mnt/ssda/datasets/abo_500/filtered_product_weights.json'
scene_dir = "/mnt/ssda/PUGS/data/abo_500/scenes"
VAL_SPLIT_PATH = [f.split('_')[0] for f in os.listdir(scene_dir)]

# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Split into train and val
split_to_values = {'train': [], 'val': []}
for product_id, weight in data.items():
    if product_id in VAL_SPLIT_PATH:
        split_to_values['val'].append(weight)
    else:
        split_to_values['train'].append(weight)

# Combined stats
all_values = list(data.values())
min_val = min(all_values)
max_val = max(all_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.hist(split_to_values['train'], bins=30, alpha=0.6, label='Train', color='blue', edgecolor='black')
plt.hist(split_to_values['val'], bins=30, alpha=0.6, label='Validation', color='orange', edgecolor='black')

# Min/max lines
plt.axvline(min_val, color='red', linestyle='--', label=f'Min: {min_val:.2f}')
plt.axvline(max_val, color='green', linestyle='--', label=f'Max: {max_val:.2f}')

plt.title('Distribution of Product Weights (Train vs Validation)')
plt.xlabel('Weight (Kg)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data_distribution.png')
