#!/bin/bash
#SBATCH --job-name=phyprop
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="a100|rtx_a6000"
#SBATCH --time=4-00:00:00
#SBATCH --output=./slurm/overfit/of_maple_classifier_bb_%j.log
#SBATCH --partition=submit
#SBATCH --exclude=ikarus

# Run your script
python train.py --overfit True --num_epochs 1000 --experiment_name classifier_3d --batch_size 32 --path_to_3d_samples /cluster/umoja/aminebdj/datasets/ABO/preprocessed --data_path /cluster/umoja/aminebdj/datasets/ABO/abo-benchmark-material --gt_path /cluster/umoja/aminebdj/datasets/ABO/abo_500/filtered_product_weights.json --val_path /cluster/umoja/aminebdj/datasets/ABO/abo_500/scenes
