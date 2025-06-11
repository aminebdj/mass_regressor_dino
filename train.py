from trainer import train
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer on DINO features')

    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--save_best_model_in', type=str, default='./logs', help='Directory to save best model')
    parser.add_argument('--experiment_name', type=str, default='experiment1', help='Directory to save best model')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--overfit', type=bool, default=False, help='Number of training epochs')
    parser.add_argument('--backbone', type=str, default='init', help='Number of training epochs')
    parser.add_argument('--data_path', type=str, default='/mnt/ssda/abo-benchmark-material', help='Number of training epochs')
    parser.add_argument('--gt_path', type=str, default='/mnt/ssda/datasets/abo_500/filtered_product_weights.json', help='Number of training epochs')
    parser.add_argument('--val_path', type=str, default='/mnt/ssda/datasets/abo_500/scenes', help='Number of training epochs')
    # parser.add_argument('--checkpoint', type='', default='', help='Number of training epochs')
    parser.add_argument('--tune_blocks', type=list, default=['layer4'], help='Number of training epochs')

    args = parser.parse_args()

    train(
        device=args.device,
        batch_size=args.batch_size,
        save_best_model_in=args.save_best_model_in+'/'+args.experiment_name,
        num_epochs=args.num_epochs,
        overfit=args.overfit,
        backbone=args.backbone,
        data_path=args.data_path,
        gt_path=args.gt_path,
        val_path=args.val_path,
        tune_blocks = args.tune_blocks
    )