from trainer import train
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer on DINO features')

    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--save_best_model_in', type=str, default='./logs', help='Directory to save best model')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--overfit', type=bool, default=False, help='Number of training epochs')

    args = parser.parse_args()

    train(
        device=args.device,
        batch_size=args.batch_size,
        save_best_model_in=args.save_best_model_in,
        num_epochs=args.num_epochs,
        overfit=args.overfit
    )