"""
Main script for PathMNIST classification (Part 1).
Students should implement the TODO sections to achieve >90% accuracy.
"""

import sys
import os

from classification_dataset import create_pathmnist_dataloaders
from classification_models import get_model, count_parameters
from classification_train import train_model

def main(args):
    print("=== CPH 100A Problem Set 2 - Part 1: PathMNIST Classification ===")
    
    # Load data
    print("Loading PathMNIST dataset...")
    train_loader, val_loader, num_classes = create_pathmnist_dataloaders(
        batch_size=32,
        num_workers=0,
        data_root='./data'
    )
    
    print(f"\nCreating {args.model_name} model...")
    model = get_model(args.model_name, num_classes=num_classes)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    print(f"\nTraining {args.model_name} model...")
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps_per_epoch=100  # Fast exploration mode. #TODO: Change for your full runs
        )
        
        print("Training completed successfully!")
        
        # Show final results
        if history['val_acc']:
            best_val_acc = max(history['val_acc'])
            print(f"Best validation accuracy: {best_val_acc:.4f}")
        
    except NotImplementedError as e:
        print(f"❌ Training failed: {e}")
        return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPH 100A Problem Set 2 - PathMNIST Classification')
    parser.add_argument('--model_name', type=str, default='mlp',
                       choices=['mlp', 'cnn'], #TODO: add your models names here
                       help='Model to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for regularization')
    args = parser.parse_args()
    
    main(args) 