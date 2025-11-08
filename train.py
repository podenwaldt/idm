#!/usr/bin/env python3
"""
Training script for Inverse Dynamics Model.

Usage:
    python train.py --train_dir data/train --val_dir data/val --test_dir data/test

For more options:
    python train.py --help
"""

import argparse
import os
from pathlib import Path

from inverse_dynamics_model.config import IDMConfig
from inverse_dynamics_model.model import InverseDynamicsModel
from inverse_dynamics_model.dataset import create_dataloaders
from inverse_dynamics_model.trainer import InverseDynamicsTrainer
from inverse_dynamics_model.utils import set_random_seed, get_device


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Inverse Dynamics Model for RC Car Control Prediction"
    )

    # Data arguments
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to validation data directory"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Path to test data directory (optional)"
    )

    # Model arguments
    parser.add_argument(
        "--num_stacked_frames",
        type=int,
        default=2,
        help="Number of consecutive frames to stack (default: 2)"
    )
    parser.add_argument(
        "--use_grayscale",
        action="store_true",
        help="Use grayscale images instead of RGB"
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Don't use pretrained ResNet-18 weights"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)"
    )

    # Optimizer & Scheduler
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use (default: adam)"
    )
    parser.add_argument(
        "--no_scheduler",
        action="store_true",
        help="Disable learning rate scheduler"
    )

    # Early Stopping
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )

    # Data Augmentation
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="Disable data augmentation"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)"
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="Only save the best model"
    )

    # Other
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="evaluation",
        help="Directory to save evaluation results (default: evaluation)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print(f"\n{'='*70}")
    print("Inverse Dynamics Model - Training")
    print(f"{'='*70}\n")

    # Create config from arguments
    config = IDMConfig(
        num_stacked_frames=args.num_stacked_frames,
        use_grayscale=args.use_grayscale,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        optimizer=args.optimizer,
        use_scheduler=not args.no_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        use_augmentation=not args.no_augmentation,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epochs=args.save_every_n_epochs,
        save_best_only=args.save_best_only,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        device=args.device if args.device else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    )

    # Set random seed
    set_random_seed(config.random_seed)

    # Get device
    device = get_device(config)

    # Create dataloaders
    print("Loading data...")
    if args.test_dir:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.train_dir,
            args.val_dir,
            config,
            test_dir=args.test_dir
        )
    else:
        train_loader, val_loader = create_dataloaders(
            args.train_dir,
            args.val_dir,
            config
        )
        test_loader = None

    # Create model
    print("Creating model...")
    model = InverseDynamicsModel(config, pretrained=not args.no_pretrained)

    # Create trainer
    trainer = InverseDynamicsTrainer(
        model,
        config,
        train_loader,
        val_loader,
        device
    )

    # Train
    history = trainer.train()

    # Evaluate on test set if available
    if test_loader is not None:
        print("\nEvaluating on test set...")
        os.makedirs(args.eval_output_dir, exist_ok=True)
        test_metrics = trainer.evaluate(test_loader, save_dir=args.eval_output_dir)

    # Save final model
    final_model_path = os.path.join(config.checkpoint_dir, "idm_final.pth")
    trainer.save_model(final_model_path)

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")
    print(f"Best model saved to: {config.checkpoint_dir}")
    if test_loader is not None:
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Evaluation results saved to: {args.eval_output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
