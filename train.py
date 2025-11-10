#!/usr/bin/env python3
"""
Training script for Inverse Dynamics Model.

Usage:
    # Train V1 (ResNet-18, 2 frames)
    python train.py --train_dir data/train --val_dir data/val --test_dir data/test

    # Train V2 (MobileNetV2, 4 frames)
    python train.py --model_version v2 --train_dir data/train --val_dir data/val --test_dir data/test

For more options:
    python train.py --help
"""

import argparse
import os
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Inverse Dynamics Model for RC Car Control Prediction"
    )

    # Model version selection
    parser.add_argument(
        "--model_version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Model version: v1 (ResNet-18, 2 frames) or v2 (MobileNetV2, 4 frames) (default: v1)"
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
        default=None,
        help="Number of consecutive frames to stack (default: 2 for v1, 4 for v2)"
    )
    parser.add_argument(
        "--use_grayscale",
        action="store_true",
        help="Use grayscale images instead of RGB"
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Don't use pretrained weights (ResNet-18 for v1, MobileNetV2 for v2)"
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
    parser.add_argument(
        "--best_model_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "loss", "both"],
        help="Metric to use for determining best model: 'accuracy', 'loss', or 'both' (default: accuracy)"
    )

    # Data Augmentation
    parser.add_argument(
        "--use_augmentation",
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

    # Import the correct version based on argument
    if args.model_version == "v2":
        from inverse_dynamics_model_v2.config import IDMConfig
        from inverse_dynamics_model_v2.model import InverseDynamicsModel
        from inverse_dynamics_model_v2.dataset import create_dataloaders
        from inverse_dynamics_model_v2.trainer import InverseDynamicsTrainer
        from inverse_dynamics_model_v2.utils import set_random_seed, get_device
        default_frames = 4
        model_name = "Inverse Dynamics Model V2 (MobileNetV2, 4 frames)"
    else:
        from inverse_dynamics_model.config import IDMConfig
        from inverse_dynamics_model.model import InverseDynamicsModel
        from inverse_dynamics_model.dataset import create_dataloaders
        from inverse_dynamics_model.trainer import InverseDynamicsTrainer
        from inverse_dynamics_model.utils import set_random_seed, get_device
        default_frames = 2
        model_name = "Inverse Dynamics Model V1 (ResNet-18, 2 frames)"

    # Set default num_stacked_frames if not provided
    num_stacked_frames = args.num_stacked_frames if args.num_stacked_frames is not None else default_frames

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}\n")

    # Create config from arguments
    config = IDMConfig(
        num_stacked_frames=num_stacked_frames,
        use_grayscale=args.use_grayscale,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        optimizer=args.optimizer,
        use_scheduler=not args.no_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        best_model_metric=args.best_model_metric,
        use_augmentation=args.use_augmentation,
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

    # Reload best model weights before saving final model
    if trainer.best_checkpoint_path is not None and os.path.exists(trainer.best_checkpoint_path):
        print(f"\nReloading best model from: {trainer.best_checkpoint_path}")
        trainer.load_model(trainer.best_checkpoint_path)
    else:
        print("\nWarning: No best checkpoint found. Saving final model with last epoch weights.")

    # Evaluate on test set if available
    if test_loader is not None:
        print("\nEvaluating on test set...")
        os.makedirs(args.eval_output_dir, exist_ok=True)
        test_metrics = trainer.evaluate(test_loader, save_dir=args.eval_output_dir)

    # Save final model (now contains best weights)
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
