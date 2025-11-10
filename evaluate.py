#!/usr/bin/env python3
"""
Standalone evaluation script for Inverse Dynamics Model.

This script loads a trained model from a .pth checkpoint file and evaluates it
on a test dataset, generating comprehensive metrics and visualizations.

Usage:
    # Evaluate V1 model
    python evaluate.py --model_version v1 --checkpoint checkpoints/idm_final.pth --test_dir data/test

    # Evaluate V2 model
    python evaluate.py --model_version v2 --checkpoint idm_final_MN_3F_V2.pth --test_dir data/test
    python evaluate.py --model_version v2 --checkpoint idm_final_MN_3F_V2.pth --test_dir recordings/demo/recording_132278

For more options:
    python evaluate.py --help
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Inverse Dynamics Model from checkpoint"
    )

    # Model version selection
    parser.add_argument(
        "--model_version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Model version: v1 (ResNet-18, 2 frames) or v2 (MobileNetV2, 4 frames) (default: v1)"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path to test data directory"
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
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
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


def load_checkpoint_and_config(checkpoint_path: str, device: torch.device, IDMConfig) -> tuple:
    """
    Load checkpoint and extract model configuration.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load checkpoint to
        IDMConfig: The IDMConfig class to use (v1 or v2)

    Returns:
        Tuple of (checkpoint_dict, config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration")

    config_dict = checkpoint['config']

    # Create IDMConfig from checkpoint config
    config = IDMConfig(
        num_stacked_frames=config_dict.get('num_stacked_frames', 2),
        use_grayscale=config_dict.get('use_grayscale', False),
        image_size=config_dict.get('image_size', (224, 224)),
        num_classes=config_dict.get('num_classes', 5),
        dropout_rate=config_dict.get('dropout_rate', 0.3),
    )

    # Print checkpoint info
    print(f"\n{'='*70}")
    print("Checkpoint Information")
    print(f"{'='*70}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Loss: {checkpoint['loss']:.4f}")
    if 'metrics' in checkpoint and 'accuracy' in checkpoint['metrics']:
        print(f"Accuracy: {checkpoint['metrics']['accuracy']:.4f}")
    print(f"\nModel Configuration:")
    print(f"  Stacked frames: {config.num_stacked_frames}")
    print(f"  Use grayscale: {config.use_grayscale}")
    print(f"  Image size: {config.image_size}")
    print(f"  Input channels: {config.input_channels}")
    print(f"  Dropout rate: {config.dropout_rate}")
    print(f"{'='*70}\n")

    return checkpoint, config


def create_model_and_load_weights(checkpoint: dict, config, device: torch.device, InverseDynamicsModel):
    """
    Create model instance and load weights from checkpoint.

    Args:
        checkpoint: Checkpoint dictionary
        config: IDMConfig object
        device: Device to load model to

    Returns:
        Loaded model
    """
    print("Creating model...")
    model = InverseDynamicsModel(config, pretrained=False)
    model.to(device)

    # Load weights
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain model weights")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Size: {model.get_model_size_mb():.2f} MB\n")

    return model


def create_test_dataloader(test_dir: str, config, batch_size: int, num_workers: int, RCCarInverseDynamicsDataset) -> DataLoader:
    """
    Create test dataloader.

    Args:
        test_dir: Path to test data directory
        config: IDMConfig object
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Test DataLoader
    """
    print(f"Loading test data from {test_dir}...")
    test_dataset = RCCarInverseDynamicsDataset(test_dir, config, split="test")
    test_dataset.print_statistics()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def evaluate_model(
    model,
    test_loader: DataLoader,
    config,
    device: torch.device,
    output_dir: str,
    utils_module
) -> Dict:
    """
    Evaluate model on test set and generate visualizations.

    Args:
        model: Trained model
        test_loader: Test data loader
        config: IDMConfig object
        device: Device to run evaluation on
        output_dir: Directory to save results

    Returns:
        Evaluation metrics dictionary
    """
    print("Evaluating model on test set...")
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for frames, targets in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device)
            targets = targets.to(device)

            outputs = model(frames)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Compute metrics
    metrics = utils_module.compute_metrics(all_predictions, all_targets, config)

    # Print metrics
    utils_module.print_metrics(metrics, config, prefix="Test")

    # Save visualizations
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix
    utils_module.plot_confusion_matrix(
        metrics['confusion_matrix'],
        config,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )

    # Per-class performance
    utils_module.plot_per_class_performance(
        metrics,
        config,
        save_path=os.path.join(output_dir, 'per_class_performance.png')
    )

    # Save metrics to text file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Test Evaluation Metrics\n")
        f.write("="*70 + "\n\n")

        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision:  {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:     {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score:   {metrics['macro_f1']:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        f.write(f"{'State':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*70 + "\n")

        for class_id in range(config.num_classes):
            class_metrics = metrics['per_class'][class_id]
            f.write(f"{class_metrics['name']:<15} "
                   f"{class_metrics['precision']:<12.4f} "
                   f"{class_metrics['recall']:<12.4f} "
                   f"{class_metrics['f1']:<12.4f} "
                   f"{class_metrics['support']:<10}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"\nEvaluation results saved to {output_dir}")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_performance.png")
    print(f"  - metrics.txt")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Import the correct version based on argument
    if args.model_version == "v2":
        from inverse_dynamics_model_v2.config import IDMConfig
        from inverse_dynamics_model_v2.model import InverseDynamicsModel
        from inverse_dynamics_model_v2.dataset import RCCarInverseDynamicsDataset
        import inverse_dynamics_model_v2.utils as utils_module
        model_name = "Inverse Dynamics Model V2 (MobileNetV2, 4 frames)"
    else:
        from inverse_dynamics_model.config import IDMConfig
        from inverse_dynamics_model.model import InverseDynamicsModel
        from inverse_dynamics_model.dataset import RCCarInverseDynamicsDataset
        import inverse_dynamics_model.utils as utils_module
        model_name = "Inverse Dynamics Model V1 (ResNet-18, 2 frames)"

    print(f"\n{'='*70}")
    print(f"Standalone Evaluation: {model_name}")
    print(f"{'='*70}\n")

    # Set random seed
    utils_module.set_random_seed(args.random_seed)

    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("Using CPU\n")

    # Load checkpoint and config
    checkpoint, config = load_checkpoint_and_config(args.checkpoint, device, IDMConfig)

    # Override batch_size and num_workers from command line
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers

    # Create model and load weights
    model = create_model_and_load_weights(checkpoint, config, device, InverseDynamicsModel)

    # Create test dataloader
    test_loader = create_test_dataloader(
        args.test_dir,
        config,
        args.batch_size,
        args.num_workers,
        RCCarInverseDynamicsDataset
    )

    # Evaluate
    metrics = evaluate_model(
        model,
        test_loader,
        config,
        device,
        args.output_dir,
        utils_module
    )

    # Final summary
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
