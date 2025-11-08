"""
Utility functions for the Inverse Dynamics Model.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from typing import Dict, List, Tuple
import os
from pathlib import Path
from typing import Optional

from .config import IDMConfig


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: IDMConfig) -> torch.device:
    """
    Get the device to use for training/inference.

    Args:
        config: IDMConfig object

    Returns:
        torch.device
    """
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate overall accuracy.

    Args:
        predictions: Predicted class indices (batch_size,)
        targets: Ground truth class indices (batch_size,)

    Returns:
        Accuracy as a float (0-1)
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def calculate_per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Dict[int, float]:
    """
    Calculate accuracy for each class.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes

    Returns:
        Dictionary mapping class_id to accuracy
    """
    per_class_acc = {}

    for class_id in range(num_classes):
        mask = targets == class_id
        if mask.sum() > 0:
            class_predictions = predictions[mask]
            class_targets = targets[mask]
            per_class_acc[class_id] = calculate_accuracy(class_predictions, class_targets)
        else:
            per_class_acc[class_id] = 0.0

    return per_class_acc


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    config: IDMConfig
) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        config: IDMConfig object

    Returns:
        Dictionary with metrics
    """
    # Overall accuracy
    accuracy = (predictions == targets).mean()

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(config.num_classes)),
        average=None,
        zero_division=0
    )

    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=list(range(config.num_classes)))

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {
            i: {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
                "name": config.STATE_NAMES[i]
            }
            for i in range(config.num_classes)
        },
        "confusion_matrix": cm
    }

    return metrics


def print_metrics(metrics: Dict, config: IDMConfig, prefix: str = ""):
    """
    Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary from compute_metrics()
        config: IDMConfig object
        prefix: Optional prefix for the output (e.g., "Validation")
    """
    print(f"\n{'='*70}")
    print(f"{prefix} Metrics")
    print(f"{'='*70}")

    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:   {metrics['macro_f1']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'State':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*70}")

    for class_id in range(config.num_classes):
        class_metrics = metrics['per_class'][class_id]
        print(f"{class_metrics['name']:<15} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['f1']:<12.4f} "
              f"{class_metrics['support']:<10}")

    print(f"{'='*70}\n")


def plot_confusion_matrix(
    cm: np.ndarray,
    config: IDMConfig,
    save_path: str= "heatmap",
    normalize: bool = True
):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (num_classes, num_classes)
        config: IDMConfig object
        save_path: Optional path to save the figure
        normalize: Whether to normalize the confusion matrix
    """
    plt.figure(figsize=(10, 8))

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    # Get class names
    class_names = [config.STATE_NAMES[i] for i in range(config.num_classes)]

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_accuracy'
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.close()


def plot_per_class_performance(
    metrics: Dict,
    config: IDMConfig,
    save_path: Optional[str] = None
):
    """
    Plot per-class performance metrics.

    Args:
        metrics: Dictionary from compute_metrics()
        config: IDMConfig object
        save_path: Optional path to save the figure
    """
    class_names = [config.STATE_NAMES[i] for i in range(config.num_classes)]
    precision = [metrics['per_class'][i]['precision'] for i in range(config.num_classes)]
    recall = [metrics['per_class'][i]['recall'] for i in range(config.num_classes)]
    f1 = [metrics['per_class'][i]['f1'] for i in range(config.num_classes)]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Control State', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class performance plot saved to {save_path}")

    plt.close()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict,
    config: IDMConfig,
    filepath: str,
    is_best: bool = False
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        config: IDMConfig object
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': {
            'num_stacked_frames': config.num_stacked_frames,
            'use_grayscale': config.use_grayscale,
            'image_size': config.image_size,
            'num_classes': config.num_classes,
            'dropout_rate': config.dropout_rate,
            'input_channels': config.input_channels
        }
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

    if is_best:
        best_filepath = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_filepath)
        print(f"Best model saved to {best_filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, float, Dict]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load model to

    Returns:
        Tuple of (epoch, loss, metrics)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    metrics = checkpoint.get('metrics', {})

    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")

    return epoch, loss, metrics
