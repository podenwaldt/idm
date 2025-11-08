"""
Training pipeline for Inverse Dynamics Model.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import InverseDynamicsModel
from .config import IDMConfig
from .utils import (
    calculate_accuracy,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_per_class_performance,
    save_checkpoint,
    load_checkpoint,
    get_device,
    set_random_seed
)


class InverseDynamicsTrainer:
    """
    Training and evaluation manager for the Inverse Dynamics Model.

    Args:
        model: InverseDynamicsModel instance
        config: IDMConfig object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
    """

    def __init__(
        self,
        model: InverseDynamicsModel,
        config: IDMConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else get_device(config)

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler() if config.use_scheduler else None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Trainer Initialized")
        print(f"{'='*70}")
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Model size: {model.get_model_size_mb():.2f} MB")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"{'='*70}\n")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.scheduler_min_lr
        )

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]")

        for batch_idx, (frames, targets) in enumerate(pbar):
            # Move to device
            frames = frames.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional)
            if self.config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_gradient_norm
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, epoch: Optional[int] = None) -> Tuple[float, float, Dict]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number (for logging)

        Returns:
            Tuple of (avg_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        desc = f"Epoch {epoch+1}/{self.config.num_epochs} [Val]" if epoch is not None else "Validation"
        pbar = tqdm(self.val_loader, desc=desc)

        with torch.no_grad():
            for frames, targets in pbar:
                # Move to device
                frames = frames.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(frames)
                loss = self.criterion(outputs, targets)

                # Get predictions
                predictions = torch.argmax(outputs, dim=1)

                # Accumulate
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        metrics = compute_metrics(all_predictions, all_targets, self.config)
        accuracy = metrics['accuracy']

        return avg_loss, accuracy, metrics

    def train(self) -> Dict:
        """
        Full training loop.

        Returns:
            Training history dictionary
        """
        print("Starting training...")
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss, val_accuracy, val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Time:       {epoch_time:.2f}s")

            # Save checkpoint
            is_best = val_accuracy > self.best_val_accuracy

            if self.config.save_best_only:
                if is_best:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"idm_epoch_{epoch+1}.pth"
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        val_loss,
                        val_metrics,
                        self.config,
                        checkpoint_path,
                        is_best=True
                    )
            else:
                # Save every N epochs or if best
                if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"idm_epoch_{epoch+1}.pth"
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        val_loss,
                        val_metrics,
                        self.config,
                        checkpoint_path,
                        is_best=is_best
                    )

            # Early stopping
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
                break

        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

        return self.history

    def evaluate(self, test_loader: DataLoader, save_dir: Optional[str] = None) -> Dict:
        """
        Evaluate model on test set and generate visualizations.

        Args:
            test_loader: Test data loader
            save_dir: Directory to save evaluation results

        Returns:
            Evaluation metrics dictionary
        """
        print("Evaluating model on test set...")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for frames, targets in tqdm(test_loader, desc="Testing"):
                frames = frames.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(frames)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets, self.config)

        # Print metrics
        print_metrics(metrics, self.config, prefix="Test")

        # Save visualizations
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            # Confusion matrix
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                self.config,
                save_path=os.path.join(save_dir, 'confusion_matrix.png')
            )

            # Per-class performance
            plot_per_class_performance(
                metrics,
                self.config,
                save_path=os.path.join(save_dir, 'per_class_performance.png')
            )

            # Training history
            plot_training_history(
                self.history,
                save_path=os.path.join(save_dir, 'training_history.png')
            )

            print(f"\nVisualization saved to {save_dir}")

        return metrics

    def save_model(self, filepath: str):
        """
        Save model for inference (weights only).

        Args:
            filepath: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_stacked_frames': self.config.num_stacked_frames,
                'use_grayscale': self.config.use_grayscale,
                'image_size': self.config.image_size,
                'num_classes': self.config.num_classes,
                'dropout_rate': self.config.dropout_rate,
                'input_channels': self.config.input_channels
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model weights.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")


def train_model(
    train_dir: str,
    val_dir: str,
    config: Optional[IDMConfig] = None,
    pretrained: bool = True
) -> Tuple[InverseDynamicsModel, InverseDynamicsTrainer]:
    """
    Convenience function to train a model from scratch.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        config: IDMConfig object (uses defaults if None)
        pretrained: Whether to use pretrained ResNet-18 weights

    Returns:
        Tuple of (trained_model, trainer)
    """
    if config is None:
        config = IDMConfig()

    # Set random seed
    set_random_seed(config.random_seed)

    # Get device
    device = get_device(config)

    # Create dataloaders
    from .dataset import create_dataloaders
    train_loader, val_loader = create_dataloaders(train_dir, val_dir, config)

    # Create model
    model = InverseDynamicsModel(config, pretrained=pretrained)

    # Create trainer
    trainer = InverseDynamicsTrainer(model, config, train_loader, val_loader, device)

    # Train
    trainer.train()

    return model, trainer
