"""
Dataset loading and preprocessing for Inverse Dynamics Model V2.
V2 uses 4 stacked frames for better temporal modeling.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IDMConfig


class RCCarInverseDynamicsDataset(Dataset):
    """
    PyTorch Dataset for RC Car Inverse Dynamics Model V2.

    Loads consecutive frames and their corresponding control states from
    a directory structure with inputs.json labels.

    V2 uses 4 consecutive frames (t-1, t, t+1, t+2) and predicts the action
    at time t, which caused the motion observed in frames t+1 and t+2.

    Directory structure:
        data/
          train/
            frame_0000.jpg
            frame_0001.jpg
            ...
            inputs.json
          val/
            frame_0000.jpg
            ...
            inputs.json

    inputs.json format:
        [
          {"frame": 0, "time": 0.0, "state": 0},
          {"frame": 1, "time": 0.033, "state": 1},
          ...
        ]

    Args:
        data_dir: Path to directory containing frames and inputs.json
        config: IDMConfig object
        split: Data split ('train', 'val', 'test')
        transform: Optional custom transform (overrides default)
    """

    def __init__(
        self,
        data_dir: str,
        config: IDMConfig,
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.num_stacked_frames = config.num_stacked_frames

        # Load inputs.json
        inputs_path = self.data_dir / "inputs.json"
        if not inputs_path.exists():
            raise FileNotFoundError(f"inputs.json not found at {inputs_path}")

        with open(inputs_path, "r") as f:
            self.labels = json.load(f)

        # Validate labels
        self._validate_labels()

        # Create frame sequences
        self.samples = self._create_samples()

        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")

    def _validate_labels(self):
        """Validate inputs.json format and content."""
        if not isinstance(self.labels, list):
            raise ValueError("inputs.json must contain a list")

        if len(self.labels) == 0:
            raise ValueError("inputs.json is empty")

        required_keys = {"frame", "time", "state"}
        for i, label in enumerate(self.labels):
            if not all(key in label for key in required_keys):
                raise ValueError(f"Label at index {i} missing required keys: {required_keys}")

            state = label["state"]
            if not isinstance(state, int) or state < 0 or state >= self.config.num_classes:
                raise ValueError(f"Invalid state {state} at index {i}. Must be 0-{self.config.num_classes-1}")

    def _create_samples(self) -> List[Dict]:
        """
        Create samples from consecutive frames.

        Each sample consists of num_stacked_frames consecutive frames
        and the control state at the second frame (true inverse dynamics).

        For 4 frames [t-1, t, t+1, t+2], we predict the action at time t,
        which caused the motion observed in frames t+1 and t+2.

        Returns:
            List of dicts with 'frames' (list of frame indices) and 'state' (int)
        """
        samples = []

        for i in range(len(self.labels) - self.num_stacked_frames + 1):
            # Get consecutive frames
            frame_indices = [self.labels[i + j]["frame"] for j in range(self.num_stacked_frames)]

            # Check frames are consecutive
            if not all(frame_indices[j+1] == frame_indices[j] + 1
                      for j in range(len(frame_indices) - 1)):
                continue  # Skip non-consecutive sequences

            # State corresponds to the second frame in the sequence (time t)
            # This action causes the motion visible in subsequent frames
            state = self.labels[i + 1]["state"]

            samples.append({
                "frames": frame_indices,
                "state": state
            })

        return samples

    def _get_default_transform(self) -> transforms.Compose:
        """Create default transform pipeline based on split and config."""
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize(self.config.image_size))

        # Data augmentation (only for training)
        if self.split == "train" and self.config.use_augmentation:
            # Random horizontal flip (handled separately to swap states)
            # Color jittering
            transform_list.append(transforms.ColorJitter(
                brightness=self.config.color_jitter_brightness,
                contrast=self.config.color_jitter_contrast,
                saturation=self.config.color_jitter_saturation,
            ))
            # Random rotation
            transform_list.append(transforms.RandomRotation(
                degrees=self.config.random_rotation_degrees
            ))

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        if not self.config.use_grayscale:
            transform_list.append(transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ))

        return transforms.Compose(transform_list)

    def _load_frame(self, frame_idx: int) -> Image.Image:
        """Load a single frame image."""
        frame_path = self.data_dir / f"frame_{frame_idx:04d}.jpg"

        # Try alternative extensions if .jpg not found
        if not frame_path.exists():
            frame_path = self.data_dir / f"frame_{frame_idx:04d}.png"

        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        img = Image.open(frame_path).convert("RGB")

        # Convert to grayscale if configured
        if self.config.use_grayscale:
            img = img.convert("L")

        return img

    def _apply_horizontal_flip_with_state_swap(
        self,
        frames: List[Image.Image],
        state: int
    ) -> Tuple[List[Image.Image], int]:
        """
        Apply random horizontal flip to frames and swap left/right states.

        When flipping horizontally:
        - ROTATE_LEFT (3) becomes ROTATE_RIGHT (4)
        - ROTATE_RIGHT (4) becomes ROTATE_LEFT (3)
        - Other states remain unchanged

        Args:
            frames: List of PIL Images
            state: Control state

        Returns:
            Tuple of (flipped frames, updated state)
        """
        if torch.rand(1).item() < self.config.random_flip_prob:
            # Flip all frames
            frames = [transforms.functional.hflip(frame) for frame in frames]

            # Swap left/right states
            if state == self.config.STATE_ROTATE_LEFT:
                state = self.config.STATE_ROTATE_RIGHT
            elif state == self.config.STATE_ROTATE_RIGHT:
                state = self.config.STATE_ROTATE_LEFT

        return frames, state

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (stacked_frames_tensor, state)
            - stacked_frames_tensor: (channels * num_frames, height, width)
              For V2 default: (12, 224, 224) = 4 frames * 3 RGB channels
            - state: int (0-4)
        """
        sample = self.samples[idx]
        frame_indices = sample["frames"]
        state = sample["state"]

        # Load frames
        frames = [self._load_frame(frame_idx) for frame_idx in frame_indices]

        # Apply horizontal flip with state swapping (training only)
        if self.split == "train" and self.config.use_augmentation:
            frames, state = self._apply_horizontal_flip_with_state_swap(frames, state)

        # Apply transforms to each frame
        transformed_frames = [self.transform(frame) for frame in frames]

        # Stack frames along channel dimension
        # Each frame: (C, H, W), stacked: (C * num_frames, H, W)
        stacked_frames = torch.cat(transformed_frames, dim=0)

        return stacked_frames, state

    def get_state_distribution(self) -> Dict[int, int]:
        """Get distribution of states in the dataset."""
        distribution = {i: 0 for i in range(self.config.num_classes)}
        for sample in self.samples:
            distribution[sample["state"]] += 1
        return distribution

    def print_statistics(self):
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print(f"Dataset Statistics: {self.split}")
        print(f"{'='*60}")
        print(f"Total samples: {len(self)}")
        print(f"Frames per sample: {self.num_stacked_frames}")

        distribution = self.get_state_distribution()
        print(f"\nState Distribution:")
        for state_id, count in distribution.items():
            state_name = self.config.STATE_NAMES[state_id]
            percentage = (count / len(self)) * 100
            print(f"  {state_name:15s} ({state_id}): {count:6d} ({percentage:5.2f}%)")

        print(f"{'='*60}\n")


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    config: IDMConfig,
    test_dir: Optional[str] = None
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and optionally test dataloaders.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        config: IDMConfig object
        test_dir: Optional path to test data directory

    Returns:
        Tuple of (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = RCCarInverseDynamicsDataset(train_dir, config, split="train")
    val_dataset = RCCarInverseDynamicsDataset(val_dir, config, split="val")

    # Print statistics
    train_dataset.print_statistics()
    val_dataset.print_statistics()

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    if test_dir is not None:
        test_dataset = RCCarInverseDynamicsDataset(test_dir, config, split="test")
        test_dataset.print_statistics()

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )

        return train_loader, val_loader, test_loader

    return train_loader, val_loader
