"""
Configuration and hyperparameters for the Inverse Dynamics Model.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class IDMConfig:
    """Configuration for Inverse Dynamics Model training and inference."""

    # Model Architecture
    num_stacked_frames: int = 2
    use_grayscale: bool = False
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 5  # STOPPED, FORWARD, BACKWARD, ROTATE_LEFT, ROTATE_RIGHT
    dropout_rate: float = 0.3

    # State Labels
    STATE_STOPPED: int = 0
    STATE_FORWARD: int = 1
    STATE_BACKWARD: int = 2
    STATE_ROTATE_LEFT: int = 3
    STATE_ROTATE_RIGHT: int = 4

    STATE_NAMES: dict = field(default_factory=lambda: {
        0: "STOPPED",
        1: "FORWARD",
        2: "BACKWARD",
        3: "ROTATE_LEFT",
        4: "ROTATE_RIGHT"
    })

    # Training Hyperparameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Optimizer & Scheduler
    optimizer: str = "adam"  # adam, sgd
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Early Stopping
    early_stopping_patience: int = 10
    best_model_metric: str = "accuracy"  # "accuracy", "loss", or "both"

    # Gradient Clipping
    use_gradient_clipping: bool = False
    max_gradient_norm: float = 1.0

    # Data Augmentation
    use_augmentation: bool = True
    random_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    random_rotation_degrees: float = 10.0

    # Normalization (ImageNet stats)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    save_best_only: bool = True
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_every_n_steps: int = 10

    # Random Seed
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.num_stacked_frames >= 2, "Must stack at least 2 frames"
        assert self.num_classes == 5, "Model expects exactly 5 control states"
        assert 0 <= self.dropout_rate < 1, "Dropout rate must be in [0, 1)"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.best_model_metric in ["accuracy", "loss", "both"], \
            "best_model_metric must be 'accuracy', 'loss', or 'both'"

    @property
    def input_channels(self) -> int:
        """Calculate number of input channels based on configuration."""
        channels_per_frame = 1 if self.use_grayscale else 3
        return channels_per_frame * self.num_stacked_frames
