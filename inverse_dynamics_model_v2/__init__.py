"""
Inverse Dynamics Model V2 (IDM V2) for RC Car Control Prediction

This package implements an improved deep learning system using MobileNetV2
and 4 stacked frames for better temporal modeling and efficiency.

Key improvements over V1:
- MobileNetV2 architecture for efficiency
- 4 consecutive frames (t-1, t, t+1, t+2) for better temporal context
"""

from .model import InverseDynamicsModel
from .dataset import RCCarInverseDynamicsDataset
from .trainer import InverseDynamicsTrainer
from .inference import InverseDynamicsPredictor
from .config import IDMConfig

__version__ = "2.0.0"

__all__ = [
    "InverseDynamicsModel",
    "RCCarInverseDynamicsDataset",
    "InverseDynamicsTrainer",
    "InverseDynamicsPredictor",
    "IDMConfig",
]
