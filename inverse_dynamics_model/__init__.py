"""
Inverse Dynamics Model (IDM) for RC Car Control Prediction

This package implements a deep learning system that predicts control inputs
from sequences of video frames captured by an RC car camera.
"""

from .model import InverseDynamicsModel
from .dataset import RCCarInverseDynamicsDataset
from .trainer import InverseDynamicsTrainer
from .inference import InverseDynamicsPredictor
from .config import IDMConfig

__version__ = "1.0.0"

__all__ = [
    "InverseDynamicsModel",
    "RCCarInverseDynamicsDataset",
    "InverseDynamicsTrainer",
    "InverseDynamicsPredictor",
    "IDMConfig",
]
