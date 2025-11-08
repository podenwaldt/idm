"""
Inverse Dynamics Model architecture based on ResNet-18.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

from .config import IDMConfig


class InverseDynamicsModel(nn.Module):
    """
    Deep learning model for predicting RC car control states from video frames.

    Architecture:
        - Modified ResNet-18 backbone (custom input channels)
        - Global Average Pooling
        - Fully Connected Layers with Dropout
        - Softmax output for 5 control states

    Args:
        config: IDMConfig object with model hyperparameters
        pretrained: Whether to use ImageNet pretrained weights for ResNet-18
    """

    def __init__(self, config: IDMConfig, pretrained: bool = True):
        super(InverseDynamicsModel, self).__init__()

        self.config = config
        self.num_classes = config.num_classes
        self.input_channels = config.input_channels

        # Load pretrained ResNet-18
        resnet18 = models.resnet18(pretrained=pretrained)

        # Modify first conv layer to accept custom number of input channels
        original_conv1 = resnet18.conv1
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize new conv1 weights
        if pretrained and not config.use_grayscale:
            # If using pretrained weights and RGB, average the weights across input channels
            with torch.no_grad():
                # original_conv1.weight shape: [64, 3, 7, 7]
                # We need to expand to [64, input_channels, 7, 7]
                original_weights = original_conv1.weight.data

                if self.input_channels == 6:  # 2 RGB frames stacked
                    # Duplicate the weights for the second frame
                    self.conv1.weight.data[:, :3, :, :] = original_weights
                    self.conv1.weight.data[:, 3:, :, :] = original_weights
                elif self.input_channels == 3:  # Single RGB frame (unusual but supported)
                    self.conv1.weight.data = original_weights
                else:
                    # For other configurations, initialize randomly (kaiming)
                    nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        else:
            # Initialize with kaiming normal
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Copy remaining layers from ResNet-18
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        # ResNet-18 residual blocks
        self.layer1 = resnet18.layer1  # 64 channels
        self.layer2 = resnet18.layer2  # 128 channels
        self.layer3 = resnet18.layer3  # 256 channels
        self.layer4 = resnet18.layer4  # 512 channels

        # Global Average Pooling
        self.avgpool = resnet18.avgpool

        # Custom Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.fc3 = nn.Linear(128, self.num_classes)

        # Initialize FC layers
        self._initialize_fc_layers()

    def _initialize_fc_layers(self):
        """Initialize fully connected layers with Xavier initialization."""
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)
               For default config: (batch_size, 6, 224, 224)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
            Note: Softmax is applied by CrossEntropyLoss during training
        """
        # Initial conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet-18 residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over classes.

        Args:
            x: Input tensor

        Returns:
            Probability distribution (batch_size, num_classes) with softmax applied
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
