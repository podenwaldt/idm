"""
Inverse Dynamics Model V2 architecture based on MobileNetV2.
This version uses MobileNetV2 for improved efficiency and 4 stacked frames.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

from .config import IDMConfig


class InverseDynamicsModel(nn.Module):
    """
    Deep learning model for predicting RC car control states from video frames.

    V2 Architecture:
        - Modified MobileNetV2 backbone (custom input channels for 4 frames)
        - Global Average Pooling
        - Fully Connected Layers with Dropout
        - Softmax output for 5 control states

    Args:
        config: IDMConfig object with model hyperparameters
        pretrained: Whether to use ImageNet pretrained weights for MobileNetV2
    """

    def __init__(self, config: IDMConfig, pretrained: bool = True):
        super(InverseDynamicsModel, self).__init__()

        self.config = config
        self.num_classes = config.num_classes
        self.input_channels = config.input_channels  # Should be 12 for 4 RGB frames

        # Load MobileNetV2
        if pretrained and config.mobilenet_pretrained:
            mobilenet = models.mobilenet_v2(pretrained=True)
        else:
            mobilenet = models.mobilenet_v2(pretrained=False)

        # MobileNetV2 structure:
        # - features: Sequential container with all conv layers and bottlenecks
        # - classifier: Sequential with dropout and final linear layer

        # Extract the feature extractor
        self.features = mobilenet.features

        # Modify first conv layer to accept custom number of input channels
        # Original first layer: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        original_conv = self.features[0][0]  # First conv in first Sequential block

        self.features[0][0] = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=original_conv.out_channels,  # 32 for standard MobileNetV2
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Initialize new conv weights
        if pretrained and config.mobilenet_pretrained and not config.use_grayscale:
            with torch.no_grad():
                # original_conv.weight shape: [32, 3, 3, 3]
                # We need to expand to [32, input_channels, 3, 3]
                original_weights = original_conv.weight.data

                if self.input_channels == 12:  # 4 RGB frames stacked
                    # Replicate the weights for each frame
                    for i in range(4):
                        self.features[0][0].weight.data[:, i*3:(i+1)*3, :, :] = original_weights
                elif self.input_channels == 6:  # 2 RGB frames stacked
                    self.features[0][0].weight.data[:, :3, :, :] = original_weights
                    self.features[0][0].weight.data[:, 3:, :, :] = original_weights
                elif self.input_channels == 3:  # Single RGB frame
                    self.features[0][0].weight.data = original_weights
                else:
                    # For other configurations, initialize randomly (kaiming)
                    nn.init.kaiming_normal_(self.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        else:
            # Initialize with kaiming normal
            nn.init.kaiming_normal_(self.features[0][0].weight, mode='fan_out', nonlinearity='relu')

        # MobileNetV2 outputs 1280 features (for width_mult=1.0)
        # The last convolutional layer outputs 1280 channels
        num_features = 1280

        # Global Average Pooling (adaptive to handle any spatial size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom Fully Connected Layers
        self.fc1 = nn.Linear(num_features, 256)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.fc3 = nn.Linear(128, self.num_classes)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

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
               For V2 default config: (batch_size, 12, 224, 224)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
            Note: Softmax is applied by CrossEntropyLoss during training
        """
        # MobileNetV2 feature extraction
        x = self.features(x)

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
