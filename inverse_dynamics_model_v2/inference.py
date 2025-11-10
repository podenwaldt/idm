"""
Inference API for Inverse Dynamics Model.
"""

import os
from pathlib import Path
from typing import List, Union, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import InverseDynamicsModel
from .config import IDMConfig


class InverseDynamicsPredictor:
    """
    Simplified inference interface for the Inverse Dynamics Model.

    Args:
        model_path: Path to trained model checkpoint (.pth file)
        device: Device to run inference on ('cuda' or 'cpu')
        num_stacked_frames: Override number of stacked frames (for weights-only checkpoints)
        image_size: Override image size as (height, width) tuple (for weights-only checkpoints)
        use_grayscale: Override grayscale setting (for weights-only checkpoints)
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        num_stacked_frames: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        use_grayscale: Optional[bool] = None
    ):
        self.model_path = model_path

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Reconstruct config from checkpoint
        config_dict = checkpoint.get('config', {})
        self.config = self._create_config_from_dict(
            config_dict,
            num_stacked_frames_override=num_stacked_frames,
            image_size_override=image_size,
            use_grayscale_override=use_grayscale
        )

        # Create model
        self.model = InverseDynamicsModel(self.config, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Create preprocessing transform
        self.transform = self._create_transform()

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Input channels: {self.config.input_channels}")
        print(f"Image size: {self.config.image_size}")

    def _create_config_from_dict(
        self,
        config_dict: dict,
        num_stacked_frames_override: Optional[int] = None,
        image_size_override: Optional[Tuple[int, int]] = None,
        use_grayscale_override: Optional[bool] = None
    ) -> IDMConfig:
        """Create IDMConfig from checkpoint dictionary with optional overrides."""
        config = IDMConfig()

        # Override with saved config
        if 'num_stacked_frames' in config_dict:
            config.num_stacked_frames = config_dict['num_stacked_frames']
        if 'use_grayscale' in config_dict:
            config.use_grayscale = config_dict['use_grayscale']
        if 'image_size' in config_dict:
            config.image_size = tuple(config_dict['image_size'])
        if 'num_classes' in config_dict:
            config.num_classes = config_dict['num_classes']
        if 'dropout_rate' in config_dict:
            config.dropout_rate = config_dict['dropout_rate']

        # Apply manual overrides (takes precedence over checkpoint config)
        if num_stacked_frames_override is not None:
            config.num_stacked_frames = num_stacked_frames_override
        if image_size_override is not None:
            config.image_size = image_size_override
        if use_grayscale_override is not None:
            config.use_grayscale = use_grayscale_override

        return config

    def _create_transform(self) -> transforms.Compose:
        """Create preprocessing transform for inference."""
        transform_list = [
            transforms.Resize(self.config.image_size),
            transforms.ToTensor()
        ]

        if not self.config.use_grayscale:
            transform_list.append(transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ))

        return transforms.Compose(transform_list)

    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """
        Load and convert image to PIL Image.

        Args:
            image: Can be:
                - str: Path to image file
                - np.ndarray: Numpy array (H, W, C) or (H, W)
                - PIL.Image: PIL Image

        Returns:
            PIL Image in RGB or L mode
        """
        if isinstance(image, str):
            # Load from file path
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.ndim == 2:  # Grayscale
                img = Image.fromarray(image.astype(np.uint8), mode='L')
            elif image.ndim == 3:  # RGB or BGR
                if image.shape[2] == 3:
                    img = Image.fromarray(image.astype(np.uint8), mode='RGB')
                elif image.shape[2] == 4:  # RGBA
                    img = Image.fromarray(image.astype(np.uint8), mode='RGBA').convert('RGB')
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Convert to RGB or L
        if self.config.use_grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        return img

    def _preprocess_frames(
        self,
        frames: List[Union[str, np.ndarray, Image.Image]]
    ) -> torch.Tensor:
        """
        Preprocess a list of frames into model input.

        Args:
            frames: List of frames (must be num_stacked_frames long)

        Returns:
            Tensor of shape (1, input_channels, height, width)
        """
        if len(frames) != self.config.num_stacked_frames:
            raise ValueError(
                f"Expected {self.config.num_stacked_frames} frames, got {len(frames)}"
            )

        # Load and transform each frame
        processed_frames = []
        for frame in frames:
            img = self._load_image(frame)
            tensor = self.transform(img)
            processed_frames.append(tensor)

        # Stack frames along channel dimension
        stacked = torch.cat(processed_frames, dim=0)

        # Add batch dimension
        stacked = stacked.unsqueeze(0)

        return stacked

    def predict(
        self,
        frames: List[Union[str, np.ndarray, Image.Image]],
        return_probabilities: bool = False,
        top_k: Optional[int] = None
    ) -> Union[int, Tuple[int, np.ndarray], List[Tuple[int, float]]]:
        """
        Predict control state from a sequence of frames.

        For V2 with 4 frames [frame_0, frame_1, frame_2, frame_3], the model
        predicts the action at frame_1 (the SECOND frame), which caused the
        motion observed in frames 2 and 3.

        Args:
            frames: List of consecutive frames (num_stacked_frames long)
            return_probabilities: Whether to return probability distribution
            top_k: If set, return top K predictions with probabilities

        Returns:
            - If return_probabilities=False and top_k=None: Predicted state (int)
            - If return_probabilities=True: (predicted_state, probabilities array)
            - If top_k is set: List of (state, probability) tuples
        """
        # Preprocess
        input_tensor = self._preprocess_frames(frames)
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        # Get prediction
        predicted_state = torch.argmax(probabilities, dim=1).item()
        probs = probabilities.cpu().numpy()[0]

        # Return based on parameters
        if top_k is not None:
            # Return top K predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]
            return [(int(idx), float(probs[idx])) for idx in top_indices]
        elif return_probabilities:
            return predicted_state, probs
        else:
            return predicted_state

    def predict_from_paths(
        self,
        frame_paths: List[str],
        return_probabilities: bool = False
    ) -> Union[int, Tuple[int, np.ndarray]]:
        """
        Convenience method to predict from frame file paths.

        Args:
            frame_paths: List of paths to frame images
            return_probabilities: Whether to return probability distribution

        Returns:
            Predicted state or (state, probabilities)
        """
        return self.predict(frame_paths, return_probabilities=return_probabilities)

    def predict_batch(
        self,
        frame_sequences: List[List[Union[str, np.ndarray, Image.Image]]],
        return_probabilities: bool = False
    ) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Predict control states for a batch of frame sequences.

        Args:
            frame_sequences: List of frame sequences
            return_probabilities: Whether to return probability distributions

        Returns:
            List of predicted states or (states, probabilities array)
        """
        # Preprocess all sequences
        batch_tensors = []
        for frames in frame_sequences:
            tensor = self._preprocess_frames(frames)
            batch_tensors.append(tensor)

        # Stack into batch
        batch = torch.cat(batch_tensors, dim=0)
        batch = batch.to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)

        # Get predictions
        predicted_states = torch.argmax(probabilities, dim=1).cpu().numpy().tolist()
        probs = probabilities.cpu().numpy()

        if return_probabilities:
            return predicted_states, probs
        else:
            return predicted_states

    def get_state_name(self, state_id: int) -> str:
        """
        Get human-readable name for a state.

        Args:
            state_id: State index (0-4)

        Returns:
            State name string
        """
        return self.config.STATE_NAMES.get(state_id, f"UNKNOWN_{state_id}")

    def predict_with_names(
        self,
        frames: List[Union[str, np.ndarray, Image.Image]],
        return_probabilities: bool = False
    ) -> Union[str, Tuple[str, dict]]:
        """
        Predict and return state name instead of index.

        Args:
            frames: List of consecutive frames
            return_probabilities: Whether to return probability distribution

        Returns:
            State name or (state_name, {state_name: probability} dict)
        """
        if return_probabilities:
            state_id, probs = self.predict(frames, return_probabilities=True)
            state_name = self.get_state_name(state_id)
            prob_dict = {
                self.get_state_name(i): float(probs[i])
                for i in range(len(probs))
            }
            return state_name, prob_dict
        else:
            state_id = self.predict(frames)
            return self.get_state_name(state_id)

    def benchmark_inference_time(
        self,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> dict:
        """
        Benchmark inference time.

        Args:
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing statistics
        """
        import time

        # Create dummy input
        dummy_input = torch.randn(
            1,
            self.config.input_channels,
            *self.config.image_size
        ).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }


def load_model_for_inference(model_path: str, device: Optional[str] = None) -> InverseDynamicsPredictor:
    """
    Convenience function to load a trained model for inference.

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        InverseDynamicsPredictor instance
    """
    return InverseDynamicsPredictor(model_path, device)
