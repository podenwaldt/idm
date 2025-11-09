#!/usr/bin/env python3
"""
Data Augmentation Script for RC Car Recordings

This script creates augmented versions of existing recordings by applying
various image transformations while preserving the label structure.

Usage:
    # Augment a single recording with brightness adjustment
    python augment_recordings.py recordings/train/recording_114838/ \\
        --output recordings/train/recording_114838_bright \\
        --brightness 1.3

    # Apply multiple augmentations
    python augment_recordings.py recordings/train/recording_114838/ \\
        --output recordings/train/recording_114838_aug \\
        --brightness 1.2 --contrast 1.1 --rotation 15

    # Batch augment multiple recordings
    python augment_recordings.py recordings/train/recording_* \\
        --output-dir recordings/train_augmented/ \\
        --brightness-range 0.7 1.3 --num-variants 3

    # Create dark/low-light version
    python augment_recordings.py data/train/ \\
        --output data/train_dark/ \\
        --brightness 0.5 --contrast 0.8

Author: Data Augmentation Tool
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

import cv2
import numpy as np
from tqdm import tqdm


class ImageAugmenter:
    """Applies various augmentation transformations to images."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.

        Args:
            image: Input image (BGR format)
            factor: Brightness factor (1.0 = no change, <1.0 = darker, >1.0 = brighter)

        Returns:
            Brightness-adjusted image
        """
        # Convert to HSV for better brightness control
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.

        Args:
            image: Input image (BGR format)
            factor: Contrast factor (1.0 = no change, <1.0 = lower, >1.0 = higher)

        Returns:
            Contrast-adjusted image
        """
        # Use formula: output = clip((input - 128) * factor + 128)
        image_float = image.astype(np.float32)
        adjusted = (image_float - 128) * factor + 128
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image saturation.

        Args:
            image: Input image (BGR format)
            factor: Saturation factor (1.0 = no change, 0.0 = grayscale, >1.0 = more saturated)

        Returns:
            Saturation-adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def adjust_hue(self, image: np.ndarray, shift: float) -> np.ndarray:
        """
        Adjust image hue.

        Args:
            image: Input image (BGR format)
            shift: Hue shift in degrees (-180 to 180)

        Returns:
            Hue-adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        # OpenCV uses 0-179 for hue
        hsv[:, :, 0] = (hsv[:, :, 0] + shift / 2) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.

        Args:
            image: Input image (BGR format)
            angle: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def add_gaussian_noise(self, image: np.ndarray, mean: float = 0,
                          std: float = 10) -> np.ndarray:
        """
        Add Gaussian noise to image.

        Args:
            image: Input image (BGR format)
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution

        Returns:
            Noisy image
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to image.

        Args:
            image: Input image (BGR format)
            kernel_size: Size of the Gaussian kernel (must be odd)

        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def add_motion_blur(self, image: np.ndarray, kernel_size: int = 9,
                       angle: float = 0) -> np.ndarray:
        """
        Apply motion blur to simulate camera movement.

        Args:
            image: Input image (BGR format)
            kernel_size: Size of the motion blur kernel
            angle: Direction of motion in degrees

        Returns:
            Motion-blurred image
        """
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Rotate kernel to desired angle
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

        return cv2.filter2D(image, -1, kernel)

    def adjust_gamma(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction.

        Args:
            image: Input image (BGR format)
            gamma: Gamma value (1.0 = no change, <1.0 = brighter, >1.0 = darker)

        Returns:
            Gamma-corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)


class RecordingAugmenter:
    """Augments entire recordings with transformations."""

    def __init__(self, augmenter: ImageAugmenter):
        """
        Initialize the recording augmenter.

        Args:
            augmenter: ImageAugmenter instance to use
        """
        self.augmenter = augmenter

    def augment_recording(self,
                         input_dir: Path,
                         output_dir: Path,
                         transformations: Dict[str, Any],
                         jpeg_quality: int = 95,
                         preserve_metadata: bool = True,
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Augment all frames in a recording.

        Args:
            input_dir: Input recording directory
            output_dir: Output directory for augmented recording
            transformations: Dictionary of transformation parameters
            jpeg_quality: JPEG compression quality (0-100)
            preserve_metadata: Whether to copy metadata.json
            verbose: Whether to show progress bar

        Returns:
            Dictionary with augmentation statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Validate input directory
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all frame images
        frame_files = sorted(list(input_dir.glob("frame_*.jpg")))
        if not frame_files:
            # Try PNG format
            frame_files = sorted(list(input_dir.glob("frame_*.png")))

        if not frame_files:
            raise ValueError(f"No frame images found in {input_dir}")

        # Copy and augment frames
        stats = {
            'total_frames': len(frame_files),
            'successful': 0,
            'failed': 0,
            'transformations': transformations
        }

        iterator = tqdm(frame_files, desc="Augmenting frames") if verbose else frame_files

        for frame_file in iterator:
            try:
                # Load image
                image = cv2.imread(str(frame_file))
                if image is None:
                    print(f"Warning: Could not load {frame_file}")
                    stats['failed'] += 1
                    continue

                # Apply transformations
                augmented = self._apply_transformations(image, transformations)

                # Save augmented image
                output_path = output_dir / frame_file.name
                cv2.imwrite(str(output_path), augmented,
                           [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

                stats['successful'] += 1

            except Exception as e:
                print(f"Error processing {frame_file}: {e}")
                stats['failed'] += 1

        # Copy inputs.json if exists
        inputs_file = input_dir / "inputs.json"
        if inputs_file.exists():
            shutil.copy2(inputs_file, output_dir / "inputs.json")

        # Copy or update metadata.json
        if preserve_metadata:
            metadata_file = input_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Add augmentation info
                metadata['augmented'] = True
                metadata['augmentation_params'] = transformations
                metadata['source_recording'] = str(input_dir)

                with open(output_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

        return stats

    def _apply_transformations(self, image: np.ndarray,
                               transformations: Dict[str, Any]) -> np.ndarray:
        """
        Apply a set of transformations to an image.

        Args:
            image: Input image
            transformations: Dictionary of transformation parameters

        Returns:
            Transformed image
        """
        result = image.copy()

        # Apply transformations in order
        if 'brightness' in transformations:
            result = self.augmenter.adjust_brightness(result, transformations['brightness'])

        if 'contrast' in transformations:
            result = self.augmenter.adjust_contrast(result, transformations['contrast'])

        if 'saturation' in transformations:
            result = self.augmenter.adjust_saturation(result, transformations['saturation'])

        if 'hue' in transformations:
            result = self.augmenter.adjust_hue(result, transformations['hue'])

        if 'gamma' in transformations:
            result = self.augmenter.adjust_gamma(result, transformations['gamma'])

        if 'rotation' in transformations:
            result = self.augmenter.rotate(result, transformations['rotation'])

        if 'blur' in transformations:
            result = self.augmenter.add_blur(result, transformations['blur'])

        if 'motion_blur' in transformations:
            result = self.augmenter.add_motion_blur(
                result,
                transformations['motion_blur'],
                transformations.get('motion_blur_angle', 0)
            )

        if 'noise_std' in transformations:
            result = self.augmenter.add_gaussian_noise(
                result,
                mean=transformations.get('noise_mean', 0),
                std=transformations['noise_std']
            )

        return result


def create_augmentation_variants(input_dirs: List[Path],
                                output_base_dir: Path,
                                brightness_range: Tuple[float, float],
                                num_variants: int,
                                seed: Optional[int] = None,
                                **kwargs) -> List[Dict[str, Any]]:
    """
    Create multiple augmented variants of recordings with random parameters.

    Args:
        input_dirs: List of input recording directories
        output_base_dir: Base directory for output
        brightness_range: Range for random brightness (min, max)
        num_variants: Number of variants to create per recording
        seed: Random seed
        **kwargs: Additional fixed transformation parameters

    Returns:
        List of statistics for each variant
    """
    if seed is not None:
        np.random.seed(seed)

    augmenter = ImageAugmenter(seed=seed)
    rec_augmenter = RecordingAugmenter(augmenter)

    output_base_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        recording_name = input_dir.name

        for variant_idx in range(num_variants):
            # Generate random brightness
            brightness = np.random.uniform(*brightness_range)

            # Create transformations dict
            transformations = {
                'brightness': brightness,
                **kwargs
            }

            # Create output directory
            output_dir = output_base_dir / f"{recording_name}_var{variant_idx + 1}"

            print(f"\nCreating variant {variant_idx + 1}/{num_variants} for {recording_name}")
            print(f"Transformations: {transformations}")

            # Augment recording
            stats = rec_augmenter.augment_recording(
                input_dir, output_dir, transformations
            )
            stats['variant_idx'] = variant_idx + 1
            stats['input_recording'] = str(input_dir)
            stats['output_recording'] = str(output_dir)
            all_stats.append(stats)

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Augment RC car recordings with image transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output arguments
    parser.add_argument('input', nargs='+', type=str,
                       help='Input recording directory/directories')
    parser.add_argument('--output', type=str,
                       help='Output directory (for single input)')
    parser.add_argument('--output-dir', type=str,
                       help='Output base directory (for multiple inputs)')

    # Transformation parameters
    parser.add_argument('--brightness', type=float,
                       help='Brightness factor (1.0 = no change, <1.0 = darker, >1.0 = brighter)')
    parser.add_argument('--contrast', type=float,
                       help='Contrast factor (1.0 = no change)')
    parser.add_argument('--saturation', type=float,
                       help='Saturation factor (1.0 = no change, 0.0 = grayscale)')
    parser.add_argument('--hue', type=float,
                       help='Hue shift in degrees (-180 to 180)')
    parser.add_argument('--gamma', type=float,
                       help='Gamma correction value (1.0 = no change)')
    parser.add_argument('--rotation', type=float,
                       help='Rotation angle in degrees')
    parser.add_argument('--blur', type=int,
                       help='Gaussian blur kernel size (odd number)')
    parser.add_argument('--motion-blur', type=int,
                       help='Motion blur kernel size')
    parser.add_argument('--motion-blur-angle', type=float, default=0,
                       help='Motion blur angle in degrees')
    parser.add_argument('--noise-std', type=float,
                       help='Standard deviation of Gaussian noise')
    parser.add_argument('--noise-mean', type=float, default=0,
                       help='Mean of Gaussian noise')

    # Batch augmentation
    parser.add_argument('--brightness-range', nargs=2, type=float,
                       metavar=('MIN', 'MAX'),
                       help='Random brightness range for batch augmentation')
    parser.add_argument('--num-variants', type=int, default=1,
                       help='Number of augmented variants to create')

    # Other options
    parser.add_argument('--jpeg-quality', type=int, default=95,
                       help='JPEG compression quality (0-100, default: 95)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Disable progress bars')

    args = parser.parse_args()

    # Validate arguments
    if len(args.input) == 1 and not args.output and not args.output_dir:
        parser.error("For single input, specify --output. For multiple inputs, specify --output-dir")

    if len(args.input) > 1 and not args.output_dir:
        parser.error("For multiple inputs, --output-dir is required")

    # Build transformations dictionary
    transformations = {}
    if args.brightness is not None:
        transformations['brightness'] = args.brightness
    if args.contrast is not None:
        transformations['contrast'] = args.contrast
    if args.saturation is not None:
        transformations['saturation'] = args.saturation
    if args.hue is not None:
        transformations['hue'] = args.hue
    if args.gamma is not None:
        transformations['gamma'] = args.gamma
    if args.rotation is not None:
        transformations['rotation'] = args.rotation
    if args.blur is not None:
        transformations['blur'] = args.blur
    if args.motion_blur is not None:
        transformations['motion_blur'] = args.motion_blur
        transformations['motion_blur_angle'] = args.motion_blur_angle
    if args.noise_std is not None:
        transformations['noise_std'] = args.noise_std
        transformations['noise_mean'] = args.noise_mean

    # Check if batch augmentation mode
    if args.brightness_range and args.num_variants > 1:
        # Batch mode with random variants
        input_paths = [Path(p) for p in args.input]
        output_base = Path(args.output_dir or args.output)

        print(f"Creating {args.num_variants} variants with brightness range {args.brightness_range}")
        print(f"Processing {len(input_paths)} recordings...")

        all_stats = create_augmentation_variants(
            input_paths,
            output_base,
            tuple(args.brightness_range),
            args.num_variants,
            seed=args.seed,
            **{k: v for k, v in transformations.items() if k != 'brightness'}
        )

        # Print summary
        print("\n" + "=" * 60)
        print("AUGMENTATION SUMMARY")
        print("=" * 60)
        total_successful = sum(s['successful'] for s in all_stats)
        total_failed = sum(s['failed'] for s in all_stats)
        print(f"Total variants created: {len(all_stats)}")
        print(f"Total frames processed: {total_successful}")
        print(f"Failed frames: {total_failed}")

    else:
        # Single/multiple recordings with fixed transformations
        if not transformations:
            parser.error("No transformations specified. Use --brightness, --contrast, etc.")

        augmenter = ImageAugmenter(seed=args.seed)
        rec_augmenter = RecordingAugmenter(augmenter)

        if len(args.input) == 1 and args.output:
            # Single recording
            input_dir = Path(args.input[0])
            output_dir = Path(args.output)

            print(f"Augmenting: {input_dir} -> {output_dir}")
            print(f"Transformations: {transformations}")

            stats = rec_augmenter.augment_recording(
                input_dir, output_dir, transformations,
                jpeg_quality=args.jpeg_quality,
                verbose=not args.no_verbose
            )

            print("\n" + "=" * 60)
            print("AUGMENTATION COMPLETE")
            print("=" * 60)
            print(f"Input: {input_dir}")
            print(f"Output: {output_dir}")
            print(f"Successful: {stats['successful']}/{stats['total_frames']}")
            print(f"Failed: {stats['failed']}")

        else:
            # Multiple recordings
            output_base = Path(args.output_dir)

            for input_path in args.input:
                input_dir = Path(input_path)
                output_dir = output_base / input_dir.name

                print(f"\nAugmenting: {input_dir} -> {output_dir}")

                stats = rec_augmenter.augment_recording(
                    input_dir, output_dir, transformations,
                    jpeg_quality=args.jpeg_quality,
                    verbose=not args.no_verbose
                )

                print(f"Successful: {stats['successful']}/{stats['total_frames']}")


if __name__ == "__main__":
    main()
