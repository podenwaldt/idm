#!/usr/bin/env python3
"""
Convert a training checkpoint to inference format.

Usage:
    python convert_checkpoint.py --input checkpoints/idm_epoch_10_best.pth --output checkpoints/idm_epoch_10_inference.pth
"""

import argparse
import torch
import os


def convert_checkpoint(input_path: str, output_path: str):
    """
    Convert training checkpoint to inference-only format.

    Args:
        input_path: Path to training checkpoint (with optimizer state, etc.)
        output_path: Path to save inference checkpoint (model weights only)
    """
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Extract only what's needed for inference
    inference_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'config': checkpoint['config']
    }

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save inference checkpoint
    torch.save(inference_checkpoint, output_path)
    print(f"Inference checkpoint saved to: {output_path}")

    # Print size comparison
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSize comparison:")
    print(f"  Training checkpoint: {input_size:.2f} MB")
    print(f"  Inference checkpoint: {output_size:.2f} MB")
    print(f"  Reduction: {input_size - output_size:.2f} MB ({(1 - output_size/input_size)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert training checkpoint to inference format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to training checkpoint (e.g., checkpoints/idm_epoch_10_best.pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save inference checkpoint (e.g., checkpoints/idm_best_inference.pth)"
    )

    args = parser.parse_args()

    # Verify input exists
    if not os.path.exists(args.input):
        print(f"Error: Input checkpoint not found: {args.input}")
        return

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
