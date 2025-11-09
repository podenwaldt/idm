#!/usr/bin/env python3
"""
Create a test video from fake data for testing visualization.

This script takes the generated fake data frames and creates a video file
that can be used to test the visualization script.

Usage:
    python visualizations/create_test_video.py --data_dir data/test --output test_video.mp4
"""

import argparse
import cv2
import json
from pathlib import Path
from tqdm import tqdm


def create_video_from_frames(
    data_dir: str,
    output_path: str,
    fps: float = 30.0,
    codec: str = 'mp4v'
):
    """
    Create a video from frame images and inputs.json.

    Args:
        data_dir: Directory containing frames and inputs.json
        output_path: Path to output video file
        fps: Frames per second
        codec: Video codec (mp4v, XVID, etc.)
    """
    data_path = Path(data_dir)

    # Load inputs.json
    inputs_path = data_path / "inputs.json"
    if not inputs_path.exists():
        raise FileNotFoundError(f"inputs.json not found at {inputs_path}")

    with open(inputs_path, "r") as f:
        inputs = json.load(f)

    print(f"Creating video from {len(inputs)} frames...")

    # Get first frame to determine dimensions
    first_frame_path = data_path / f"frame_{inputs[0]['frame']:04d}.jpg"
    first_frame = cv2.imread(str(first_frame_path))

    if first_frame is None:
        raise ValueError(f"Cannot read first frame: {first_frame_path}")

    height, width = first_frame.shape[:2]

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {len(inputs)}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    # Write frames
    for entry in tqdm(inputs, desc="Writing frames"):
        frame_idx = entry['frame']
        frame_path = data_path / f"frame_{frame_idx:04d}.jpg"

        if not frame_path.exists():
            print(f"Warning: Frame {frame_idx} not found, skipping...")
            continue

        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)

    out.release()

    print(f"\n✓ Video created: {output_path}")
    print(f"\nYou can now visualize predictions with:")
    print(f"  python visualize_predictions.py \\")
    print(f"    --model_path checkpoints/idm_best.pth \\")
    print(f"    --video_path {output_path} \\")
    print(f"    --output_path predictions_visualization.mp4")


def main():
    parser = argparse.ArgumentParser(
        description="Create test video from fake data frames"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="fake_data/test",
        help="Directory containing frames and inputs.json (default: fake_data/test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_video.mp4",
        help="Output video path (default: test_video.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Video codec (default: mp4v)"
    )

    args = parser.parse_args()

    create_video_from_frames(
        data_dir=args.data_dir,
        output_path=args.output,
        fps=args.fps,
        codec=args.codec
    )


if __name__ == "__main__":
    main()
