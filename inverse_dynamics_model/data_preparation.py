"""
Data preparation utilities for converting video to frames and generating inputs.json
"""

import os
import json
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse


class VideoToFramesConverter:
    """
    Convert video files to frames and generate inputs.json for training.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames and inputs.json
        fps: Target frames per second (None = original fps)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (None = entire video)
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        fps: Optional[float] = None,
        start_time: float = 0,
        end_time: Optional[float] = None
    ):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.target_fps = fps
        self.start_time = start_time
        self.end_time = end_time

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.original_fps

        print(f"Video: {video_path}")
        print(f"  FPS: {self.original_fps}")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Duration: {self.duration:.2f}s")

        # Determine actual fps to use
        if self.target_fps is None:
            self.target_fps = self.original_fps

        self.frame_interval = int(self.original_fps / self.target_fps)

    def extract_frames(self) -> int:
        """
        Extract frames from video.

        Returns:
            Number of frames extracted
        """
        # Calculate start and end frame indices
        start_frame = int(self.start_time * self.original_fps)
        end_frame = int(self.end_time * self.original_fps) if self.end_time else self.total_frames

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        saved_count = 0

        print(f"\nExtracting frames...")
        print(f"  Start frame: {start_frame}")
        print(f"  End frame: {end_frame}")
        print(f"  Frame interval: {self.frame_interval}")

        while True:
            ret, frame = self.cap.read()
            if not ret or (start_frame + frame_count) >= end_frame:
                break

            # Save frame at specified interval
            if frame_count % self.frame_interval == 0:
                frame_path = self.output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1

                if saved_count % 100 == 0:
                    print(f"  Extracted {saved_count} frames...")

            frame_count += 1

        self.cap.release()

        print(f"\nExtracted {saved_count} frames to {self.output_dir}")
        return saved_count

    def create_inputs_json_template(
        self,
        num_frames: int,
        default_state: int = 0
    ):
        """
        Create a template inputs.json with default states.

        You should manually edit this file to assign correct states.

        Args:
            num_frames: Number of frames
            default_state: Default state to assign (0 = STOPPED)
        """
        inputs = []

        for frame_idx in range(num_frames):
            time = frame_idx / self.target_fps
            inputs.append({
                "frame": frame_idx,
                "time": round(time, 3),
                "state": default_state
            })

        inputs_path = self.output_dir / "inputs.json"
        with open(inputs_path, "w") as f:
            json.dump(inputs, f, indent=2)

        print(f"\nTemplate inputs.json created: {inputs_path}")
        print(f"Please edit this file to assign correct states:")
        print(f"  0 = STOPPED")
        print(f"  1 = FORWARD")
        print(f"  2 = BACKWARD")
        print(f"  3 = ROTATE_LEFT")
        print(f"  4 = ROTATE_RIGHT")


def convert_video_to_dataset(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = 30,
    start_time: float = 0,
    end_time: Optional[float] = None
):
    """
    Convert a video to a dataset directory with frames and template inputs.json.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames and inputs.json
        fps: Target frames per second
        start_time: Start time in seconds
        end_time: End time in seconds (None = entire video)
    """
    converter = VideoToFramesConverter(
        video_path,
        output_dir,
        fps=fps,
        start_time=start_time,
        end_time=end_time
    )

    num_frames = converter.extract_frames()
    converter.create_inputs_json_template(num_frames)

    print(f"\n{'='*70}")
    print("Video conversion complete!")
    print(f"{'='*70}")
    print(f"Next steps:")
    print(f"1. Edit {output_dir}/inputs.json to assign correct states")
    print(f"2. Verify frames are correct")
    print(f"3. Split data into train/val/test sets if needed")
    print(f"{'='*70}\n")


def split_dataset(
    source_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_base_dir: Optional[str] = None
):
    """
    Split a dataset directory into train/val/test sets.

    Args:
        source_dir: Source directory with frames and inputs.json
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        output_base_dir: Base directory for output (default: source_dir/../data)
    """
    source_path = Path(source_dir)

    if output_base_dir is None:
        output_base_dir = source_path.parent / "data"
    else:
        output_base_dir = Path(output_base_dir)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Load inputs.json
    inputs_path = source_path / "inputs.json"
    if not inputs_path.exists():
        raise FileNotFoundError(f"inputs.json not found in {source_dir}")

    with open(inputs_path, "r") as f:
        inputs = json.load(f)

    num_samples = len(inputs)

    # Calculate split indices
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    splits = {
        "train": inputs[:train_end],
        "val": inputs[train_end:val_end],
        "test": inputs[val_end:]
    }

    print(f"Splitting dataset:")
    print(f"  Total samples: {num_samples}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")

    # Create split directories and copy data
    for split_name, split_data in splits.items():
        split_dir = output_base_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save split inputs.json
        split_inputs_path = split_dir / "inputs.json"
        with open(split_inputs_path, "w") as f:
            json.dump(split_data, f, indent=2)

        # Copy frames
        print(f"\nCopying {split_name} frames...")
        for entry in split_data:
            frame_idx = entry["frame"]
            src_frame = source_path / f"frame_{frame_idx:04d}.jpg"
            dst_frame = split_dir / f"frame_{frame_idx:04d}.jpg"

            if src_frame.exists():
                import shutil
                shutil.copy2(src_frame, dst_frame)

    print(f"\n{'='*70}")
    print(f"Dataset split complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_base_dir}")
    print(f"  train/: {len(splits['train'])} samples")
    print(f"  val/:   {len(splits['val'])} samples")
    print(f"  test/:  {len(splits['test'])} samples")
    print(f"{'='*70}\n")


def validate_dataset(dataset_dir: str):
    """
    Validate a dataset directory structure and inputs.json.

    Args:
        dataset_dir: Path to dataset directory
    """
    dataset_path = Path(dataset_dir)

    print(f"Validating dataset: {dataset_dir}")

    # Check inputs.json exists
    inputs_path = dataset_path / "inputs.json"
    if not inputs_path.exists():
        print(f"  ❌ inputs.json not found")
        return False

    # Load and validate inputs.json
    with open(inputs_path, "r") as f:
        inputs = json.load(f)

    print(f"  ✓ inputs.json loaded ({len(inputs)} entries)")

    # Check all frames exist
    missing_frames = []
    for entry in inputs:
        frame_idx = entry["frame"]
        frame_path = dataset_path / f"frame_{frame_idx:04d}.jpg"

        if not frame_path.exists():
            # Try .png extension
            frame_path = dataset_path / f"frame_{frame_idx:04d}.png"

        if not frame_path.exists():
            missing_frames.append(frame_idx)

    if missing_frames:
        print(f"  ❌ Missing frames: {len(missing_frames)}")
        print(f"     First 10: {missing_frames[:10]}")
        return False
    else:
        print(f"  ✓ All frames present")

    # Check state distribution
    state_counts = {}
    for entry in inputs:
        state = entry["state"]
        state_counts[state] = state_counts.get(state, 0) + 1

    print(f"  State distribution:")
    state_names = {0: "STOPPED", 1: "FORWARD", 2: "BACKWARD", 3: "ROTATE_LEFT", 4: "ROTATE_RIGHT"}
    for state_id in sorted(state_counts.keys()):
        count = state_counts[state_id]
        percentage = (count / len(inputs)) * 100
        state_name = state_names.get(state_id, f"UNKNOWN_{state_id}")
        print(f"    {state_name:15s}: {count:5d} ({percentage:5.2f}%)")

    print(f"  ✓ Dataset validation passed\n")
    return True


def main():
    """Command-line interface for data preparation."""
    parser = argparse.ArgumentParser(description="Data preparation for Inverse Dynamics Model")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert video command
    convert_parser = subparsers.add_parser("convert", help="Convert video to frames")
    convert_parser.add_argument("video_path", help="Path to video file")
    convert_parser.add_argument("output_dir", help="Output directory")
    convert_parser.add_argument("--fps", type=float, default=30, help="Target FPS (default: 30)")
    convert_parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    convert_parser.add_argument("--end", type=float, default=None, help="End time in seconds")

    # Split dataset command
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("source_dir", help="Source directory with frames and inputs.json")
    split_parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default: 0.7)")
    split_parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    split_parser.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    split_parser.add_argument("--output", help="Output base directory")

    # Validate dataset command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("dataset_dir", help="Dataset directory to validate")

    args = parser.parse_args()

    if args.command == "convert":
        convert_video_to_dataset(
            args.video_path,
            args.output_dir,
            fps=args.fps,
            start_time=args.start,
            end_time=args.end
        )
    elif args.command == "split":
        split_dataset(
            args.source_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            output_base_dir=args.output
        )
    elif args.command == "validate":
        validate_dataset(args.dataset_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
