#!/usr/bin/env python3
"""
Visualize training data with ground truth labels.

This script processes training data frames and creates an output video showing:
- The original training frames
- Ground truth control state labels
- Visual timeline of state changes

This is useful for validating that training labels are correctly aligned with video footage.

Usage:
    python visualize_labels.py \
        --data_path data/train \
        --output_path labels_visualization.mp4
"""

import argparse
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm


class LabelVisualizer:
    """
    Create video visualizations of ground truth labels.

    Args:
        data_path: Path to data directory containing frames and inputs.json
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

        # Load labels from inputs.json
        inputs_json_path = self.data_path / "inputs.json"
        if not inputs_json_path.exists():
            raise ValueError(f"inputs.json not found in {data_path}")

        with open(inputs_json_path, 'r') as f:
            self.labels = json.load(f)

        print(f"Loaded {len(self.labels)} labels from {inputs_json_path}")

        # Create frame index to label mapping
        self.label_map = {entry['frame']: entry for entry in self.labels}

        # State configuration
        self.STATE_NAMES = {
            0: "STOPPED",
            1: "FORWARD",
            2: "BACKWARD",
            3: "ROTATE_LEFT",
            4: "ROTATE_RIGHT"
        }

        # Colors for each state (BGR format for OpenCV)
        self.state_colors = {
            0: (128, 128, 128),  # STOPPED - gray
            1: (0, 255, 0),      # FORWARD - green
            2: (0, 0, 255),      # BACKWARD - red
            3: (255, 0, 0),      # ROTATE_LEFT - blue
            4: (0, 255, 255),    # ROTATE_RIGHT - yellow
        }

    def get_frame_paths(self) -> List[Path]:
        """Get sorted list of frame image paths."""
        frame_paths = sorted(self.data_path.glob("frame_*.jpg"))
        if not frame_paths:
            raise ValueError(f"No frame images found in {self.data_path}")
        return frame_paths

    def draw_label_panel(
        self,
        width: int,
        height: int,
        state: int,
        frame_idx: int,
        timestamp: float
    ) -> np.ndarray:
        """
        Create visualization panel showing ground truth label.

        Args:
            width: Panel width
            height: Panel height
            state: Ground truth state (0-4)
            frame_idx: Current frame index
            timestamp: Frame timestamp in seconds

        Returns:
            BGR image array
        """
        # Create black background
        panel = np.zeros((height, width, 3), dtype=np.uint8)

        # Get state info
        state_name = self.STATE_NAMES[state]
        state_color = self.state_colors[state]

        # Define layout
        title_height = 80
        info_section_height = 100

        # Draw header section
        cv2.rectangle(panel, (0, 0), (width, title_height), (50, 50, 50), -1)

        # Draw title
        cv2.putText(
            panel,
            "Ground Truth Label",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2
        )

        # Draw frame and time info
        cv2.putText(
            panel,
            f"Frame {frame_idx} | Time: {timestamp:.3f}s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1
        )

        # Draw state name prominently
        y_offset = title_height + 60

        # State label background
        label_box_height = 60
        cv2.rectangle(
            panel,
            (20, y_offset - 45),
            (width - 20, y_offset + 15),
            state_color,
            -1
        )

        # State text
        cv2.putText(
            panel,
            state_name,
            (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        return panel

    def create_video(
        self,
        output_path: str,
        panel_height: int = 200,
        fps: float = 12.0,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize_width: Optional[int] = None
    ):
        """
        Create visualization video from training data.

        Args:
            output_path: Path to output video
            panel_height: Height of label panel
            fps: Output FPS
            start_frame: First frame to process
            end_frame: Last frame to process (None = all)
            resize_width: Resize video width (maintains aspect ratio)
        """
        # Get all frame paths
        frame_paths = self.get_frame_paths()

        if end_frame is None:
            end_frame = len(frame_paths)

        # Validate frame range
        if start_frame >= len(frame_paths):
            raise ValueError(f"start_frame {start_frame} exceeds available frames {len(frame_paths)}")

        end_frame = min(end_frame, len(frame_paths))

        print(f"\nInput data: {self.data_path}")
        print(f"  Total frames: {len(frame_paths)}")
        print(f"  Processing frames: {start_frame} to {end_frame}")

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Cannot read first frame: {frame_paths[0]}")

        frame_height, frame_width = first_frame.shape[:2]

        print(f"  Frame resolution: {frame_width}x{frame_height}")

        # Calculate output dimensions
        if resize_width is not None:
            output_width = resize_width
            output_height = int(frame_height * (resize_width / frame_width))
        else:
            output_width = frame_width
            output_height = frame_height

        output_full_height = output_height + panel_height

        print(f"\nOutput video: {output_path}")
        print(f"  Resolution: {output_width}x{output_full_height}")
        print(f"  FPS: {fps}")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (output_width, output_full_height)
        )

        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        # Process frames
        frames_processed = 0

        pbar = tqdm(
            total=end_frame - start_frame,
            desc="Processing frames"
        )

        for idx in range(start_frame, end_frame):
            # Read frame
            frame_path = frame_paths[idx]
            frame = cv2.imread(str(frame_path))

            if frame is None:
                print(f"\nWarning: Cannot read frame {frame_path}, skipping")
                continue

            # Resize if needed
            if resize_width is not None:
                frame = cv2.resize(frame, (output_width, output_height))

            # Get label for this frame
            if idx in self.label_map:
                label_info = self.label_map[idx]
                state = label_info['state']
                timestamp = label_info['time']
            else:
                # If no label for this frame, show as unknown
                print(f"\nWarning: No label found for frame {idx}")
                state = 0  # Default to STOPPED
                timestamp = 0.0

            # Create label panel
            panel = self.draw_label_panel(
                output_width,
                panel_height,
                state,
                idx,
                timestamp
            )

            # Combine frame and panel (frame on top, labels below)
            combined = np.vstack([frame, panel])

            # Write frame
            out.write(combined)

            frames_processed += 1
            pbar.update(1)

        pbar.close()

        # Cleanup
        out.release()

        print(f"\n✓ Processed {frames_processed} frames")
        print(f"✓ Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training data with ground truth labels"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory containing frames and inputs.json"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="labels_visualization.mp4",
        help="Path to output video (default: labels_visualization.mp4)"
    )
    parser.add_argument(
        "--panel_height",
        type=int,
        default=200,
        help="Height of label panel in pixels (default: 200)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Output FPS (default: 12.0)"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First frame to process (default: 0)"
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Last frame to process (default: all)"
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Resize video to this width (maintains aspect ratio)"
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Training Data Label Visualization")
    print(f"{'='*70}\n")

    # Create visualizer
    visualizer = LabelVisualizer(args.data_path)

    # Create video
    visualizer.create_video(
        output_path=args.output_path,
        panel_height=args.panel_height,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        resize_width=args.resize_width
    )

    print(f"\n{'='*70}")
    print("✓ Visualization Complete!")
    print(f"{'='*70}")
    print(f"\nYou can now view the output video:")
    print(f"  {args.output_path}")
    print(f"\nThis video shows your training frames with the labeled actions.")
    print(f"Use this to validate that your labels are correctly aligned with the footage.")
    print()


if __name__ == "__main__":
    main()
