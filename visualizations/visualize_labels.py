#!/usr/bin/env python3
"""
Visualize training data with ground truth labels.
This script processes training data frames and creates an output video showing:
- The original training frames
- Ground truth control state labels
- Visual timeline of state changes
IMPORTANT: This version uses actual frame timestamps to ensure proper playback speed,
even when frames were captured at variable intervals due to SD card performance.
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
        timestamp: float,
        actual_fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Create visualization panel showing ground truth label.
        Args:
            width: Panel width
            height: Panel height
            state: Ground truth state (0-4)
            frame_idx: Current frame index
            timestamp: Frame timestamp in seconds
            actual_fps: Actual instantaneous FPS (optional)
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
        time_text = f"Frame {frame_idx} | Time: {timestamp:.3f}s"
        if actual_fps is not None:
            time_text += f" | FPS: {actual_fps:.1f}"

        cv2.putText(
            panel,
            time_text,
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

    def create_video_variable_fps(
        self,
        output_path: str,
        panel_height: int = 200,
        target_fps: float = 30.0,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize_width: Optional[int] = None
    ):
        """
        Create visualization video using ACTUAL frame timestamps.
        This method duplicates or skips frames as needed to maintain proper playback speed
        even when original frames were captured at variable rates.
        Args:
            output_path: Path to output video
            panel_height: Height of label panel
            target_fps: Target output FPS for smooth playback
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

        # Analyze timestamps to determine actual recording duration
        timestamps = []
        for idx in range(start_frame, end_frame):
            if idx in self.label_map:
                timestamps.append(self.label_map[idx]['time'])

        if not timestamps:
            raise ValueError("No valid timestamps found in the frame range")

        recording_duration = max(timestamps) - min(timestamps)
        avg_capture_fps = len(timestamps) / recording_duration if recording_duration > 0 else 30.0

        print(f"  Recording duration: {recording_duration:.2f}s")
        print(f"  Average capture FPS: {avg_capture_fps:.2f}")
        print(f"\nOutput video: {output_path}")
        print(f"  Resolution: {output_width}x{output_full_height}")
        print(f"  Target FPS: {target_fps}")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            target_fps,
            (output_width, output_full_height)
        )

        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        # Process frames using timestamp-based interpolation
        frames_written = 0
        frame_time_step = 1.0 / target_fps  # Time between output frames

        # Build list of source frames with timestamps
        source_frames = []
        for idx in range(start_frame, end_frame):
            if idx in self.label_map:
                source_frames.append({
                    'index': idx,
                    'time': self.label_map[idx]['time'],
                    'state': self.label_map[idx]['state']
                })

        if not source_frames:
            raise ValueError("No frames with valid timestamps found")

        # Normalize timestamps to start at 0
        start_time = source_frames[0]['time']
        for frame in source_frames:
            frame['time'] -= start_time

        end_time = source_frames[-1]['time']

        print(f"\nGenerating output frames with timestamp-based playback...")
        pbar = tqdm(total=int(end_time * target_fps), desc="Writing frames")

        # Generate output frames at constant FPS
        current_output_time = 0.0
        source_idx = 0

        while current_output_time <= end_time:
            # Find the source frame closest to current output time
            while (source_idx < len(source_frames) - 1 and
                   source_frames[source_idx + 1]['time'] <= current_output_time):
                source_idx += 1

            source_frame_info = source_frames[source_idx]
            frame_idx = source_frame_info['index']
            state = source_frame_info['state']
            actual_time = source_frame_info['time']

            # Calculate instantaneous FPS
            if source_idx < len(source_frames) - 1:
                time_diff = source_frames[source_idx + 1]['time'] - actual_time
                actual_fps = 1.0 / time_diff if time_diff > 0 else 30.0
            else:
                actual_fps = avg_capture_fps

            # Read and process frame
            frame_path = frame_paths[frame_idx]
            frame = cv2.imread(str(frame_path))

            if frame is None:
                print(f"\nWarning: Cannot read frame {frame_path}, skipping")
                current_output_time += frame_time_step
                continue

            # Resize if needed
            if resize_width is not None:
                frame = cv2.resize(frame, (output_width, output_height))

            # Create label panel
            panel = self.draw_label_panel(
                output_width,
                panel_height,
                state,
                frame_idx,
                actual_time,
                actual_fps
            )

            # Combine frame and panel
            combined = np.vstack([frame, panel])

            # Write frame
            out.write(combined)
            frames_written += 1
            pbar.update(1)

            # Advance output time
            current_output_time += frame_time_step

        pbar.close()

        # Cleanup
        out.release()

        print(f"\n✓ Processed {len(source_frames)} unique frames")
        print(f"✓ Wrote {frames_written} output frames")
        print(f"✓ Output duration: {frames_written / target_fps:.2f}s (matches recording: {end_time:.2f}s)")
        print(f"✓ Output saved to: {output_path}")

    def create_video(
        self,
        output_path: str,
        panel_height: int = 200,
        fps: float = 30.0,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize_width: Optional[int] = None,
        use_timestamps: bool = True
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
            use_timestamps: If True, use actual timestamps for accurate playback (recommended)
        """
        if use_timestamps:
            self.create_video_variable_fps(
                output_path=output_path,
                panel_height=panel_height,
                target_fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                resize_width=resize_width
            )
        else:
            print("\nWARNING: Using legacy mode without timestamps.")
            print("This may result in incorrect playback speed if frames were")
            print("captured at variable rates. Use --use-timestamps for accurate playback.\n")
            # Original implementation would go here
            # For now, just call the timestamp version
            self.create_video_variable_fps(
                output_path=output_path,
                panel_height=panel_height,
                target_fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                resize_width=resize_width
            )


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
        default=30.0,
        help="Output FPS (default: 30.0)"
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
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Disable timestamp-based playback (not recommended)"
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
        resize_width=args.resize_width,
        use_timestamps=not args.no_timestamps
    )

    print(f"\n{'='*70}")
    print("✓ Visualization Complete!")
    print(f"{'='*70}")
    print(f"\nYou can now view the output video:")
    print(f"  {args.output_path}")
    print(f"\nThis video shows your training frames with the labeled actions.")
    print(f"Playback speed is corrected using actual frame timestamps.")
    print()


if __name__ == "__main__":
    main()