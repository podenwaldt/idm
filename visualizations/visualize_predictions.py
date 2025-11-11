#!/usr/bin/env python3
"""
Visualize model predictions on video.

This script processes a video file, runs inference on consecutive frames,
and creates an output video showing:
- The original video frames
- Predicted control state and confidence
- Probability distribution over all states

Supports models with 2, 3, or 4 stacked frames (automatically detected from checkpoint).

Usage:
    # V2 model with full checkpoint (config included)
    python visualizations/visualize_predictions.py \
        --model_path idm_final.pth \
        --model_version v2 \
        --video_path test_video.mp4 \
        --output_path predictions_visualization.mp4

    # V1 model with weights-only checkpoint (manual config required)
    python visualizations/visualize_predictions.py \
        --model_path idm_final.pth \
        --model_version v1 \
        --video_path test_video.mp4 \
        --output_path predictions_visualization.mp4 \
        --num_stacked_frames 2 \
        --image_size 224 224

    python visualizations/visualize_predictions.py \
        --model_path idm_final_MN_3F_V2.pth \
        --model_version v2 \
        --video_path recordings/demo/GT.mp4\
        --output_path visualizations/outputs/demo_preds_GT.mp4 \
        --num_stacked_frames 3

    python visualizations/visualize_predictions.py \
        --model_path idm_final_MN_3F_V2.pth \
        --model_version v2 \
        --video_path recordings/demo/demo_385.mp4\
        --output_path visualizations/outputs/demo_preds_385.mp4 \
        --num_stacked_frames 3 \
        --fps 6

    python visualizations/visualize_predictions.py \
        --model_path idm_final_RN_3F.pth \
        --model_version v1 \
        --video_path recordings/demo/944Track.mp4\
        --output_path visualizations/outputs/demo_preds_944_RN.mp4 \
        --num_stacked_frames 3

    python visualizations/visualize_predictions.py \
        --model_path idm_final_RN_3F.pth \
        --model_version v1 \
        --video_path recordings/demo/demo_hallway.mp4\
        --output_path visualizations/outputs/demo_preds_hallway_RN.mp4 \
        --num_stacked_frames 3 \
        --fps 6

"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from collections import deque
import torch
from PIL import Image
from tqdm import tqdm

import inverse_dynamics_model.inference as idm_v1
import inverse_dynamics_model_v2.inference as idm_v2


class PredictionVisualizer:
    """
    Create video visualizations of model predictions.

    Args:
        model_path: Path to trained model checkpoint
        model_version: Model version ('v1' or 'v2')
        device: Device to run inference on
        num_stacked_frames: Override number of stacked frames (for weights-only checkpoints)
        image_size: Override image size as (height, width) tuple (for weights-only checkpoints)
        use_grayscale: Override grayscale setting (for weights-only checkpoints)
    """

    def __init__(
        self,
        model_path: str,
        model_version: str = 'v2',
        device: Optional[str] = None,
        num_stacked_frames: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        use_grayscale: Optional[bool] = None
    ):
        # Select the appropriate predictor class based on model version
        if model_version == 'v1':
            predictor_class = idm_v1.InverseDynamicsPredictor
        elif model_version == 'v2':
            predictor_class = idm_v2.InverseDynamicsPredictor
        else:
            raise ValueError(f"Invalid model_version: {model_version}. Must be 'v1' or 'v2'")

        self.predictor = predictor_class(
            model_path,
            device=device,
            num_stacked_frames=num_stacked_frames,
            image_size=image_size,
            use_grayscale=use_grayscale
        )
        self.config = self.predictor.config

        # Colors for each state (BGR format for OpenCV)
        self.state_colors = {
            0: (128, 128, 128),  # STOPPED - gray
            1: (0, 255, 0),      # FORWARD - green
            2: (0, 0, 255),      # BACKWARD - red
            3: (255, 0, 0),      # ROTATE_LEFT - blue
            4: (0, 255, 255),    # ROTATE_RIGHT - yellow
        }

    def draw_prediction_panel(
        self,
        width: int,
        height: int,
        state: int,
        probabilities: np.ndarray,
        frame_idx: int
    ) -> np.ndarray:
        """
        Create visualization panel showing predictions.

        Args:
            width: Panel width
            height: Panel height
            state: Predicted state (0-4)
            probabilities: Probability distribution
            frame_idx: Current frame index

        Returns:
            BGR image array
        """
        # Create black background
        panel = np.zeros((height, width, 3), dtype=np.uint8)

        # Get state info
        state_name = self.config.STATE_NAMES[state]
        confidence = probabilities[state]

        # Define layout
        title_height = 60
        bars_height = height - title_height - 40
        bar_spacing = 10
        bar_height = (bars_height - (len(probabilities) - 1) * bar_spacing) // len(probabilities)

        # Draw title section
        cv2.rectangle(panel, (0, 0), (width, title_height), (50, 50, 50), -1)

        # Draw frame number
        cv2.putText(
            panel,
            f"Frame {frame_idx}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

        # Draw predicted state and confidence
        state_color = self.state_colors[state]
        cv2.putText(
            panel,
            f"{state_name}",
            (20, title_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            state_color,
            2
        )

        cv2.putText(
            panel,
            f"{confidence:.1%}",
            (width - 120, title_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Draw probability bars for each state
        y_offset = title_height + 20
        max_bar_width = width - 220

        for i in range(len(probabilities)):
            prob = probabilities[i]
            name = self.config.STATE_NAMES[i]
            color = self.state_colors[i]

            # State label
            cv2.putText(
                panel,
                name,
                (20, y_offset + bar_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Probability bar background
            cv2.rectangle(
                panel,
                (160, y_offset),
                (160 + max_bar_width, y_offset + bar_height),
                (40, 40, 40),
                -1
            )

            # Probability bar
            bar_width = int(max_bar_width * prob)
            if bar_width > 0:
                cv2.rectangle(
                    panel,
                    (160, y_offset),
                    (160 + bar_width, y_offset + bar_height),
                    color,
                    -1
                )

            # Probability percentage
            cv2.putText(
                panel,
                f"{prob:.1%}",
                (160 + max_bar_width + 10, y_offset + bar_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            y_offset += bar_height + bar_spacing

        return panel

    def process_video(
        self,
        video_path: str,
        output_path: str,
        panel_height: int = 300,
        fps: Optional[float] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize_width: Optional[int] = None
    ):
        """
        Process video and create visualization.

        Args:
            video_path: Path to input video
            output_path: Path to output video
            panel_height: Height of prediction panel
            fps: Output FPS (None = use input FPS)
            start_frame: First frame to process
            end_frame: Last frame to process (None = all)
            resize_width: Resize video width (maintains aspect ratio)
        """
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps is None:
            fps = input_fps

        if end_frame is None:
            end_frame = total_frames

        print(f"\nInput video: {video_path}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {input_fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Processing frames: {start_frame} to {end_frame}")

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

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Process frames
        # Use a deque to maintain a buffer of frames based on model requirements
        num_required_frames = self.config.num_stacked_frames
        frame_buffer = deque(maxlen=num_required_frames)
        frames_processed = 0

        pbar = tqdm(
            total=end_frame - start_frame,
            desc="Processing video"
        )

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if needed
            if resize_width is not None:
                frame = cv2.resize(frame, (output_width, output_height))

            # Add frame to buffer
            frame_buffer.append(frame.copy())

            # Wait until we have enough frames for prediction
            if len(frame_buffer) < num_required_frames:
                # Create a "waiting" panel
                panel = np.zeros((panel_height, output_width, 3), dtype=np.uint8)
                cv2.putText(
                    panel,
                    f"Collecting frames... ({len(frame_buffer)}/{num_required_frames})",
                    (20, panel_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
            else:
                # Run inference on the buffered frames
                # Convert BGR (OpenCV) to RGB (PIL/model) for all frames in buffer
                frame_pils = []
                for buffered_frame in frame_buffer:
                    rgb = cv2.cvtColor(buffered_frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    frame_pils.append(pil)

                # Predict
                state, probs = self.predictor.predict(
                    frame_pils,
                    return_probabilities=True
                )

                # Create prediction panel
                panel = self.draw_prediction_panel(
                    output_width,
                    panel_height,
                    state,
                    probs,
                    frame_idx
                )

            # Combine frame and panel
            combined = np.vstack([frame, panel])

            # Write frame
            out.write(combined)

            frames_processed += 1
            pbar.update(1)

        pbar.close()

        # Cleanup
        cap.release()
        out.release()

        print(f"\n✓ Processed {frames_processed} frames")
        print(f"✓ Output saved to: {output_path}")

    def process_video_with_ground_truth(
        self,
        video_path: str,
        inputs_json_path: str,
        output_path: str,
        panel_height: int = 400,
        **kwargs
    ):
        """
        Process video with ground truth labels for comparison.

        Args:
            video_path: Path to input video
            inputs_json_path: Path to inputs.json with ground truth
            output_path: Path to output video
            panel_height: Height of prediction panel (needs more space for comparison)
            **kwargs: Additional arguments for process_video
        """
        import json

        # Load ground truth
        with open(inputs_json_path, 'r') as f:
            ground_truth = json.load(f)

        # Create frame_idx to state mapping
        gt_map = {entry['frame']: entry['state'] for entry in ground_truth}

        print(f"Loaded ground truth from: {inputs_json_path}")
        print(f"  Labels for {len(gt_map)} frames")

        # TODO: Implement comparison visualization
        # This would show predicted vs actual side by side
        print("\nNote: Ground truth comparison visualization is a future enhancement.")
        print("For now, processing with predictions only.\n")

        self.process_video(video_path, output_path, panel_height, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions on video"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Model version: v1 (ResNet-18, 2 frames) or v2 (MobileNetV2, 4 frames) (default: v2)"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions_visualization.mp4",
        help="Path to output video (default: predictions_visualization.mp4)"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Optional: Path to inputs.json with ground truth labels"
    )
    parser.add_argument(
        "--panel_height",
        type=int,
        default=300,
        help="Height of prediction panel in pixels (default: 300)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS (default: same as input)"
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
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--num_stacked_frames",
        type=int,
        default=None,
        help="Number of stacked frames (override for weights-only checkpoints)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Image size as HEIGHT WIDTH (override for weights-only checkpoints)"
    )
    parser.add_argument(
        "--use_grayscale",
        action="store_true",
        help="Use grayscale images (override for weights-only checkpoints)"
    )

    args = parser.parse_args()

    # Convert image_size list to tuple if provided
    image_size = tuple(args.image_size) if args.image_size else None

    print(f"\n{'='*70}")
    print("Inverse Dynamics Model - Video Prediction Visualization")
    print(f"{'='*70}\n")

    # Create visualizer
    visualizer = PredictionVisualizer(
        args.model_path,
        model_version=args.model_version,
        device=args.device,
        num_stacked_frames=args.num_stacked_frames,
        image_size=image_size,
        use_grayscale=args.use_grayscale if args.use_grayscale else None
    )

    # Process video
    if args.ground_truth:
        visualizer.process_video_with_ground_truth(
            video_path=args.video_path,
            inputs_json_path=args.ground_truth,
            output_path=args.output_path,
            panel_height=args.panel_height,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            resize_width=args.resize_width
        )
    else:
        visualizer.process_video(
            video_path=args.video_path,
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
    print()


if __name__ == "__main__":
    main()
