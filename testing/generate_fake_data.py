#!/usr/bin/env python3
"""
Generate synthetic/fake data for testing the Inverse Dynamics Model.

This creates random images with corresponding state labels for testing
the training and inference pipeline before real data is available.

Usage:
    python generate_fake_data.py --output_dir fake_data --num_train 200 --num_val 50 --num_test 50
"""

import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_colored_frame(
    width: int = 224,
    height: int = 224,
    state: int = 0
) -> Image.Image:
    """
    Generate a synthetic frame with colors corresponding to the state.

    This makes it easier to visually verify the data and potentially
    helps the model learn patterns (though it's still synthetic).

    Args:
        width: Image width
        height: Image height
        state: Control state (0-4)

    Returns:
        PIL Image
    """
    # Define colors for each state (makes it easier to visually verify)
    state_colors = {
        0: (50, 50, 50),      # STOPPED - dark gray
        1: (0, 255, 0),       # FORWARD - green
        2: (255, 0, 0),       # BACKWARD - red
        3: (0, 0, 255),       # ROTATE_LEFT - blue
        4: (255, 255, 0),     # ROTATE_RIGHT - yellow
    }

    base_color = state_colors.get(state, (128, 128, 128))

    # Create image with base color plus some noise for variation
    img_array = np.ones((height, width, 3), dtype=np.uint8) * base_color

    # Add random noise to make each frame unique
    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add some random shapes to simulate objects/features
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Draw a few random shapes
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, width - 20)
        y1 = random.randint(0, height - 20)
        x2 = x1 + random.randint(10, 40)
        y2 = y1 + random.randint(10, 40)

        shape_type = random.choice(['rectangle', 'ellipse'])
        color = tuple(random.randint(0, 255) for _ in range(3))

        if shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)

    # Add text label to indicate state (optional, for debugging)
    state_names = {0: "STOP", 1: "FWD", 2: "BACK", 3: "LEFT", 4: "RIGHT"}
    try:
        draw.text((10, 10), state_names[state], fill=(255, 255, 255))
    except:
        pass  # If font fails, skip text

    return img


def generate_random_frame(
    width: int = 224,
    height: int = 224
) -> Image.Image:
    """
    Generate a completely random frame (noise).

    Args:
        width: Image width
        height: Image height

    Returns:
        PIL Image
    """
    # Random RGB noise
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def generate_synthetic_dataset(
    output_dir: str,
    num_frames: int,
    split_name: str,
    fps: float = 30.0,
    use_colored: bool = True,
    state_distribution: dict = None
):
    """
    Generate a synthetic dataset with frames and inputs.json.

    Args:
        output_dir: Directory to save frames and inputs.json
        num_frames: Number of frames to generate
        split_name: Name of the split (train/val/test)
        fps: Frames per second (for timestamp calculation)
        use_colored: Use colored frames (based on state) vs random noise
        state_distribution: Dictionary with target distribution of states
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default state distribution if not provided
    if state_distribution is None:
        state_distribution = {
            0: 0.15,  # STOPPED - 15%
            1: 0.40,  # FORWARD - 40%
            2: 0.10,  # BACKWARD - 10%
            3: 0.175, # ROTATE_LEFT - 17.5%
            4: 0.175, # ROTATE_RIGHT - 17.5%
        }

    # Generate state sequence based on distribution
    states = []
    for state, prob in state_distribution.items():
        count = int(num_frames * prob)
        states.extend([state] * count)

    # Fill remaining frames to reach exact num_frames
    while len(states) < num_frames:
        states.append(random.choice(list(state_distribution.keys())))

    # Shuffle to randomize order
    random.shuffle(states)
    states = states[:num_frames]

    # Add some temporal coherence (states tend to persist for a few frames)
    coherent_states = []
    i = 0
    while i < len(states):
        current_state = states[i]
        # Keep same state for 3-10 frames
        duration = random.randint(3, 10)
        coherent_states.extend([current_state] * duration)
        i += duration

    # Trim to exact length
    states = coherent_states[:num_frames]

    # Generate frames and metadata
    inputs_data = []

    print(f"Generating {num_frames} frames for {split_name}...")

    for frame_idx in range(num_frames):
        state = states[frame_idx]

        # Generate image
        if use_colored:
            img = generate_colored_frame(state=state)
        else:
            img = generate_random_frame()

        # Save image
        frame_path = output_path / f"frame_{frame_idx:04d}.jpg"
        img.save(frame_path, quality=95)

        # Add to metadata
        time = frame_idx / fps
        inputs_data.append({
            "frame": frame_idx,
            "time": round(time, 3),
            "state": int(state)
        })

        if (frame_idx + 1) % 50 == 0:
            print(f"  Generated {frame_idx + 1}/{num_frames} frames...")

    # Save inputs.json
    inputs_path = output_path / "inputs.json"
    with open(inputs_path, "w") as f:
        json.dump(inputs_data, f, indent=2)

    # Print statistics
    state_counts = {}
    for entry in inputs_data:
        state = entry["state"]
        state_counts[state] = state_counts.get(state, 0) + 1

    state_names = {0: "STOPPED", 1: "FORWARD", 2: "BACKWARD", 3: "ROTATE_LEFT", 4: "ROTATE_RIGHT"}

    print(f"\n{split_name} dataset created:")
    print(f"  Location: {output_path}")
    print(f"  Frames: {num_frames}")
    print(f"  State distribution:")
    for state_id in sorted(state_counts.keys()):
        count = state_counts[state_id]
        percentage = (count / num_frames) * 100
        print(f"    {state_names[state_id]:15s}: {count:4d} ({percentage:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic/fake data for testing Inverse Dynamics Model"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="fake_data",
        help="Base directory for output (default: fake_data)"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=200,
        help="Number of training frames (default: 200)"
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=50,
        help="Number of validation frames (default: 50)"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=50,
        help="Number of test frames (default: 50)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for timestamp calculation (default: 30)"
    )
    parser.add_argument(
        "--random_noise",
        action="store_true",
        help="Use random noise instead of colored frames"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*70}")
    print("Generating Synthetic Dataset")
    print(f"{'='*70}\n")

    base_dir = Path(args.output_dir)

    # Generate train set
    generate_synthetic_dataset(
        output_dir=base_dir / "train",
        num_frames=args.num_train,
        split_name="train",
        fps=args.fps,
        use_colored=not args.random_noise
    )

    print()

    # Generate validation set
    generate_synthetic_dataset(
        output_dir=base_dir / "val",
        num_frames=args.num_val,
        split_name="val",
        fps=args.fps,
        use_colored=not args.random_noise
    )

    print()

    # Generate test set
    generate_synthetic_dataset(
        output_dir=base_dir / "test",
        num_frames=args.num_test,
        split_name="test",
        fps=args.fps,
        use_colored=not args.random_noise
    )

    print(f"\n{'='*70}")
    print("Synthetic Dataset Generation Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {base_dir}")
    print(f"\nDirectory structure:")
    print(f"  {base_dir}/")
    print(f"    train/")
    print(f"      frame_0000.jpg ... frame_{args.num_train-1:04d}.jpg")
    print(f"      inputs.json")
    print(f"    val/")
    print(f"      frame_0000.jpg ... frame_{args.num_val-1:04d}.jpg")
    print(f"      inputs.json")
    print(f"    test/")
    print(f"      frame_0000.jpg ... frame_{args.num_test-1:04d}.jpg")
    print(f"      inputs.json")

    print(f"\nYou can now train the model with:")
    print(f"  python train.py \\")
    print(f"    --train_dir {base_dir}/train \\")
    print(f"    --val_dir {base_dir}/val \\")
    print(f"    --test_dir {base_dir}/test \\")
    print(f"    --batch_size 16 \\")
    print(f"    --num_epochs 10")

    print(f"\nOr test inference with:")
    print(f"  python example_inference.py \\")
    print(f"    --model_path checkpoints/idm_best.pth \\")
    print(f"    --frames {base_dir}/test/frame_0000.jpg {base_dir}/test/frame_0001.jpg")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
