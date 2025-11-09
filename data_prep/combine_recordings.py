#!/usr/bin/env python3
"""
Combine multiple RC car recording runs into a single training dataset.

Each recording run should have:
- frames: frame_0000.jpg, frame_0001.jpg, ...
- inputs.json: frame labels with timestamps and states

Usage:
    python combine_recordings.py recording1/ recording2/ recording3/ --output data/train/
    python combine_recordings.py recording_* --output data/train/
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import sys


def load_inputs_json(recording_dir: Path) -> List[Dict[str, Any]]:
    """Load inputs.json from a recording directory."""
    inputs_path = recording_dir / "inputs.json"
    if not inputs_path.exists():
        raise FileNotFoundError(f"No inputs.json found in {recording_dir}")

    with open(inputs_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"inputs.json in {recording_dir} should be a list")

    return data


def find_frames(recording_dir: Path) -> List[Path]:
    """Find all frame files in a recording directory."""
    frames = sorted(recording_dir.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frame files found in {recording_dir}")
    return frames


def validate_recording(recording_dir: Path) -> tuple[List[Dict], List[Path]]:
    """Validate a recording directory and return inputs and frames."""
    print(f"Validating {recording_dir}...")

    inputs = load_inputs_json(recording_dir)
    frames = find_frames(recording_dir)

    print(f"  Found {len(frames)} frames and {len(inputs)} entries in inputs.json")

    # Validate that frame indices in inputs.json match available frames
    expected_frames = {entry["frame"] for entry in inputs}
    actual_frames = {int(f.stem.split("_")[1]) for f in frames}

    if expected_frames != actual_frames:
        missing = expected_frames - actual_frames
        extra = actual_frames - expected_frames
        if missing:
            print(f"  WARNING: Frames in inputs.json but missing on disk: {sorted(missing)[:10]}")
        if extra:
            print(f"  WARNING: Frames on disk but not in inputs.json: {sorted(extra)[:10]}")

    return inputs, frames


def combine_recordings(
    recording_dirs: List[Path],
    output_dir: Path,
    continuous_time: bool = False,
    time_gap: float = 0.0
):
    """
    Combine multiple recordings into a single dataset.

    Args:
        recording_dirs: List of recording directories to combine
        output_dir: Output directory for combined dataset
        continuous_time: If True, adjust timestamps to be continuous across recordings
        time_gap: Gap in seconds to add between recordings (only if continuous_time=True)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_inputs = []
    frame_counter = 0
    time_offset = 0.0

    for idx, recording_dir in enumerate(recording_dirs):
        print(f"\n[{idx + 1}/{len(recording_dirs)}] Processing {recording_dir.name}...")

        # Load and validate recording
        inputs, frames = validate_recording(recording_dir)

        # Sort inputs by frame number
        inputs = sorted(inputs, key=lambda x: x["frame"])

        # Build frame index mapping (old frame number -> Path)
        frame_map = {int(f.stem.split("_")[1]): f for f in frames}

        # Process each input entry
        recording_start_time = inputs[0]["time"] if inputs else 0.0

        for entry in inputs:
            old_frame_num = entry["frame"]

            # Skip if frame file doesn't exist
            if old_frame_num not in frame_map:
                print(f"  WARNING: Skipping frame {old_frame_num} (file not found)")
                continue

            # Copy frame with new number
            src_frame = frame_map[old_frame_num]
            dst_frame = output_dir / f"frame_{frame_counter:04d}.jpg"
            shutil.copy2(src_frame, dst_frame)

            # Adjust timestamp
            if continuous_time:
                # Make timestamps continuous across recordings
                adjusted_time = time_offset + (entry["time"] - recording_start_time)
            else:
                # Keep original timestamps (will have gaps/overlaps)
                adjusted_time = entry["time"]

            # Create new entry with adjusted frame number and time
            new_entry = {
                "frame": frame_counter,
                "time": round(adjusted_time, 3),
                "state": entry["state"]
            }
            combined_inputs.append(new_entry)

            frame_counter += 1

        # Update time offset for next recording
        if continuous_time and inputs:
            last_time = inputs[-1]["time"]
            time_offset += (last_time - recording_start_time) + time_gap

        print(f"  Copied {len(inputs)} frames (total so far: {frame_counter})")

    # Save combined inputs.json
    output_json = output_dir / "inputs.json"
    with open(output_json, 'w') as f:
        json.dump(combined_inputs, f, indent=2)

    print(f"\n✓ Successfully combined {len(recording_dirs)} recordings!")
    print(f"  Total frames: {frame_counter}")
    print(f"  Output directory: {output_dir}")
    print(f"  inputs.json: {output_json}")

    # Print state distribution
    state_counts = {}
    for entry in combined_inputs:
        state = entry["state"]
        state_counts[state] = state_counts.get(state, 0) + 1

    state_names = {
        0: "STOPPED",
        1: "FORWARD",
        2: "BACKWARD",
        3: "ROTATE_LEFT",
        4: "ROTATE_RIGHT"
    }

    print(f"\nState distribution:")
    for state, count in sorted(state_counts.items()):
        name = state_names.get(state, f"UNKNOWN_{state}")
        percentage = (count / frame_counter * 100) if frame_counter > 0 else 0
        print(f"  {state} ({name:12s}): {count:4d} frames ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple RC car recording runs into a single dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine three recordings into a new directory
  python combine_recordings.py recording1/ recording2/ recording3/ --output combined/

  # Combine all recordings matching a pattern
  python combine_recordings.py recording_* --output data/train/

  # Combine with continuous timestamps and 1-second gap between recordings
  python combine_recordings.py rec1/ rec2/ --output combined/ --continuous-time --time-gap 1.0
        """
    )

    parser.add_argument(
        "recordings",
        nargs='+',
        type=Path,
        help="Recording directories to combine (each should contain frames and inputs.json)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for combined dataset"
    )

    parser.add_argument(
        "--continuous-time",
        action="store_true",
        help="Adjust timestamps to be continuous across recordings (default: keep original times)"
    )

    parser.add_argument(
        "--time-gap",
        type=float,
        default=0.0,
        help="Time gap in seconds to add between recordings (requires --continuous-time)"
    )

    args = parser.parse_args()

    # Validate recording directories
    valid_dirs = []
    for rec_dir in args.recordings:
        if not rec_dir.is_dir():
            print(f"WARNING: {rec_dir} is not a directory, skipping", file=sys.stderr)
            continue
        valid_dirs.append(rec_dir)

    if not valid_dirs:
        print("ERROR: No valid recording directories provided", file=sys.stderr)
        sys.exit(1)

    if len(valid_dirs) < 2:
        print(f"WARNING: Only {len(valid_dirs)} recording provided. You can still use this to copy/renumber a single recording.")

    print(f"Combining {len(valid_dirs)} recording(s)...")
    print(f"Output: {args.output}")
    print(f"Continuous time: {args.continuous_time}")
    if args.continuous_time and args.time_gap > 0:
        print(f"Time gap between recordings: {args.time_gap}s")
    print()

    try:
        combine_recordings(
            recording_dirs=valid_dirs,
            output_dir=args.output,
            continuous_time=args.continuous_time,
            time_gap=args.time_gap
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
