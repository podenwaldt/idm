#!/usr/bin/env python3
"""
Example inference script for Inverse Dynamics Model.

Usage:
    python example_inference.py --model_path checkpoints/idm_best.pth --frames frame1.jpg frame2.jpg
"""

import argparse
from pathlib import Path

from inverse_dynamics_model.inference import InverseDynamicsPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained Inverse Dynamics Model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--frames",
        type=str,
        nargs="+",
        required=True,
        help="Paths to consecutive frame images"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Show top K predictions (default: None, show only best)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference time benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    predictor = InverseDynamicsPredictor(args.model_path, device=args.device)

    print(f"\n{'='*70}")
    print("Running Inference")
    print(f"{'='*70}")

    # Check frames
    for frame_path in args.frames:
        if not Path(frame_path).exists():
            print(f"Error: Frame not found: {frame_path}")
            return

    print(f"Input frames: {len(args.frames)}")
    for i, frame_path in enumerate(args.frames):
        print(f"  Frame {i}: {frame_path}")

    # Run inference with probabilities
    state_name, probabilities = predictor.predict_with_names(
        args.frames,
        return_probabilities=True
    )

    print(f"\n{'='*70}")
    print("Prediction Results")
    print(f"{'='*70}")
    print(f"Predicted State: {state_name}")
    print(f"\nProbability Distribution:")

    # Sort by probability (descending)
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for state_name, prob in sorted_probs:
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {state_name:15s} [{bar}] {prob:.4f}")

    # Top-K predictions if requested
    if args.top_k is not None:
        print(f"\n{'='*70}")
        print(f"Top {args.top_k} Predictions")
        print(f"{'='*70}")

        top_k_results = predictor.predict(args.frames, top_k=args.top_k)

        for rank, (state_id, prob) in enumerate(top_k_results, 1):
            state_name = predictor.get_state_name(state_id)
            print(f"  {rank}. {state_name:15s} - {prob:.4f}")

    # Benchmark if requested
    if args.benchmark:
        print(f"\n{'='*70}")
        print("Inference Time Benchmark")
        print(f"{'='*70}")

        benchmark_results = predictor.benchmark_inference_time(num_runs=100)

        print(f"  Mean:   {benchmark_results['mean_ms']:.2f} ms")
        print(f"  Std:    {benchmark_results['std_ms']:.2f} ms")
        print(f"  Min:    {benchmark_results['min_ms']:.2f} ms")
        print(f"  Max:    {benchmark_results['max_ms']:.2f} ms")
        print(f"  Median: {benchmark_results['median_ms']:.2f} ms")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
