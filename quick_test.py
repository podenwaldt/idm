#!/usr/bin/env python3
"""
Quick test script to verify the entire pipeline works.

This script:
1. Generates a small synthetic dataset
2. Trains the model for a few epochs
3. Runs inference
4. Cleans up temporary files

Usage:
    python quick_test.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False

    print(f"\n✓ Success: {description}")
    return True


def main():
    print(f"\n{'='*70}")
    print("Inverse Dynamics Model - Quick Test")
    print(f"{'='*70}")
    print("\nThis will:")
    print("  1. Generate a small synthetic dataset (100 train, 30 val, 30 test)")
    print("  2. Train the model for 3 epochs")
    print("  3. Run inference on test frames")
    print("  4. Clean up temporary files")
    print("\nThis test takes approximately 2-5 minutes.\n")

    # Temporary directories
    test_data_dir = Path("test_fake_data")
    test_checkpoint_dir = Path("test_checkpoints")
    test_eval_dir = Path("test_evaluation")

    try:
        # Step 1: Generate synthetic data
        if not run_command(
            ["python", "generate_fake_data.py",
             "--output_dir", str(test_data_dir),
             "--num_train", "100",
             "--num_val", "30",
             "--num_test", "30"],
            "Step 1: Generating synthetic dataset"
        ):
            return 1

        # Step 2: Train model for a few epochs
        if not run_command(
            ["python", "train.py",
             "--train_dir", str(test_data_dir / "train"),
             "--val_dir", str(test_data_dir / "val"),
             "--test_dir", str(test_data_dir / "test"),
             "--batch_size", "16",
             "--num_epochs", "3",
             "--checkpoint_dir", str(test_checkpoint_dir),
             "--eval_output_dir", str(test_eval_dir),
             "--num_workers", "0"],  # Avoid multiprocessing issues
            "Step 2: Training model (3 epochs)"
        ):
            return 1

        # Step 3: Run inference
        test_frames = [
            str(test_data_dir / "test" / "frame_0000.jpg"),
            str(test_data_dir / "test" / "frame_0001.jpg")
        ]

        model_path = test_checkpoint_dir / "idm_best.pth"
        if not model_path.exists():
            # Try final model if best doesn't exist
            model_path = test_checkpoint_dir / "idm_final.pth"

        if not model_path.exists():
            print("\n❌ Model checkpoint not found!")
            return 1

        if not run_command(
            ["python", "example_inference.py",
             "--model_path", str(model_path),
             "--frames"] + test_frames + ["--benchmark"],
            "Step 3: Running inference"
        ):
            return 1

        print(f"\n{'='*70}")
        print("✓ All Tests Passed!")
        print(f"{'='*70}")
        print("\nThe pipeline is working correctly!")
        print("\nGenerated files:")
        print(f"  Data: {test_data_dir}/")
        print(f"  Checkpoints: {test_checkpoint_dir}/")
        print(f"  Evaluation: {test_eval_dir}/")

        # Ask if user wants to clean up
        print(f"\n{'='*70}")
        response = input("Delete test files? (y/n): ").strip().lower()

        if response == 'y':
            print("\nCleaning up test files...")
            if test_data_dir.exists():
                shutil.rmtree(test_data_dir)
                print(f"  ✓ Deleted {test_data_dir}")
            if test_checkpoint_dir.exists():
                shutil.rmtree(test_checkpoint_dir)
                print(f"  ✓ Deleted {test_checkpoint_dir}")
            if test_eval_dir.exists():
                shutil.rmtree(test_eval_dir)
                print(f"  ✓ Deleted {test_eval_dir}")
            print("\n✓ Cleanup complete!")
        else:
            print("\nTest files kept. You can delete them manually:")
            print(f"  rm -rf {test_data_dir} {test_checkpoint_dir} {test_eval_dir}")

        print(f"\n{'='*70}")
        print("Next Steps:")
        print(f"{'='*70}")
        print("\n1. Generate your own dataset:")
        print("   - Option A: From video")
        print("     python -m inverse_dynamics_model.data_preparation convert video.mp4 data/raw")
        print("   - Option B: Use existing frames and create inputs.json")
        print("\n2. Train on your data:")
        print("   python train.py --train_dir data/train --val_dir data/val --test_dir data/test")
        print("\n3. Run inference:")
        print("   python example_inference.py --model_path checkpoints/idm_best.pth --frames frame1.jpg frame2.jpg")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
