#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not found. Run: pip install torch torchvision")
        return False

    try:
        import torchvision
        print(f"  ✓ TorchVision version: {torchvision.__version__}")
    except ImportError:
        print("  ✗ TorchVision not found. Run: pip install torchvision")
        return False

    try:
        from inverse_dynamics_model import (
            IDMConfig,
            InverseDynamicsModel,
            RCCarInverseDynamicsDataset,
            InverseDynamicsTrainer,
            InverseDynamicsPredictor
        )
        print("  ✓ All inverse_dynamics_model modules imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import inverse_dynamics_model: {e}")
        return False

    return True


def test_model_creation():
    """Test model creation and forward pass."""
    print("\nTesting model creation...")

    try:
        import torch
        from inverse_dynamics_model import IDMConfig, InverseDynamicsModel

        config = IDMConfig()
        model = InverseDynamicsModel(config, pretrained=False)

        print(f"  ✓ Model created successfully")
        print(f"    Parameters: {model.count_parameters():,}")
        print(f"    Model size: {model.get_model_size_mb():.2f} MB")

        # Test forward pass
        x = torch.randn(2, config.input_channels, *config.image_size)
        y = model(x)

        expected_shape = (2, config.num_classes)
        if y.shape == expected_shape:
            print(f"  ✓ Forward pass successful: {y.shape}")
        else:
            print(f"  ✗ Forward pass output shape incorrect: {y.shape}, expected {expected_shape}")
            return False

    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False

    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    try:
        from inverse_dynamics_model import IDMConfig

        config = IDMConfig()

        print(f"  ✓ Config created successfully")
        print(f"    Input channels: {config.input_channels}")
        print(f"    Image size: {config.image_size}")
        print(f"    Num classes: {config.num_classes}")
        print(f"    Batch size: {config.batch_size}")
        print(f"    Learning rate: {config.learning_rate}")

    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  ⚠ CUDA not available (CPU only)")

    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print(f"\n{'='*70}")
    print("Inverse Dynamics Model - Installation Test")
    print(f"{'='*70}\n")

    tests = [
        test_imports,
        test_config,
        test_model_creation,
        test_cuda,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            results.append(False)

    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("  1. Prepare your data (see README.md)")
        print("  2. Train the model: python train.py --train_dir data/train --val_dir data/val")
        print("  3. Run inference: python example_inference.py --model_path checkpoints/idm_best.pth --frames frame1.jpg frame2.jpg")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
