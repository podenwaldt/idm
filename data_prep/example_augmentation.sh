#!/bin/bash
# Example script showing common augmentation workflows

echo "=========================================="
echo "Image Augmentation Examples"
echo "=========================================="
echo ""

# Example 1: Create a brighter version of a single recording
echo "Example 1: Creating bright version..."
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output /tmp/example_bright \
    --brightness 1.3

echo ""
echo "✓ Bright version created at /tmp/example_bright"
echo ""

# Example 2: Create a darker/low-light version
echo "Example 2: Creating dark/low-light version..."
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output /tmp/example_dark \
    --brightness 0.6 \
    --contrast 0.8

echo ""
echo "✓ Dark version created at /tmp/example_dark"
echo ""

# Example 3: Create multiple random variants
echo "Example 3: Creating 3 random variants..."
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output-dir /tmp/example_variants \
    --brightness-range 0.7 1.4 \
    --num-variants 3 \
    --seed 42

echo ""
echo "✓ 3 variants created at /tmp/example_variants/"
echo ""

# Example 4: Batch process multiple recordings
echo "Example 4: Batch processing (skipped - uncomment to run)"
# Uncomment below to process all training recordings
# python data_prep/augment_recordings.py \
#     recordings/train/recording_* \
#     --output-dir /tmp/train_bright \
#     --brightness 1.3

echo ""
echo "=========================================="
echo "Examples complete!"
echo "=========================================="
echo ""
echo "Check the output directories:"
echo "  - /tmp/example_bright/"
echo "  - /tmp/example_dark/"
echo "  - /tmp/example_variants/"
echo ""
echo "For more examples, see data_prep/AUGMENTATION_GUIDE.md"
