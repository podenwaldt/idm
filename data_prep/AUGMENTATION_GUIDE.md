# Image Augmentation Guide

This guide explains how to use `augment_recordings.py` to create augmented datasets for training your RC car inverse dynamics model.

## Overview

The augmentation script allows you to create additional training data by applying various image transformations to existing recordings. This can help improve model robustness and performance, especially for conditions that are underrepresented in your original dataset.

## Installation

The script requires the following dependencies (already in `requirements.txt`):

```bash
pip install opencv-python numpy tqdm
```

## Basic Usage

### Single Recording with Brightness Adjustment

Create a brighter version of a recording:

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output recordings/train/recording_114838_bright \
    --brightness 1.3
```

Create a darker/low-light version:

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output recordings/train/recording_114838_dark \
    --brightness 0.6
```

### Multiple Transformations

Apply several transformations at once:

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output recordings/train/recording_114838_aug \
    --brightness 1.2 \
    --contrast 1.1 \
    --saturation 0.9 \
    --rotation 5
```

## Available Transformations

| Parameter | Description | Default Range | Example Values |
|-----------|-------------|---------------|----------------|
| `--brightness` | Adjust image brightness | 0.5 - 1.5 | 0.6 (dark), 1.3 (bright) |
| `--contrast` | Adjust image contrast | 0.5 - 1.5 | 0.8 (low), 1.2 (high) |
| `--saturation` | Adjust color saturation | 0.0 - 2.0 | 0.0 (grayscale), 1.5 (vivid) |
| `--hue` | Shift color hue | -180 to 180° | -30, 45 |
| `--gamma` | Apply gamma correction | 0.5 - 2.0 | 0.8 (brighter), 1.2 (darker) |
| `--rotation` | Rotate image | ±45° | -10, 15 |
| `--blur` | Apply Gaussian blur | kernel size | 3, 5, 7 (odd numbers) |
| `--motion-blur` | Simulate motion blur | kernel size | 5, 9, 15 |
| `--motion-blur-angle` | Direction of motion blur | 0-360° | 0 (horizontal), 90 (vertical) |
| `--noise-std` | Add Gaussian noise | std deviation | 5, 10, 20 |

## Advanced Usage

### Batch Augmentation with Random Variations

Create multiple augmented variants with random brightness:

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output-dir recordings/train_augmented/ \
    --brightness-range 0.7 1.4 \
    --num-variants 3 \
    --seed 42
```

This creates 3 new recordings:
- `recording_114838_var1` (random brightness between 0.7-1.4)
- `recording_114838_var2` (different random brightness)
- `recording_114838_var3` (another random brightness)

### Process Multiple Recordings

Augment several recordings at once:

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    recordings/train/recording_171046/ \
    recordings/train/recording_day1/ \
    --output-dir recordings/train_bright/ \
    --brightness 1.3
```

Using wildcards (bash expansion):

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_day* \
    --output-dir recordings/train_dark/ \
    --brightness 0.6 \
    --contrast 0.8
```

### Create Low-Light Training Data

Simulate nighttime or low-light conditions:

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_* \
    --output-dir recordings/train_lowlight/ \
    --brightness 0.5 \
    --contrast 0.7 \
    --saturation 0.8
```

### Augment Entire Dataset

Apply augmentations to the combined training dataset:

```bash
python data_prep/augment_recordings.py \
    data/train/ \
    --output data/train_aug/ \
    --brightness 1.2 \
    --contrast 1.1
```

## Practical Workflow Examples

### Scenario 1: Expand Training Data with Lighting Variations

```bash
# Original recordings: 17 in recordings/train/

# Create bright variants
python data_prep/augment_recordings.py \
    recordings/train/recording_* \
    --output-dir recordings/train_bright/ \
    --brightness-range 1.2 1.5 \
    --num-variants 1

# Create dark variants
python data_prep/augment_recordings.py \
    recordings/train/recording_* \
    --output-dir recordings/train_dark/ \
    --brightness-range 0.5 0.8 \
    --num-variants 1

# Now you have 51 recordings (17 original + 17 bright + 17 dark)
```

### Scenario 2: Test Model Robustness to Weather/Lighting

```bash
# Create test sets with different conditions
python data_prep/augment_recordings.py \
    recordings/test/ \
    --output recordings/test_foggy/ \
    --brightness 0.8 \
    --contrast 0.7 \
    --blur 5

python data_prep/augment_recordings.py \
    recordings/test/ \
    --output recordings/test_motion/ \
    --motion-blur 9 \
    --motion-blur-angle 0
```

### Scenario 3: Create Diverse Training Variants

```bash
# Multiple variants with different augmentation combinations
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output-dir augmented_variants/ \
    --brightness-range 0.7 1.4 \
    --contrast 1.1 \
    --saturation 0.95 \
    --num-variants 5 \
    --seed 42
```

## Integration with Training Pipeline

After creating augmented recordings, combine them with your original data:

```bash
# Combine original + augmented recordings
python data_prep/combine_recordings.py \
    recordings/train/recording_* \
    recordings/train_bright/recording_* \
    recordings/train_dark/recording_* \
    --output data/train_augmented/ \
    --continuous-time

# Train with augmented dataset
python train.py --data-dir data/train_augmented/
```

## Best Practices

1. **Start Conservative**: Begin with subtle augmentations (brightness 0.8-1.2) and gradually increase
2. **Preserve Labels**: The script automatically copies `inputs.json` and `metadata.json` - labels remain unchanged
3. **Check Disk Space**: Augmented recordings require additional storage (each recording ~5-10 MB)
4. **Use Seeds**: For reproducible augmentations, always specify `--seed` parameter
5. **Test First**: Augment a single recording first to verify parameters before batch processing
6. **Monitor Training**: Track validation performance to ensure augmentations improve (not hurt) the model

## Quality Settings

Control JPEG compression quality:

```bash
python data_prep/augment_recordings.py \
    recordings/train/recording_114838/ \
    --output recordings/train/recording_114838_aug \
    --brightness 1.2 \
    --jpeg-quality 95  # Default is 95 (higher = better quality, larger files)
```

## Output Structure

The script preserves the original recording structure:

```
output_recording/
├── frame_0000.jpg       # Augmented frames
├── frame_0001.jpg
├── ...
├── inputs.json          # Copied from original (labels unchanged)
└── metadata.json        # Updated with augmentation info
```

The `metadata.json` includes augmentation parameters:

```json
{
  "recording_start_millis": 575278,
  "frame_rate": 30,
  "resolution": "640x480",
  "total_frames": 250,
  "augmented": true,
  "augmentation_params": {
    "brightness": 1.3,
    "contrast": 1.1
  },
  "source_recording": "recordings/train/recording_114838"
}
```

## Troubleshooting

### "No frame images found" Error

Ensure the input directory contains `frame_*.jpg` or `frame_*.png` files.

### Memory Issues with Large Batches

Process recordings in smaller batches or reduce the number of variants.

### Unexpected Results

- Check transformation parameters are in reasonable ranges
- View sample augmented frames to verify they look correct
- Use `--no-verbose` to disable progress bars if they interfere with output

## Performance

- Processing speed: ~80-200 frames/second (depends on transformations)
- A 250-frame recording takes ~1-3 seconds to augment
- Batch processing 17 recordings with 3 variants each: ~1-2 minutes

## Examples Gallery

Here are recommended parameter combinations for common scenarios:

| Scenario | Parameters |
|----------|------------|
| Bright daylight | `--brightness 1.4 --contrast 1.1` |
| Overcast/cloudy | `--brightness 0.8 --saturation 0.85` |
| Evening/dusk | `--brightness 0.7 --saturation 0.9 --hue -10` |
| Indoor/artificial light | `--brightness 0.9 --gamma 1.1` |
| Motion blur | `--motion-blur 7 --motion-blur-angle 0` |
| Camera shake | `--blur 3 --rotation 3` |
| Low contrast | `--contrast 0.75` |
| Desaturated | `--saturation 0.5` |
| Slight noise | `--noise-std 5` |

## Questions?

For issues or questions about the augmentation script:
1. Check this guide for examples
2. Run `python data_prep/augment_recordings.py --help` for full parameter list
3. Review the script documentation in `augment_recordings.py`
