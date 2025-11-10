# Inverse Dynamics Model (IDM) for RC Car Control

A deep learning system that predicts RC car control states from video frames.

## What It Does

Given consecutive video frames from an RC car's camera, the model predicts which control is active:
- **STOPPED** (0)
- **FORWARD** (1)
- **BACKWARD** (2)
- **ROTATE_LEFT** (3)
- **ROTATE_RIGHT** (4)

## Model Versions

### V1: ResNet-18 (2 frames)
- Architecture: ResNet-18
- Input: 2 consecutive frames
- Parameters: ~11M
- Model size: ~44 MB

### V2: MobileNetV2 (4 frames)
- Architecture: MobileNetV2
- Input: 4 consecutive frames
- Parameters: ~3-4M
- Model size: ~14-16 MB
- **Faster inference** and **smaller size** than V1

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

For GPU support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Prepare Training Data

The repository includes recordings organized in `recordings/train/`, `recordings/val/`, and `recordings/test/`. Each recording contains:
- Frame images: `frame_0000.jpg`, `frame_0001.jpg`, ...
- Labels: `inputs.json` (frame number, timestamp, control state)
- Metadata: `metadata.json`

**Combine recordings into datasets:**

```bash
# Combine all training recordings
python data_prep/combine_recordings.py recordings/train/recording_* --output data/train/

# Combine validation recordings
python data_prep/combine_recordings.py recordings/val/recording_* --output data/val/

# Combine test recordings
python data_prep/combine_recordings.py recordings/test/recording_* --output data/test/
```

**Optional: Augment data for better training:**

```bash
# Create darker/brighter variants
python data_prep/augment_recordings.py data/train/ \
    --output data/train_dark/ \
    --brightness 0.5

# Apply multiple transformations
python data_prep/augment_recordings.py data/train/ \
    --output data/train_aug/ \
    --brightness 1.2 --contrast 1.1 --rotation 5
```

### Train a Model

**Train V1 (ResNet-18, 2 frames):**

```bash
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --test_dir data/test \
    --batch_size 32 \
    --num_epochs 50
```

**Train V2 (MobileNetV2, 4 frames):**

```bash
python train.py \
    --model_version v2 \
    --train_dir data/train \
    --val_dir data/val \
    --test_dir data/test \
    --batch_size 32 \
    --num_epochs 50
```

The best model will be saved to `checkpoints/idm_best.pth`.

### Evaluate a Model

```bash
# Evaluate V1
python evaluate.py \
    --model_version v1 \
    --checkpoint checkpoints/idm_best.pth \
    --test_dir data/test

# Evaluate V2
python evaluate.py \
    --model_version v2 \
    --checkpoint checkpoints_v2/idm_best.pth \
    --test_dir data/test
```

Results are saved to `evaluation_results/`:
- Confusion matrix
- Per-class performance charts
- Detailed metrics

### Run Inference

```bash
python testing/example_inference.py \
    --model_path checkpoints/idm_best.pth \
    --frames frame_0010.jpg frame_0011.jpg
```

### Visualize Predictions

```bash
# Create visualization video
python visualizations/visualize_predictions.py \
    --model_path checkpoints/idm_best.pth \
    --video_path input_video.mp4 \
    --output_path predictions.mp4
```

## Project Structure

```
idm/
├── inverse_dynamics_model/       # V1 model (ResNet-18, 2 frames)
├── inverse_dynamics_model_v2/    # V2 model (MobileNetV2, 4 frames)
├── recordings/                   # Training data recordings
│   ├── train/                    # Training recordings
│   ├── val/                      # Validation recordings
│   └── test/                     # Test recordings
├── data/                         # Combined datasets (after running combine_recordings.py)
│   ├── train/
│   ├── val/
│   └── test/
├── data_prep/                    # Data preparation tools
│   ├── combine_recordings.py     # Combine multiple recordings
│   └── augment_recordings.py     # Augment with transformations
├── visualizations/               # Visualization tools
│   ├── visualize_predictions.py  # Visualize model predictions on video
│   ├── visualize_labels.py       # Visualize training labels
│   └── create_test_video.py      # Create test videos
├── testing/                      # Testing utilities
│   ├── example_inference.py      # Run inference example
│   ├── quick_test.py            # Quick pipeline test
│   └── generate_fake_data.py    # Generate synthetic test data
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
└── requirements.txt              # Python dependencies
```

## Data Preparation Tools

### combine_recordings.py

Combines multiple recording sessions into a single dataset:

```bash
# Combine with continuous timestamps
python data_prep/combine_recordings.py \
    recordings/train/recording_* \
    --output data/train/ \
    --continuous-time
```

### augment_recordings.py

Creates augmented versions of recordings:

```bash
# Create multiple random variants
python data_prep/augment_recordings.py \
    recordings/train/recording_* \
    --output-dir recordings/train_augmented/ \
    --brightness-range 0.7 1.3 \
    --num-variants 3
```

Available augmentations:
- Brightness, contrast, saturation adjustment
- Hue shift and gamma correction
- Rotation and blur
- Gaussian noise and motion blur

## Python API

### Training

```python
from inverse_dynamics_model_v2 import IDMConfig
from inverse_dynamics_model_v2.trainer import train_model

config = IDMConfig(
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001
)

model, trainer = train_model(
    train_dir="data/train",
    val_dir="data/val",
    config=config
)
```

### Inference

```python
from inverse_dynamics_model_v2.inference import InverseDynamicsPredictor

predictor = InverseDynamicsPredictor("checkpoints/idm_best.pth")

# Predict from frame paths
state_name, probabilities = predictor.predict_with_names(
    ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"],
    return_probabilities=True
)

print(f"Predicted: {state_name}")
print(f"Confidence: {probabilities}")
```

## Configuration

Key parameters in `IDMConfig`:

| Parameter | V1 Default | V2 Default | Description |
|-----------|------------|------------|-------------|
| `num_stacked_frames` | 2 | 4 | Consecutive frames |
| `batch_size` | 32 | 32 | Training batch size |
| `num_epochs` | 50 | 50 | Training epochs |
| `learning_rate` | 0.001 | 0.001 | Learning rate |
| `dropout_rate` | 0.3 | 0.3 | Dropout rate |
| `use_augmentation` | True | True | Data augmentation |

## Performance Targets

- **Overall accuracy**: >85%
- **Per-state accuracy**: >80%
- **Inference time**: <50ms per prediction (GPU)
- **Training time**: ~2 hours for 10k frames

## Testing Without Real Data

Generate synthetic test data:

```bash
# Quick test of entire pipeline
python testing/quick_test.py

# Generate custom synthetic data
python testing/generate_fake_data.py \
    --num_train 1000 \
    --num_val 200 \
    --num_test 200
```

## Troubleshooting

### Out of Memory
Reduce batch size: `--batch_size 16`

### Poor Accuracy
- Check dataset balance (should have good distribution across all 5 states)
- Increase dataset size
- Try data augmentation
- Use V2 model with 4 frames for better temporal context

### Slow Training
- Ensure GPU is being used
- Increase batch size if memory allows
- Use V2 model (faster than V1)

## License

MIT License
