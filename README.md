# Inverse Dynamics Model (IDM) for RC Car Control Prediction

A deep learning system that predicts control inputs from sequences of video frames captured by an RC car camera.

## Overview

The Inverse Dynamics Model uses a modified ResNet-18 architecture to predict which control state (stopped, forward, backward, rotate left, rotate right) is active based on consecutive video frames.

### Key Features

- **Modified ResNet-18 Architecture**: Pre-trained on ImageNet with custom input/output layers
- **Temporal Information**: Stacks consecutive frames to capture motion
- **High Accuracy**: Target >85% overall accuracy, >80% per control state
- **Fast Inference**: <50ms per frame pair on GPU
- **Comprehensive Training Pipeline**: Includes data augmentation, early stopping, learning rate scheduling
- **Easy-to-Use API**: Simple interfaces for training and inference

## Architecture

```
Input: 2 consecutive frames (224x224x6)
  ↓
Conv Layer 1 (6 → 64 channels, 7x7 kernel)
  ↓
Batch Norm + ReLU + MaxPool
  ↓
ResNet-18 Residual Blocks (4 layers)
  ↓
Global Average Pooling
  ↓
FC Layers: 512 → 256 → 128 → 5
  ↓
Output: [P(stopped), P(forward), P(backward), P(rotate_left), P(rotate_right)]
```

**Model Stats:**
- Parameters: ~11 million
- Model size: ~44 MB
- Input size: 224x224x6 (2 RGB frames)
- Output size: 5 (probability distribution)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 0. Quick Test (Optional)

Before preparing real data, you can test the entire pipeline with synthetic data:

```bash
# Generate small synthetic dataset and run a quick training test
python quick_test.py
```

This will:
1. Generate 100 train, 30 val, 30 test synthetic frames
2. Train the model for 3 epochs
3. Run inference
4. Verify everything works (~2-5 minutes)

Or generate fake data manually:

```bash
# Generate synthetic data for testing (customizable sizes)
python generate_fake_data.py \
    --output_dir fake_data \
    --num_train 200 \
    --num_val 50 \
    --num_test 50

# Then train on fake data
python train.py \
    --train_dir fake_data/train \
    --val_dir fake_data/val \
    --test_dir fake_data/test \
    --batch_size 16 \
    --num_epochs 10
```

**Note**: Synthetic data uses colored frames corresponding to states (green=forward, red=backward, etc.) to help verify the pipeline, but won't achieve high accuracy since it's not real RC car footage.

### 1. Prepare Your Data

#### Option A: From Video File

```bash
python -m inverse_dynamics_model.data_preparation convert \
    video.mp4 \
    data/raw \
    --fps 30
```

This creates:
- Extracted frames: `data/raw/frame_0000.jpg`, `frame_0001.jpg`, ...
- Template labels: `data/raw/inputs.json`

Edit `inputs.json` to assign correct states:
- `0` = STOPPED
- `1` = FORWARD
- `2` = BACKWARD
- `3` = ROTATE_LEFT
- `4` = ROTATE_RIGHT

#### Option B: Manual Structure

Create this directory structure:
```
data/
  train/
    frame_0000.jpg
    frame_0001.jpg
    ...
    inputs.json
  val/
    frame_0000.jpg
    ...
    inputs.json
  test/
    frame_0000.jpg
    ...
    inputs.json
```

`inputs.json` format:
```json
[
  {"frame": 0, "time": 0.0, "state": 0},
  {"frame": 1, "time": 0.033, "state": 1},
  {"frame": 2, "time": 0.066, "state": 1}
]
```

#### Split Dataset

```bash
python -m inverse_dynamics_model.data_preparation split \
    data/raw \
    --train 0.7 \
    --val 0.15 \
    --test 0.15
```

#### Validate Dataset

```bash
python -m inverse_dynamics_model.data_preparation validate data/train
```

### 2. Train the Model

```bash
python train.py \
    --train_dir data/train \
    --val_dir data/val \
    --test_dir data/test \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.001
```

Training progress will be displayed with progress bars. Best model will be saved to `checkpoints/idm_best.pth`.

### 3. Run Inference

```bash
python example_inference.py \
    --model_path checkpoints/idm_best.pth \
    --frames frame_0010.jpg frame_0011.jpg \
    --benchmark
```

### 4. Visualize Predictions on Video

```bash
# Visualize predictions on a video file
python visualize_predictions.py \
    --model_path checkpoints/idm_best.pth \
    --video_path input_video.mp4 \
    --output_path predictions_visualization.mp4
```

This creates a video showing:
- Original video frames (top)
- Predicted control state with confidence (bottom panel)
- Probability distribution bars for all states

For testing with fake data:
```bash
# First, create a video from fake data frames
python create_test_video.py --data_dir fake_data/test --output test_video.mp4

# Then visualize predictions
python visualize_predictions.py \
    --model_path checkpoints/idm_best.pth \
    --video_path test_video.mp4 \
    --output_path predictions_visualization.mp4
```

## Usage Examples

### Training from Python

```python
from inverse_dynamics_model import IDMConfig, InverseDynamicsModel
from inverse_dynamics_model.trainer import train_model

# Create config
config = IDMConfig(
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    early_stopping_patience=10
)

# Train model
model, trainer = train_model(
    train_dir="data/train",
    val_dir="data/val",
    config=config
)
```

### Inference from Python

```python
from inverse_dynamics_model.inference import InverseDynamicsPredictor

# Load model
predictor = InverseDynamicsPredictor("checkpoints/idm_best.pth")

# Predict from frame paths
state_name, probabilities = predictor.predict_with_names(
    ["frame_0010.jpg", "frame_0011.jpg"],
    return_probabilities=True
)

print(f"Predicted state: {state_name}")
print(f"Probabilities: {probabilities}")

# Batch prediction
frame_sequences = [
    ["frame_0010.jpg", "frame_0011.jpg"],
    ["frame_0012.jpg", "frame_0013.jpg"],
]
predictions = predictor.predict_batch(frame_sequences)
```

### Custom Training Loop

```python
from inverse_dynamics_model import (
    IDMConfig,
    InverseDynamicsModel,
    RCCarInverseDynamicsDataset,
    InverseDynamicsTrainer
)
from torch.utils.data import DataLoader

# Create config
config = IDMConfig(batch_size=32, num_epochs=50)

# Create datasets
train_dataset = RCCarInverseDynamicsDataset("data/train", config, split="train")
val_dataset = RCCarInverseDynamicsDataset("data/val", config, split="val")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Create model
model = InverseDynamicsModel(config, pretrained=True)

# Create trainer
trainer = InverseDynamicsTrainer(model, config, train_loader, val_loader)

# Train
history = trainer.train()

# Evaluate
test_dataset = RCCarInverseDynamicsDataset("data/test", config, split="test")
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
metrics = trainer.evaluate(test_loader, save_dir="evaluation")
```

## Configuration

Key configuration parameters in `IDMConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_stacked_frames` | 2 | Number of consecutive frames |
| `use_grayscale` | False | Use grayscale instead of RGB |
| `image_size` | (224, 224) | Input image size |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 50 | Number of training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `dropout_rate` | 0.3 | Dropout rate |
| `early_stopping_patience` | 10 | Early stopping patience |
| `use_augmentation` | True | Enable data augmentation |

See `inverse_dynamics_model/config.py` for all options.

## Data Requirements

### Dataset Size Guidelines

| Size | Frames | Duration (30fps) | Use Case |
|------|--------|------------------|----------|
| Minimum | 1,000 | ~33 seconds | Quick testing |
| Recommended | 10,000 | ~5.5 minutes | Solid accuracy |
| Ideal | 50,000+ | ~28 minutes | Production quality |

### State Distribution (Target)

- STOPPED: 10-20%
- FORWARD: 35-45%
- BACKWARD: 5-10%
- ROTATE_LEFT: 15-20%
- ROTATE_RIGHT: 15-20%

### Diversity Requirements

- At least 2 different locations
- At least 2 different lighting conditions
- Multiple speed variations
- Frame rate: ≥30 FPS
- No single state >50% of dataset

## Performance Targets

- **Training time**: <2 hours for 10k frames, 50 epochs (with GPU)
- **Inference time**: <50ms per frame pair (with GPU)
- **Overall accuracy**: >85%
- **Per-state accuracy**: >80%

## Model Outputs

The model outputs a probability distribution over 5 control states:

```python
# Example output
{
    'STOPPED': 0.05,
    'FORWARD': 0.85,
    'BACKWARD': 0.02,
    'ROTATE_LEFT': 0.05,
    'ROTATE_RIGHT': 0.03
}
```

States are mutually exclusive (only one active at a time).

## Evaluation Metrics

The training pipeline computes:

- **Overall accuracy**: Percentage of correct predictions
- **Per-class metrics**: Precision, recall, F1-score for each state
- **Confusion matrix**: 5x5 matrix showing prediction patterns
- **Visualizations**: Loss curves, accuracy curves, per-class performance

Evaluation results are saved to the specified output directory.

## Project Structure

```
inverse_dynamics_model/
├── __init__.py              # Package initialization
├── config.py                # Configuration and hyperparameters
├── model.py                 # Model architecture (ResNet-18 based)
├── dataset.py               # Dataset loading and preprocessing
├── trainer.py               # Training loop and evaluation
├── inference.py             # Inference API
├── data_preparation.py      # Data processing utilities
└── utils.py                 # Helper functions

train.py                     # Training script
example_inference.py         # Inference example
visualize_predictions.py     # Visualize predictions on video
create_test_video.py         # Create test video from frames
generate_fake_data.py        # Generate synthetic data for testing
quick_test.py                # Quick pipeline test
test_installation.py         # Installation verification
requirements.txt             # Dependencies
README.md                    # This file
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py --batch_size 16 ...
```

### Poor Accuracy

1. **Check dataset balance**: Validate state distribution
2. **Increase dataset size**: More diverse data improves accuracy
3. **Adjust learning rate**: Try lower values (0.0001)
4. **Enable augmentation**: `--use_augmentation` (enabled by default)

### Slow Training

1. **Use GPU**: Ensure CUDA is available
2. **Increase batch size**: If GPU memory allows
3. **Reduce workers**: `--num_workers 2` on systems with limited CPU
4. **Use grayscale**: `--use_grayscale` reduces input size

## Testing

### Quick Test

Test the entire pipeline with synthetic data:

```bash
python quick_test.py
```

### Installation Test

Verify installation and imports:

```bash
python test_installation.py
```

### Manual Tests

```bash
# Test data loading
python -c "from inverse_dynamics_model.dataset import RCCarInverseDynamicsDataset; from inverse_dynamics_model.config import IDMConfig; ds = RCCarInverseDynamicsDataset('data/train', IDMConfig()); print(f'Loaded {len(ds)} samples')"

# Test model forward pass
python -c "from inverse_dynamics_model import InverseDynamicsModel, IDMConfig; import torch; model = InverseDynamicsModel(IDMConfig()); x = torch.randn(1, 6, 224, 224); y = model(x); print(f'Output shape: {y.shape}')"
```

### Generate Synthetic Data

For testing without real data:

```bash
# Small dataset (quick test)
python generate_fake_data.py --num_train 100 --num_val 30 --num_test 30

# Larger dataset (better for testing training dynamics)
python generate_fake_data.py --num_train 1000 --num_val 200 --num_test 200

# Use random noise instead of colored frames
python generate_fake_data.py --random_noise
```

## Citation

If you use this code in your research, please cite:

```
@software{inverse_dynamics_model,
  title={Inverse Dynamics Model for RC Car Control Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/idm}
}
```

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the code documentation

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Changelog

### Version 1.0.0 (2025-01-XX)
- Initial release
- ResNet-18 based architecture
- Complete training pipeline
- Inference API
- Data preparation utilities
