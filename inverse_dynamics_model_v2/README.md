# Inverse Dynamics Model V2

## Overview

Version 2 of the Inverse Dynamics Model introduces significant improvements over V1:

### Key Improvements

1. **MobileNetV2 Architecture**
   - More efficient than ResNet-18
   - Smaller model size and faster inference
   - Better suited for mobile/embedded deployment

2. **4-Frame Temporal Context**
   - Uses 4 consecutive frames (t-1, t, t+1, t+2) instead of 2
   - Predicts action at time t using both past and future context (true inverse dynamics)
   - Provides richer temporal information for control prediction
   - Input shape: (12, 224, 224) = 4 frames × 3 RGB channels

### Architecture Details

```
Input: 4 RGB frames stacked (12 channels, 224x224)
  ↓
MobileNetV2 Feature Extractor
  ↓
Global Average Pooling (1280 features)
  ↓
FC1: 1280 → 256 + Dropout(0.3)
  ↓
FC2: 256 → 128 + Dropout(0.3)
  ↓
FC3: 128 → 5 (control states)
```

### Model Comparison

| Feature | V1 | V2 |
|---------|----|----|
| Architecture | ResNet-18 | MobileNetV2 |
| Input Frames | 2 | 4 |
| Input Channels | 6 | 12 |
| Parameters | ~11M | ~3-4M |
| Model Size | ~44 MB | ~14-16 MB |
| Inference Speed | Baseline | 2-3x faster |

## Usage

### Training V2 Model

```bash
# Train with default settings (4 frames, MobileNetV2)
python train.py --model_version v2 \
    --train_dir data/train \
    --val_dir data/val \
    --test_dir data/test

# Custom training parameters
python train.py --model_version v2 \
    --train_dir data/train \
    --val_dir data/val \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --checkpoint_dir checkpoints_v2
```

### Evaluating V2 Model

```bash
python evaluate.py --model_version v2 \
    --checkpoint checkpoints_v2/idm_final.pth \
    --test_dir data/test \
    --output_dir evaluation_v2
```

### Inference with V2

```python
from inverse_dynamics_model_v2 import (
    IDMConfig,
    InverseDynamicsModel,
    InverseDynamicsPredictor
)

# Create config
config = IDMConfig()

# Create model
model = InverseDynamicsModel(config, pretrained=True)

# Load trained weights
model.load_state_dict(torch.load('checkpoints_v2/idm_final.pth')['model_state_dict'])

# Create predictor
predictor = InverseDynamicsPredictor(model, config)

# Predict from 4 frames
control_state = predictor.predict_from_frame_paths([
    'frame_0000.jpg',
    'frame_0001.jpg',
    'frame_0002.jpg',
    'frame_0003.jpg'
])
```

## Configuration

The V2 model uses the same configuration structure as V1 with additional parameters:

```python
config = IDMConfig(
    # V2 defaults
    num_stacked_frames=4,           # 4 frames instead of 2
    mobilenet_width_mult=1.0,       # MobileNetV2 width multiplier
    mobilenet_pretrained=True,      # Use ImageNet pretrained weights

    # Training (same as V1)
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    dropout_rate=0.3,

    # Checkpointing
    checkpoint_dir="checkpoints_v2"  # Separate checkpoint dir for v2
)
```

## Expected Performance

With 4 frames and MobileNetV2 architecture, V2 is expected to provide:

- **Better temporal understanding**: More frames capture motion dynamics better
- **Faster inference**: MobileNetV2 is optimized for efficiency
- **Smaller model size**: Easier to deploy on edge devices
- **Similar or better accuracy**: Additional temporal context compensates for smaller architecture

## Files Structure

```
inverse_dynamics_model_v2/
├── __init__.py              # Package initialization
├── config.py                # Configuration with V2 defaults
├── model.py                 # MobileNetV2-based architecture
├── dataset.py               # Dataset loader (supports 4 frames)
├── trainer.py               # Training pipeline
├── inference.py             # Inference API
├── utils.py                 # Utility functions
├── data_preparation.py      # Data preprocessing
└── README.md               # This file
```

## Training Tips

1. **Batch Size**: MobileNetV2 uses less memory, so you can increase batch size
2. **Learning Rate**: Start with 0.001, reduce if training is unstable
3. **Data Augmentation**: Same augmentation as V1 works well
4. **Frame Spacing**: Ensure frames are consecutive for best results

## Backwards Compatibility

V2 maintains the same interface as V1, making it easy to switch between versions:

```python
# Switch from V1 to V2 by changing import
# from inverse_dynamics_model import ...  # V1
from inverse_dynamics_model_v2 import ...  # V2

# Everything else stays the same!
```

## Future Improvements

Potential enhancements for future versions:

- [ ] Temporal attention mechanisms
- [ ] 3D convolutions for better temporal modeling
- [ ] Optical flow as additional input channel
- [ ] EfficientNet or Vision Transformer backbones
- [ ] Variable frame rates and spacing
