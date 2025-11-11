# nAGV: Inverse Dynamics Model for RC Car Control

This project investigates whether an inverse dynamics model can be constructed to predict the input controls of an RC car given frames of video filmed from the front of the car. This involved building an RC car that could remotely receive commands, record them, and align them to frames being captured from the onboard camera. 

Approximately 12k frames were captured from this vehicle driving around an NYC apartment. 

All firmware for the RC car can be found in this repository: [cc](https://github.com/podenwaldt/cc)



## What It Does

Given consecutive video frames from an RC car's camera, the model predicts which control is active:
- STOPPED (0)
- FORWARD (1)
- BACKWARD (2)
- ROTATE_LEFT (3)
- ROTATE_RIGHT (4)

This is done by training an off-the-shelf CNN on stacked input frames paired with controls active per frame. Frame stacking for each input sample naively implies temporal information and takes advantage of non-causality during training. Optimal model performance was found at a stack size of 3, predicting the action at time t in a frame stack of (t-1, t, t+1). 

## Model Versions

Two models are auditioned in this project to investigate if there are significant differences in performance based on model size. Within each model, various settings for frame-stacking are tested to audition different levels of "naive-temporal" information included per sample. 


### V1: ResNet-18 
- Architecture: ResNet-18
- Parameters: ~11M
- Model size: ~44 MB

### V2: MobileNetV2 
- Architecture: MobileNetV2
- Parameters: ~3-4M
- Model size: ~14-16 MB

## Data in Use

### Overview:

- Total samples: 11,877
    - Training: 9,994
    - Validation: 1,212
    - Test: 671
- Varied lighting conditions
- ~30 mins of footage
- ~5fps avg
- ~100 laps around apartment

### Training State Distribution:            
- STOPPED         (0):   1814 (18.15%)
- FORWARD         (1):   2726 (27.28%)
- BACKWARD        (2):   1544 (15.45%)
- ROTATE_LEFT     (3):   1995 (19.96%)
- ROTATE_RIGHT    (4):   1915 (19.16%)



## System Performance: High Level

Models perform fairly well when evaluated on new videos sourced from the same envrionment (NYC apartment). They appear to struggle with any generalization beyond this environment, likely due to specific and limited training data. Further investigation is required in this area. 

### MobileNetV2 trained on 3-frame stacks

- Overall Accuracy: 0.9194
- Macro Precision:  0.9130
- Macro Recall:     0.9157
- Macro F1-Score:   0.9130

### ResNet18 trained on 3-frame stacks

- Overall Accuracy: 0.9149
- Macro Precision:  0.9084
- Macro Recall:     0.9137
- Macro F1-Score:   0.9101



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

**Optional [UNTESTED] : Augment data for better training:** 

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
!python train.py \
  --model_version v1 \
  --train_dir data/train \
  --val_dir data/val \
  --test_dir data/test \
  --num_stacked_frames 2 \
  --batch_size 128 \
  --num_epochs 20
```

**Train V2 (MobileNetV2, 3 frames):**

```bash
!python train.py \
  --model_version v2 \
  --train_dir data/train \
  --val_dir data/val \
  --test_dir data/test \
  --num_stacked_frames 3 \
  --batch_size 128 \
  --num_epochs 20
```

The best model will be saved to `checkpoints/idm_best.pth`.

### Evaluate a Model

Note: If you supply a --test_dir during training, evaluation will take place automatically post-training.

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

Note: Must match model version and num_stacked_frames it was trained on

```bash
# Create visualization video
    python visualizations/visualize_predictions.py \
        --model_path idm_final.pth \
        --model_version v1 \
        --video_path recordings/demo/test_video.mp4\
        --output_path visualizations/outputs/test_video_preds.mp4 \
        --num_stacked_frames 3
```

### Create Test Video

Needed if you start from raw frames instead of video.
--data_dir should be a folder of frames. 

```bash
# Create visualization video
    python visualizations/create_test_video.py \
    --data_dir recordings/demo/recording_385126 \
    --output recordings/demo/demo_385.mp4
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

Note: These are untested!

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


## Inference

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

### Worker errors
- Set --num_workers 0 in training call
- Get a GPU

### Model size mismatches in evaluation or training
- Verify model version is set correctly in call

-------------
-------------

Author: Paul Odenwaldt (& Claude)