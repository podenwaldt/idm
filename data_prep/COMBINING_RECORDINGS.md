# Combining Multiple Recording Runs

The `combine_recordings.py` script merges multiple RC car recording runs into a single training dataset.

## Quick Start

```bash
# Basic usage - combine multiple recordings
python combine_recordings.py recording1/ recording2/ recording3/ --output data/train/

# Use wildcards to combine all recordings
python combine_recordings.py recording_* --output combined_data/

# Combine with continuous timestamps
python combine_recordings.py rec1/ rec2/ --continuous-time --output combined/
```

## What It Does

The script will:

1. **Renumber all frames sequentially** - Frames from multiple recordings are renumbered starting from `frame_0000.jpg`
2. **Merge inputs.json files** - All label entries are combined with adjusted frame indices
3. **Copy frame files** - All frames are copied to the output directory with new names
4. **Generate statistics** - Shows state distribution and frame counts

## Input Format

Each recording directory should contain:

```
recording1/
├── frame_0000.jpg
├── frame_0001.jpg
├── frame_0002.jpg
├── ...
└── inputs.json
```

Where `inputs.json` has the format:

```json
[
  {"frame": 0, "time": 0.000, "state": 0},
  {"frame": 1, "time": 0.033, "state": 1},
  {"frame": 2, "time": 0.067, "state": 1}
]
```

## Options

### `--output` (required)
Output directory where the combined dataset will be saved.

```bash
python combine_recordings.py rec1/ rec2/ --output data/train/
```

### `--continuous-time`
Adjusts timestamps to be continuous across recordings. Without this flag, original timestamps are preserved (may have gaps or overlaps).

**Without flag:**
- Recording 1: times 0.0 - 10.0s
- Recording 2: times 0.0 - 8.0s (starts from 0 again)

**With flag:**
- Recording 1: times 0.0 - 10.0s
- Recording 2: times 10.0 - 18.0s (continues from recording 1)

```bash
python combine_recordings.py rec1/ rec2/ --continuous-time --output combined/
```

### `--time-gap`
Adds a time gap (in seconds) between recordings. Only works with `--continuous-time`.

```bash
# Add 1 second gap between recordings
python combine_recordings.py rec1/ rec2/ --continuous-time --time-gap 1.0 --output combined/
```

## Example Workflow

### 1. Collect multiple recordings

```
recordings/
├── recording1/
│   ├── frame_0000.jpg ... frame_0150.jpg
│   └── inputs.json (151 frames)
├── recording2/
│   ├── frame_0000.jpg ... frame_0200.jpg
│   └── inputs.json (201 frames)
└── recording3/
    ├── frame_0000.jpg ... frame_0099.jpg
    └── inputs.json (100 frames)
```

### 2. Combine them

```bash
python combine_recordings.py recordings/recording* --output data/train/ --continuous-time
```

### 3. Result

```
data/train/
├── frame_0000.jpg ... frame_0450.jpg  # 151 + 201 + 100 = 452 frames
└── inputs.json  # 452 entries with continuous frame numbers and timestamps
```

## Output

The script provides detailed output:

```
Combining 3 recording(s)...
Output: data/train
Continuous time: True

[1/3] Processing recording1...
Validating recording1...
  Found 151 frames and 151 entries in inputs.json
  Copied 151 frames (total so far: 151)

[2/3] Processing recording2...
Validating recording2...
  Found 201 frames and 201 entries in inputs.json
  Copied 201 frames (total so far: 352)

[3/3] Processing recording3...
Validating recording3...
  Found 100 frames and 100 entries in inputs.json
  Copied 100 frames (total so far: 452)

✓ Successfully combined 3 recordings!
  Total frames: 452
  Output directory: data/train
  inputs.json: data/train/inputs.json

State distribution:
  0 (STOPPED     ):   67 frames ( 14.8%)
  1 (FORWARD     ):  215 frames ( 47.6%)
  2 (BACKWARD    ):   36 frames (  8.0%)
  3 (ROTATE_LEFT ):   67 frames ( 14.8%)
  4 (ROTATE_RIGHT):   67 frames ( 14.8%)
```

## Tips

1. **Backup first**: Always keep your original recordings before combining
2. **Check state distribution**: The output shows if you have balanced data across all states
3. **Timestamp continuity**: Use `--continuous-time` if you want seamless temporal flow
4. **Validation**: After combining, validate the dataset:
   ```bash
   python -m inverse_dynamics_model.data_preparation validate data/train/
   ```

## Integrating with Training Pipeline

After combining recordings:

```bash
# 1. Combine your recordings
python combine_recordings.py recordings/* --output data/all_data/ --continuous-time

# 2. Split into train/val/test (if needed)
# You can manually split or use data_preparation.py split functionality

# 3. Start training
python train.py --train_dir data/train --val_dir data/val --test_dir data/test
```

## Troubleshooting

### "No inputs.json found"
Make sure each recording directory has an `inputs.json` file.

### "No frame files found"
Check that frames are named `frame_XXXX.jpg` (with 4-digit zero-padded numbers).

### Frame/label mismatches
The script will warn you if there are frames in inputs.json that don't exist on disk, or vice versa. These mismatches will be skipped.

### State values
Valid state values are 0-4:
- 0: STOPPED
- 1: FORWARD
- 2: BACKWARD
- 3: ROTATE_LEFT
- 4: ROTATE_RIGHT
