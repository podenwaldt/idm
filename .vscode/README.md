# VSCode Configuration for Inverse Dynamics Model

This directory contains VSCode configuration files for easy development and testing of the IDM project.

## Quick Start

### Run with F5 (Debug/Run)

1. Press **F5** or click the "Run and Debug" icon in the left sidebar
2. Select one of the pre-configured launch configurations from the dropdown:

#### Testing & Verification
- **Quick Test (Full Pipeline)** - Run complete pipeline test (generate data + train + inference)
- **Test Installation** - Verify all dependencies are installed correctly

#### Data Generation
- **Generate Fake Data (Small)** - Create 200/50/50 synthetic frames (quick test)
- **Generate Fake Data (Large)** - Create 1000/200/200 synthetic frames (better training)

#### Training
- **Train Model (Fake Data - Quick)** - Train for 10 epochs on synthetic data
- **Train Model (Real Data)** - Train for 50 epochs on real data

#### Inference
- **Run Inference (Test Frames)** - Run inference on test frames with benchmarking

#### Data Preparation
- **Data Prep: Convert Video** - Convert video to frames and inputs.json
- **Data Prep: Split Dataset** - Split dataset into train/val/test
- **Data Prep: Validate Dataset** - Validate dataset structure

#### General
- **Python: Current File** - Run/debug the currently open Python file

### Run with Tasks (Ctrl+Shift+B)

1. Press **Ctrl+Shift+P** (Cmd+Shift+P on Mac) to open command palette
2. Type "Tasks: Run Task"
3. Select from available tasks:

#### Quick Tasks
- **Quick Test** - Full pipeline test (default build task)
- **Test Installation** - Verify installation

#### Data Tasks
- **Generate Fake Data (Small)** - 200/50/50 frames
- **Generate Fake Data (Large)** - 1000/200/200 frames
- **Convert Video to Frames** - Interactive: prompts for video path
- **Split Dataset** - Interactive: prompts for source directory
- **Validate Dataset** - Interactive: prompts for dataset directory

#### Training Tasks
- **Train Model (Fake Data - Quick)** - 10 epochs on synthetic data
- **Train Model (Real Data)** - 50 epochs on real data

#### Utility Tasks
- **Clean Test Files** - Remove fake_data, test_checkpoints, etc.
- **Install Requirements** - Run pip install -r requirements.txt

### Default Build Task

Press **Ctrl+Shift+B** (Cmd+Shift+B on Mac) to run the default build task: **Quick Test**

## Configuration Files

### launch.json
Defines debug/run configurations accessible via F5 or the Run panel.

**Key Features:**
- Pre-configured arguments for each script
- Integrated terminal output
- Proper working directory setup
- Python debugging support

### tasks.json
Defines tasks accessible via Command Palette → "Tasks: Run Task"

**Key Features:**
- Interactive tasks with input prompts
- Default build task (Quick Test)
- Dedicated terminal panels for long-running tasks
- Clean output presentation

### settings.json
Project-specific VSCode settings for Python development.

**Key Features:**
- Python interpreter configuration
- Linting with flake8
- Auto-import completions
- Exclude data/checkpoint directories from search
- PYTHONPATH setup
- Editor settings (tab size, rulers)

### extensions.json
Recommended VSCode extensions for this project.

**Recommended Extensions:**
- **ms-python.python** - Python language support
- **ms-python.vscode-pylance** - Fast Python language server
- **ms-python.debugpy** - Python debugger
- **ms-toolsai.jupyter** - Jupyter notebook support
- **njpwerner.autodocstring** - Auto-generate docstrings
- **kevinrose.vsc-python-indent** - Correct Python indentation
- **visualstudioexptteam.vscodeintellicode** - AI-assisted IntelliSense

VSCode will prompt you to install these when you open the project.

## Common Workflows

### 1. First-Time Setup
```
F5 → "Test Installation"
```
This verifies all dependencies are installed correctly.

### 2. Quick Pipeline Test
```
F5 → "Quick Test (Full Pipeline)"
```
or
```
Ctrl+Shift+B (runs default build task)
```
Generates synthetic data, trains for 3 epochs, and runs inference.

### 3. Development with Fake Data
```
1. F5 → "Generate Fake Data (Small)"
2. F5 → "Train Model (Fake Data - Quick)"
3. F5 → "Run Inference (Test Frames)"
```

### 4. Working with Real Data
```
1. Ctrl+Shift+P → "Tasks: Run Task" → "Convert Video to Frames"
   (Enter your video path when prompted)

2. Edit data/raw/inputs.json to assign correct states

3. Ctrl+Shift+P → "Tasks: Run Task" → "Split Dataset"

4. F5 → "Train Model (Real Data)"
```

### 5. Debugging
1. Set breakpoints in code by clicking left of line numbers
2. Press F5 and select appropriate configuration
3. Use debug controls: Continue (F5), Step Over (F10), Step Into (F11), Step Out (Shift+F11)

## Customizing Configurations

### Modify Launch Configuration Arguments

Edit `.vscode/launch.json` and change the `args` array:

```json
{
    "name": "Train Model (Custom)",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/train.py",
    "args": [
        "--train_dir", "my_data/train",
        "--val_dir", "my_data/val",
        "--batch_size", "64",
        "--num_epochs", "100"
    ]
}
```

### Add New Task

Edit `.vscode/tasks.json`:

```json
{
    "label": "My Custom Task",
    "type": "shell",
    "command": "python",
    "args": ["script.py", "--arg", "value"],
    "problemMatcher": []
}
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **F5** | Start debugging/run selected configuration |
| **Ctrl+Shift+B** | Run default build task (Quick Test) |
| **Ctrl+Shift+P** | Open command palette (access all tasks) |
| **Ctrl+`** | Toggle integrated terminal |
| **Ctrl+Shift+`** | Create new terminal |
| **F9** | Toggle breakpoint |
| **F10** | Step over (during debug) |
| **F11** | Step into (during debug) |
| **Shift+F11** | Step out (during debug) |

## Tips

1. **Multiple Terminals**: Tasks run in separate terminals, so you can have training running in one terminal while generating data in another.

2. **Quick Access**: Use Ctrl+Shift+P and type "Tasks" to quickly access all tasks without remembering shortcuts.

3. **Persistent Terminals**: Task terminals remain open after completion so you can review output.

4. **Interactive Tasks**: Some tasks (like "Convert Video") will prompt you for inputs. You can modify the default values in `tasks.json` → `inputs` section.

5. **Clean Workspace**: Use the "Clean Test Files" task to quickly remove all generated test data and checkpoints.

6. **Python Environment**: If using a virtual environment, VSCode will auto-detect it. You can manually select it via Ctrl+Shift+P → "Python: Select Interpreter".

## Troubleshooting

### "Python interpreter not found"
- Open Command Palette (Ctrl+Shift+P)
- Type "Python: Select Interpreter"
- Choose your Python installation

### "Module not found" errors
- Run the "Install Requirements" task
- Or manually: `pip install -r requirements.txt`

### Tasks not showing up
- Reload VSCode: Ctrl+Shift+P → "Developer: Reload Window"
- Check that you're in the correct workspace folder

### Debug not working
- Ensure the Python extension is installed
- Check that `debugpy` is installed: `pip install debugpy`

## Additional Resources

- [VSCode Python Documentation](https://code.visualstudio.com/docs/python/python-tutorial)
- [VSCode Debugging Guide](https://code.visualstudio.com/docs/editor/debugging)
- [VSCode Tasks Documentation](https://code.visualstudio.com/docs/editor/tasks)
