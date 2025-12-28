# Clashgent: Training a model to play Clash Royale autonomously

I wanted to play 2v2 Clash with my boyfriend but I realized I don't have enough cards and it is too time consuming to play Clash Royale on my own solo, so I'm going to train a neural network to play for me, and run this in the background.

## Overview

Currently using:

- Bluestacks (emulator, gets around Clash Royale emulator detection)

## Project Structure

```
clashgent/
├── configs/           # Configuration files
│   └── example.json   # Example config (copy and customize)
├── scripts/           # Utility scripts for ADB/emulator
├── src/clashgent/     # Main Python package
│   ├── bridges/       # Emulator interaction (screenshot, actions)
│   ├── game/          # Game state and action definitions
│   ├── rl/            # Reinforcement learning (PPO, environment)
│   ├── verifiers/     # Reward shaping plugins
│   └── vision/        # YOLO-style object detection
└── train.py           # Training entry point
```

## Configuration

Create your own config by copying the example:

```bash
cp configs/example.json configs/myconfig.json
```

Edit `myconfig.json` to match your setup. Key fields to configure:

```json
{
  "device": "mps",                              // "cuda", "mps", or "cpu"
  "adb": {
    "adb_path": "adb",                          // or full path like "/Users/you/Library/Android/sdk/platform-tools/adb"
    "device_id": "127.0.0.1:5555",              // BlueStacks default
    "bluestacks_path": "/Applications/BlueStacks.app"
  }
}
```

Then run training with your config:

```bash
python train.py --config configs/myconfig.json
```

Or use command-line overrides:

```bash
python train.py --config configs/myconfig.json --timesteps 500000 --lr 0.001
```

## Quick Start

```bash
# Install with uv
uv venv && uv pip install -e .

# Test with mock environment (no emulator needed)
python train.py --mock

# Run with real BlueStacks
python train.py --config configs/myconfig.json
```
