# Goods Counter 

A real-time goods/object counter using SAM3 (Segment Anything Model 3) video tracking. This application uses a multi-threaded architecture to count objects in video feeds with customizable cargo categories like sacks, boxes, and crates.

## Features

- **Real-time video processing** using SAM3 video tracking model
- **Multi-threaded architecture** for decoupled capture, inference, and rendering
- **Configurable cargo categories** (sack, box, crate) with custom prompts and colors
- **Noise filtering** - requires confirmed object detections over multiple frames before updating count
- **GPU memory management** - reactive memory management to prevent out-of-memory errors
- **MLflow integration** - automatic experiment tracking and metrics logging
- **Confidence threshold filtering** - configurable detection confidence levels

## Architecture

The application uses a three-thread model:

1. **Capture Thread** - Reads frames from video source (webcam or file)
2. **Inference Thread** - Runs SAM3 forward passes on the latest available frame
3. **Main Thread** - Renders overlays, displays results, and logs metrics via MLflow

This decoupled design ensures smooth real-time performance with no blocking between capture, inference, and display.

## Requirements

See `requirements.txt` for all dependencies. Key requirements:

- Python 3.8+
- PyTorch with CUDA support (optional but recommended for GPU acceleration)
- OpenCV
- Transformers
- Accelerate
- MLflow

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the model registry name in `config.py` if needed.

## Usage

### Command Line

Run the counter from the command line:

```bash
python -m good_counter.main
```

### MLflow Projects

Run via MLflow Projects:

```bash
mlflow run . -e count_goods_registry --env-manager=local
```

### Configuration

Edit `config.py` to customize:

- `VIDEO_SOURCE` - Input source (0 for webcam, or path to video file)
- `MODEL_REGISTRY_NAME` - Name of the SAM3 model in MLflow registry
- `CARGO_CLASSES` - Object categories to detect with custom prompts and display colors
- `INFER_WIDTH` - Model inference width (trades speed vs. accuracy)
- `CONFIDENCE_THRESHOLD` - Minimum confidence for detections
- `COUNT_CONFIRM_FRAMES` - Frames to hold steady before accepting new count
- `GPU_MEMORY_RESET_THRESHOLD_MB` - Memory threshold for session reset
- `MLFLOW_EXPERIMENT_NAME` - Experiment name for logging

## Controls

- **ESC** - Exit the application

## Modules

- **main.py** - Entry point and main application loop
- **config.py** - All configuration constants
- **capture.py** - Video capture thread worker
- **inference.py** - SAM3 inference thread worker
- **rendering.py** - Display and visualization
- **counting.py** - Goods counting logic with noise filtering
- **model_loader.py** - Model loading and device setup
- **mlflow_tracking.py** - MLflow experiment tracking
- **sessions.py** - SAM3 tracking session management
- **shared_state.py** - Thread-safe shared state
- **__init__.py** - Package initialization

## Performance

- **GPU Recommended** - Runs on CPU but will be significantly slower
- **Tested GPU** - NVIDIA GPUs with CUDA support
- **Resolution** - Adaptive to input video resolution

## Notes

- The application automatically resets tracking sessions when GPU memory usage exceeds configured threshold
- Count updates are filtered to reduce noise - a genuine new object must stay visible for multiple consecutive frames
- Object IDs shown on-screen are cosmetic and may change after session resets
- Final counts are logged to console and MLflow upon exit
