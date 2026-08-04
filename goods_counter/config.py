"""
All tunable constants live here. Nothing in this file has side effects,
so it's safe to import from anywhere (including tests) without triggering
model loads, camera opens, or MLflow runs.
"""

VIDEO_SOURCE = 0  # 0 = webcam. Set to a file path string to use a video file instead.

MODEL_REGISTRY_NAME = "sam3-video-tracking-model"  # must match register_model.py

# Categories kept deliberately distinct in shape so SAM3 (and you, reading the
# counts) can actually tell them apart. Cardboard box / carton / package were
# merged into one "box" category since they're the same real-world object.
CARGO_CLASSES = {
    "sack":  {"prompt": "sack or bag",             "color": (0, 165, 255)},  # orange
    "box":   {"prompt": "cardboard box or carton", "color": (255, 100, 0)},  # blue
    "crate": {"prompt": "wooden or plastic crate", "color": (255, 0, 255)},  # magenta
}

INFER_WIDTH = 384
CONFIDENCE_THRESHOLD = 0.5
MASK_ALPHA = 0.35

SESSION_RESET_EVERY_N_INFER = 1500
LOG_GPU_MEMORY_EVERY_N_INFER = 200

MLFLOW_EXPERIMENT_NAME = "goods-count-from-registry"
MLFLOW_LOG_EVERY_N_FRAMES = 30

WINDOW_NAME = "SAM3 Goods Counter (from Registry)"