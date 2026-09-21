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

INFER_WIDTH = 512
CONFIDENCE_THRESHOLD = 0.5
MASK_ALPHA = 0.35

# Reactive memory management: once GPU memory currently in use crosses this
# many MB, all tracking sessions are reset right then (not on a fixed
# schedule). Tune based on your GPU's total VRAM -- e.g. on a 6GB card,
# something like 4000-4500 leaves reasonable headroom; on a 24GB card you
# could set this much higher.
GPU_MEMORY_RESET_THRESHOLD_MB = 4500

# Only used as a fallback when no GPU is available (memory can't be
# queried the same way on CPU) -- resets every N cycles instead.
CPU_FALLBACK_RESET_EVERY_N_INFER = 1500

LOG_GPU_MEMORY_EVERY_N_INFER = 200

MLFLOW_EXPERIMENT_NAME = "goods-count-from-registry"
MLFLOW_LOG_EVERY_N_FRAMES = 30

# A newly-detected higher count must hold steady for this many consecutive
# render frames before the grand total accepts it. Filters out one-frame
# noise spikes (mask fragmentation, borderline confidence, motion blur)
# without needing appearance memory or identity tracking. Raise this if
# you still see occasional false bumps; lower it if a genuine new object
# takes too long to register.
COUNT_CONFIRM_FRAMES = 8

WINDOW_NAME = "SAM3 Goods Counter"