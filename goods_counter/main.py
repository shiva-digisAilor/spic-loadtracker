"""
Same behavior as the original single-file script, just wired together from
the modules in this package instead of being one long file.

THREADING MODEL (three threads, decoupled by shared/locked buffers):
  1. capture_thread   -> only calls cap.read(); never blocked by inference
                         or rendering.
  2. inference_worker -> only runs SAM3 forward passes on the latest
                         available frame; never blocked by capture I/O
                         or cv2.imshow.
  3. main thread       -> only draws overlays / imshow / waitKey / MLflow
                         logging on the latest available frame + result.

Usage:
    python -m goods_counter.main
or via MLflow Projects:
    mlflow run . -e count_goods_registry --env-manager=local
"""

import threading

import cv2
import torch

from good_counter2 import config
from good_counter2 import model_loader
from good_counter2 import mlflow_tracking
from good_counter2.sessions import start_new_sessions
from good_counter2.shared_state import SharedState
from good_counter2.counting import GoodsCounter
from good_counter2.capture import capture_worker
from good_counter2.inference import inference_worker, SessionHolder
from good_counter2.rendering import render_loop


def main():
    # ---- device / model / processor ----
    accelerator, device, using_gpu = model_loader.setup_accelerator()
    video_model, latest_version = model_loader.load_model_from_registry(
        config.MODEL_REGISTRY_NAME, device
    )
    video_processor = model_loader.load_processor()
    torch.backends.cudnn.benchmark = True

    # ---- MLflow run for THIS session's metrics (separate from model registry) ----
    mlflow_tracking.start_run(config.MLFLOW_EXPERIMENT_NAME)
    mlflow_tracking.log_run_params(
        model_registry_name=config.MODEL_REGISTRY_NAME,
        latest_version=latest_version,
        device=device,
        using_gpu=using_gpu,
        infer_width=config.INFER_WIDTH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        video_source=config.VIDEO_SOURCE,
        cargo_classes=config.CARGO_CLASSES,
    )

    # ---- video source ----
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.VIDEO_SOURCE}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source opened: {width}x{height}")

    scale = config.INFER_WIDTH / width
    infer_w, infer_h = config.INFER_WIDTH, int(height * scale)

    # ---- tracking sessions (persistent, periodically reset) / shared state / counting ----
    session_holder = SessionHolder(start_new_sessions(video_processor, device, config.CARGO_CLASSES))
    print("Streaming session started. Press ESC to quit.")

    state = SharedState(config.CARGO_CLASSES)
    counter = GoodsCounter(config.CARGO_CLASSES, confirm_frames=config.COUNT_CONFIRM_FRAMES)

    # ---- start capture + inference threads ----
    capture_thread = threading.Thread(target=capture_worker, args=(cap, state), daemon=True)
    worker_thread = threading.Thread(
        target=inference_worker,
        args=(state, session_holder, video_model, video_processor, device, using_gpu,
              config.CARGO_CLASSES, scale, infer_w, infer_h,
              config.CONFIDENCE_THRESHOLD,
              config.GPU_MEMORY_RESET_THRESHOLD_MB,
              config.CPU_FALLBACK_RESET_EVERY_N_INFER,
              config.LOG_GPU_MEMORY_EVERY_N_INFER),
        daemon=True,
    )
    capture_thread.start()
    worker_thread.start()

    # ---- main thread: render loop ----
    frame_idx = 0
    try:
        frame_idx = render_loop(
            state, counter, width, height, config.CARGO_CLASSES, config.MASK_ALPHA,
            config.WINDOW_NAME, config.MLFLOW_LOG_EVERY_N_FRAMES, using_gpu, device,
        )
    finally:
        state.stop_flag = True
        capture_thread.join(timeout=2)
        worker_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        grand_total = counter.grand_total()
        print(f"Stopped. Frames rendered: {frame_idx}.")
        print("Final counts:", counter.total_count, "| Grand total:", grand_total)

        mlflow_tracking.log_final_summary(frame_idx, counter)


if __name__ == "__main__":
    main()