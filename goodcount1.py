import cv2
import time
import torch
import threading
import numpy as np
import mlflow
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor

# ============================================================
# CONFIG
# ============================================================

VIDEO_SOURCE = 0  # 0 = webcam. Set to a file path string to use a video file instead.

# Each cargo type gets its own tracking session, its own count, and its own
# color in the display. Colors are in BGR (OpenCV format), not RGB.
CARGO_CLASSES = {
    "sack":  {"prompt": "sack or bag",           "color": (0, 165, 255)},   # orange
    "box":   {"prompt": "cardboard box or carton", "color": (255, 100, 0)}, # blue
    "crate": {"prompt": "wooden or plastic crate", "color": (255, 0, 255)}, # magenta
}

INFER_WIDTH = 384          # lower this (e.g. 256) if FPS is too slow with 5 classes
CONFIDENCE_THRESHOLD = 0.5
MASK_ALPHA = 0.35          # strength of the highlighted-region fill

SESSION_RESET_EVERY_N_INFER = 1500
LOG_GPU_MEMORY_EVERY_N_INFER = 200

MLFLOW_EXPERIMENT_NAME = "goods-count-simple"
MLFLOW_LOG_EVERY_N_FRAMES = 30

WINDOW_NAME = "SAM3 Goods Counter"

# ============================================================
# Setup
# ============================================================

accelerator = Accelerator()
device = accelerator.device
using_gpu = (device.type == "cuda") if hasattr(device, "type") else ("cuda" in str(device))
if not using_gpu:
    print("!! WARNING: no CUDA GPU detected -- SAM3 on CPU will be slow, "
          "and this multi-class script runs one pass PER class PER frame.")

print("Loading SAM3 video (tracking) model...")
video_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print("Model loaded on:", device)

torch.backends.cudnn.benchmark = True

# ---- MLflow ----
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow_run = mlflow.start_run(run_name=f"goods-simple-{int(time.time())}")
print(f"MLflow run started: {mlflow_run.info.run_id} (experiment: {MLFLOW_EXPERIMENT_NAME})")

mlflow.log_params({
    "video_model": "facebook/sam3",
    "device": str(device),
    "using_gpu": using_gpu,
    "infer_width": INFER_WIDTH,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "video_source": str(VIDEO_SOURCE),
    **{f"prompt_{code}": spec["prompt"] for code, spec in CARGO_CLASSES.items()},
})

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video source opened: {width}x{height}")

scale = INFER_WIDTH / width
infer_w, infer_h = INFER_WIDTH, int(height * scale)


def start_new_sessions():
    """One independent tracking session per cargo class, each with its own
    object-id namespace -- so a 'sack' id 3 and a 'box' id 3 are unrelated."""
    sessions = {}
    for code, spec in CARGO_CLASSES.items():
        session = video_processor.init_video_session(
            inference_device=device,
            dtype=torch.bfloat16,
        )
        session = video_processor.add_text_prompt(
            inference_session=session,
            text=spec["prompt"],
        )
        sessions[code] = session
    return sessions


inference_sessions = start_new_sessions()
print("Streaming session started. Press ESC to quit.")

# ============================================================
# Counting logic - NO line. As soon as SAM3 assigns a NEW tracking id to an
# item (one it hasn't seen before), that item is counted, once, forever.
# ============================================================

counted_ids = {code: set() for code in CARGO_CLASSES}   # ids already counted
total_count = {code: 0 for code in CARGO_CLASSES}        # running total per class


# ============================================================
# Threaded inference
# ============================================================

lock = threading.Lock()
latest_frame_bgr = None
latest_result = {code: {"boxes": np.zeros((0, 4)), "masks": [], "ids": np.zeros((0,), dtype=int),
                         "scores": np.zeros((0,))} for code in CARGO_CLASSES}
stop_flag = False


def inference_worker():
    global latest_result, inference_sessions
    infer_cycle = 0

    while not stop_flag:
        try:
            with lock:
                frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            if infer_cycle > 0 and infer_cycle % SESSION_RESET_EVERY_N_INFER == 0:
                print(f"[cycle {infer_cycle}] Resetting all tracking sessions to bound memory growth...")
                inference_sessions = start_new_sessions()
                torch.cuda.empty_cache()

            if infer_cycle % LOG_GPU_MEMORY_EVERY_N_INFER == 0 and using_gpu:
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                print(f"[cycle {infer_cycle}] GPU mem: {allocated:.0f}MB allocated / {reserved:.0f}MB reserved")

            small_bgr = cv2.resize(frame, (infer_w, infer_h))
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            inputs = video_processor(images=small_rgb, device=device, return_tensors="pt")

            new_result = {}
            for code, session in inference_sessions.items():
                with torch.inference_mode():
                    model_outputs = video_model(inference_session=session, frame=inputs.pixel_values[0])
                outputs = video_processor.postprocess_outputs(
                    session, model_outputs, original_sizes=inputs.original_sizes
                )

                boxes = np.zeros((0, 4))
                masks = []
                ids = np.zeros((0,), dtype=int)
                scores = np.zeros((0,))
                if outputs and len(outputs["object_ids"]) > 0:
                    raw_boxes = outputs["boxes"].cpu().numpy() / scale
                    raw_masks = outputs["masks"].cpu().numpy()
                    raw_ids = outputs["object_ids"].cpu().numpy()
                    raw_scores = outputs["scores"].cpu().numpy()

                    keep = raw_scores >= CONFIDENCE_THRESHOLD
                    boxes = raw_boxes[keep]
                    masks = raw_masks[keep]
                    ids = raw_ids[keep]
                    scores = raw_scores[keep]

                new_result[code] = {"boxes": boxes, "masks": masks, "ids": ids, "scores": scores}

            with lock:
                latest_result = new_result

            infer_cycle += 1

        except Exception:
            import traceback
            print("!! Inference worker crashed, attempting recovery:")
            traceback.print_exc()
            try:
                torch.cuda.empty_cache()
                inference_sessions = start_new_sessions()
                print("Recovered: started fresh tracking sessions.")
            except Exception:
                print("!! Recovery failed too. Stopping inference worker.")
                traceback.print_exc()
                break
            time.sleep(0.5)


worker_thread = threading.Thread(target=inference_worker, daemon=True)
worker_thread.start()

frame_idx = 0
fps_t0 = time.time()
fps_count = 0
fps_display = 0.0

try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("End of video / failed to read frame.")
            break

        with lock:
            latest_frame_bgr = frame_bgr
            result = latest_result

        overlay = frame_bgr.copy()

        for code, spec in CARGO_CLASSES.items():
            color = spec["color"]
            class_result = result[code]
            boxes = class_result["boxes"]
            masks = class_result["masks"]
            ids = class_result["ids"]
            scores = class_result["scores"]

            for i in range(len(ids)):
                obj_id = int(ids[i])
                confidence = float(scores[i]) if i < len(scores) else 0.0

                x1, y1, x2, y2 = boxes[i].astype(int)

                # New id for this class we haven't seen before -> count it, once.
                if obj_id not in counted_ids[code]:
                    counted_ids[code].add(obj_id)
                    total_count[code] += 1
                    print(f"Counted {code} #{total_count[code]} (object id {obj_id})")

                # Highlight: soft color fill over the exact segmented shape...
                if i < len(masks):
                    mask = cv2.resize(masks[i].astype(np.uint8), (width, height)) > 0
                    overlay[mask] = (overlay[mask] * (1 - MASK_ALPHA) + np.array(color) * MASK_ALPHA).astype(np.uint8)

                    # ...plus a crisp outline traced around that same shape.
                    mask_u8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, color, 2)
                else:
                    # Fallback if no mask came back for this detection: box outline only.
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                label = f"{code} #{obj_id} {confidence:.2f}"
                cv2.putText(overlay, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # ---- HUD: FPS + per-class counts ----
        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps_display = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

        cv2.putText(overlay, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        hud_y = 55
        grand_total = 0
        for code, spec in CARGO_CLASSES.items():
            count = total_count[code]
            grand_total += count
            cv2.putText(overlay, f"{code}: {count}", (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, spec["color"], 2)
            hud_y += 28
        cv2.putText(overlay, f"TOTAL: {grand_total}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # ---- MLflow periodic logging ----
        if frame_idx % MLFLOW_LOG_EVERY_N_FRAMES == 0:
            try:
                metrics_to_log = {"fps": fps_display, "grand_total_count": grand_total}
                for code in CARGO_CLASSES:
                    metrics_to_log[f"count_{code}"] = total_count[code]
                    metrics_to_log[f"tracked_now_{code}"] = len(result[code]["ids"])
                if using_gpu:
                    metrics_to_log["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    metrics_to_log["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
                mlflow.log_metrics(metrics_to_log, step=frame_idx)
            except Exception as e:
                print(f"MLflow logging failed (continuing anyway): {e}")

        cv2.imshow(WINDOW_NAME, overlay)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop_flag = True
    worker_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    grand_total = sum(total_count.values())
    print(f"Stopped. Frames processed: {frame_idx}.")
    print("Final counts:", total_count, "| Grand total:", grand_total)

    try:
        mlflow.log_metric("total_frames_processed", frame_idx)
        mlflow.log_metric("final_grand_total", grand_total)
        for code, count in total_count.items():
            mlflow.log_metric(f"final_count_{code}", count)
        mlflow.end_run()
        print("MLflow run closed. View results with: mlflow ui --workers 1")
    except Exception as e:
        print(f"MLflow cleanup failed: {e}")