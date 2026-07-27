import cv2
import time
import torch
import threading
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from accelerate import Accelerator
from transformers import Sam3VideoProcessor

# ============================================================
# CONFIG
# ============================================================

VIDEO_SOURCE = 0  # 0 = webcam. Set to a file path string to use a video file instead.

MODEL_REGISTRY_NAME = "sam3-video-tracking-model"  # must match register_model.py

# Categories kept deliberately distinct in shape so SAM3 
CARGO_CLASSES = {
    "sack": {
        "prompt": "a cargo sack, woven sack, fabric bag, or filled storage bag",
        "color": (0, 165, 255),
    },
    "box": {
        "prompt": "a cardboard box, carton, shipping box, or package",
        "color": (255, 100, 0),
    },
    "crate": {
        "prompt": "a wooden crate, plastic crate, storage crate, or cargo crate",
        "color": (255, 0, 255),
    },
}

# INFER_WIDTH lowered 384 -> 256: on a 6GB laptop GPU (e.g. RTX 4050) this is
# close to a 2x cut on the vision backbone's cost, on top of the 3x cut from
# single-session batching below. Bump back up only if detection quality
# visibly suffers.
INFER_WIDTH = 256
CONFIDENCE_THRESHOLD = 0.5
MASK_ALPHA = 0.35

# Lowered 1500 -> 500: with only 6GB VRAM, recycling tracking sessions more
# often keeps memory growth in check and avoids CUDA OOM / stutters on long
# runs.
SESSION_RESET_EVERY_N_INFER = 500
LOG_GPU_MEMORY_EVERY_N_INFER = 200

MLFLOW_EXPERIMENT_NAME = "goods-count-from-registry"
MLFLOW_LOG_EVERY_N_FRAMES = 30

WINDOW_NAME = "SAM3 Goods Counter (from Registry)"

# ============================================================
# Setup
# ============================================================

accelerator = Accelerator()
device = accelerator.device
using_gpu = (device.type == "cuda") if hasattr(device, "type") else ("cuda" in str(device))
if not using_gpu:
    print("!! WARNING: no CUDA GPU detected -- this will be slow.")

# ---- Load the model FROM the MLflow Model Registry ----
print(f"Looking up latest registered version of '{MODEL_REGISTRY_NAME}'...")
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
if not versions:
    raise RuntimeError(
        f"No registered versions found for '{MODEL_REGISTRY_NAME}'. "
        f"Run register_model.py first (see that file's instructions)."
    )
latest_version = max(int(v.version) for v in versions)
model_uri = f"models:/{MODEL_REGISTRY_NAME}/{latest_version}"
print(f"Loading model from MLflow registry: {model_uri}")
print("(This loads from MLflow's local artifact store, not HuggingFace.)")

video_model = mlflow.pytorch.load_model(model_uri)
video_model = video_model.to(device, dtype=torch.bfloat16)
video_model.eval()
print("Model loaded from registry onto:", device)

# The processor (tokenizer/preprocessing config) is lightweight and NOT part
# of the registered weights -- it still loads from HuggingFace, same as always.
video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

torch.backends.cudnn.benchmark = True

# ---- MLflow run for THIS session's metrics (separate from model registry) ----
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow_run = mlflow.start_run(run_name=f"goods-registry-{int(time.time())}")
print(f"MLflow run started: {mlflow_run.info.run_id} (experiment: {MLFLOW_EXPERIMENT_NAME})")

mlflow.log_params({
    "model_source": "mlflow_registry",
    "model_registry_name": MODEL_REGISTRY_NAME,
    "model_registry_version": latest_version,
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
    """
    Builds ONE shared SAM3 video session with all cargo-class prompts
    attached to it (one add_text_prompt() call per class). This replaces
    the old per-class session dict -- inference_worker() now runs a single
    model forward pass per frame instead of one per class.
    """
    session = video_processor.init_video_session(
        inference_device=device,
        dtype=torch.bfloat16,
    )
    for spec in CARGO_CLASSES.values():
        session = video_processor.add_text_prompt(
            inference_session=session,
            text=spec["prompt"],
        )
    return session


inference_sessions = start_new_sessions()  # a single shared session, despite the plural name (kept for diff-friendliness)
print("Streaming session started. Press ESC to quit.")

# ============================================================
# Counting logic - no line. New tracked id -> count it, once.
# ============================================================

counted_ids = {code: set() for code in CARGO_CLASSES}
total_count = {code: 0 for code in CARGO_CLASSES}


@mlflow.trace(name="frame_summary")
def log_frame_trace(frame_idx, counts, fps, objects_visible):
    """Wrapping this in @mlflow.trace records a trace every time it's
    called -- visible under MLflow's Traces tab, showing these inputs and
    the returned dict as the trace's output."""
    return {
        "frame_idx": frame_idx,
        "counts": dict(counts),
        "fps": round(fps, 2),
        "objects_currently_visible": objects_visible,
    }

# ============================================================
# Shared, lock-protected state between the three threads
# ============================================================

frame_lock = threading.Lock()          # guards latest_frame_bgr / latest_frame_idx
result_lock = threading.Lock()         # guards latest_result

latest_frame_bgr = None
latest_frame_idx = -1                  # monotonically increasing capture counter

latest_result = {code: {"boxes": np.zeros((0, 4)), "masks": [], "ids": np.zeros((0,), dtype=int),
                         "scores": np.zeros((0,))} for code in CARGO_CLASSES}

video_ended = threading.Event()        # set by capture_thread when the source runs out
stop_flag = False                      # set by main thread to shut everything down

capture_fps_display = 0.0


# ============================================================
# Thread 1: capture -- ONLY reads frames. Nothing else touches cap.read().
# ============================================================

def capture_worker():
    global latest_frame_bgr, latest_frame_idx, capture_fps_display

    local_idx = 0
    fps_t0 = time.time()
    fps_count = 0

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Capture worker: end of video / failed to read frame.")
            video_ended.set()
            break

        with frame_lock:
            latest_frame_bgr = frame
            latest_frame_idx = local_idx

        local_idx += 1
        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            capture_fps_display = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

    print("Capture worker stopped.")


# ============================================================
# Thread 2: inference -- ONLY runs SAM3 on whatever frame is freshest.
# One forward pass per frame (single shared multi-prompt session), then the
# combined results are split back into per-class buckets using
# outputs["prompt_to_obj_ids"].
# ============================================================

def inference_worker():
    global latest_result, inference_sessions
    infer_cycle = 0

    while not stop_flag:
        try:
            with frame_lock:
                frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()

            if frame is None:
                if video_ended.is_set():
                    break
                time.sleep(0.005)
                continue

            if infer_cycle > 0 and infer_cycle % SESSION_RESET_EVERY_N_INFER == 0:
                print(f"[cycle {infer_cycle}] Resetting tracking session to bound memory growth...")
                inference_sessions = start_new_sessions()
                torch.cuda.empty_cache()

            if infer_cycle % LOG_GPU_MEMORY_EVERY_N_INFER == 0 and using_gpu:
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                print(f"[cycle {infer_cycle}] GPU mem: {allocated:.0f}MB allocated / {reserved:.0f}MB reserved")

            small_bgr = cv2.resize(frame, (infer_w, infer_h))
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            inputs = video_processor(images=small_rgb, device=device, return_tensors="pt")

            # -- ONE forward pass for every class, via the shared session --
            with torch.inference_mode():
                model_outputs = video_model(inference_session=inference_sessions, frame=inputs.pixel_values[0])
            outputs = video_processor.postprocess_outputs(
                inference_sessions, model_outputs, original_sizes=inputs.original_sizes
            )

            new_result = {code: {"boxes": np.zeros((0, 4)), "masks": [], "ids": np.zeros((0,), dtype=int),
                                  "scores": np.zeros((0,))} for code in CARGO_CLASSES}

            if outputs and len(outputs.get("object_ids", [])) > 0:
                object_ids = outputs["object_ids"].cpu().numpy()
                raw_boxes = outputs["boxes"].cpu().numpy() / scale
                raw_masks = outputs["masks"].cpu().numpy()
                raw_scores = outputs["scores"].cpu().numpy()
                prompt_to_obj_ids = outputs.get("prompt_to_obj_ids", {})
                index_by_id = {int(oid): i for i, oid in enumerate(object_ids)}

                for code, spec in CARGO_CLASSES.items():
                    class_obj_ids = prompt_to_obj_ids.get(spec["prompt"], [])
                    idxs = [index_by_id[int(o)] for o in class_obj_ids if int(o) in index_by_id]
                    if not idxs:
                        continue

                    idx = np.array(idxs)
                    cls_scores = raw_scores[idx]
                    keep = cls_scores >= CONFIDENCE_THRESHOLD

                    new_result[code] = {
                        "boxes": raw_boxes[idx][keep],
                        "masks": raw_masks[idx][keep],
                        "ids": object_ids[idx][keep].astype(int),
                        "scores": cls_scores[keep],
                    }

            with result_lock:
                latest_result = new_result

            infer_cycle += 1

        except Exception:
            import traceback
            print("!! Inference worker crashed, attempting recovery:")
            traceback.print_exc()
            try:
                torch.cuda.empty_cache()
                inference_sessions = start_new_sessions()
                print("Recovered: started fresh tracking session.")
            except Exception:
                print("!! Recovery failed too. Stopping inference worker.")
                traceback.print_exc()
                break
            time.sleep(0.5)

    print("Inference worker stopped.")


capture_thread = threading.Thread(target=capture_worker, daemon=True)
worker_thread = threading.Thread(target=inference_worker, daemon=True)
capture_thread.start()
worker_thread.start()

# ============================================================
# Thread 3 (main): ONLY renders -- draws overlays, imshow, waitKey,
# MLflow logging. Never touches cap.read() or the model.
# ============================================================

frame_idx = 0
render_fps_t0 = time.time()
render_fps_count = 0
render_fps_display = 0.0
last_rendered_idx = -1

try:
    while True:
        with frame_lock:
            frame_bgr = None if latest_frame_bgr is None else latest_frame_bgr.copy()
            captured_idx = latest_frame_idx
        with result_lock:
            result = latest_result

        if frame_bgr is None:
            if video_ended.is_set():
                break
            time.sleep(0.005)
            continue

        if captured_idx == last_rendered_idx:
            # No new frame yet -- don't burn CPU re-drawing the same image,
            # but stay responsive to ESC and to end-of-stream.
            if video_ended.is_set():
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.002)
            continue
        last_rendered_idx = captured_idx

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

                if obj_id not in counted_ids[code]:
                    counted_ids[code].add(obj_id)
                    total_count[code] += 1
                    print(f"Counted {code} #{total_count[code]} (object id {obj_id})")

                if i < len(masks):
                    mask = cv2.resize(masks[i].astype(np.uint8), (width, height)) > 0
                    overlay[mask] = (overlay[mask] * (1 - MASK_ALPHA) + np.array(color) * MASK_ALPHA).astype(np.uint8)
                    mask_u8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, color, 2)
                else:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                label = f"{code} #{obj_id} {confidence:.2f}"
                cv2.putText(overlay, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        render_fps_count += 1
        if time.time() - render_fps_t0 >= 1.0:
            render_fps_display = render_fps_count / (time.time() - render_fps_t0)
            render_fps_count = 0
            render_fps_t0 = time.time()

        cv2.putText(overlay, f"Render FPS: {render_fps_display:.1f}  Capture FPS: {capture_fps_display:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

        if frame_idx % MLFLOW_LOG_EVERY_N_FRAMES == 0:
            try:
                log_frame_trace(frame_idx, total_count, render_fps_display,
                                 sum(len(result[c]["ids"]) for c in CARGO_CLASSES))

                metrics_to_log = {
                    "render_fps": render_fps_display,
                    "capture_fps": capture_fps_display,
                    "grand_total_count": grand_total,
                }
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
    capture_thread.join(timeout=2)
    worker_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    grand_total = sum(total_count.values())
    print(f"Stopped. Frames rendered: {frame_idx}.")
    print("Final counts:", total_count, "| Grand total:", grand_total)

    try:
        mlflow.log_metric("total_frames_rendered", frame_idx)
        mlflow.log_metric("final_grand_total", grand_total)
        for code, count in total_count.items():
            mlflow.log_metric(f"final_count_{code}", count)
        mlflow.end_run()
        print("MLflow run closed. View results with: mlflow ui --workers 1")
    except Exception as e:
        print(f"MLflow cleanup failed: {e}")