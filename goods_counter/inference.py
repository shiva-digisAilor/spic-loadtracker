"""
Thread 2: inference -- ONLY runs SAM3 on whatever frame is freshest.
"""

import time
import traceback
import cv2
import numpy as np
import torch

from goods_counter.sessions import start_new_sessions


class SessionHolder:
    """Tiny mutable box for the current {class_code: session} dict, so the
    inference worker can reset all sessions in place without needing a
    `global` statement or returning state back out through a queue."""

    def __init__(self, sessions):
        self.sessions = sessions


def _run_one_class(video_model, video_processor, session, inputs, scale, confidence_threshold):
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

        keep = raw_scores >= confidence_threshold
        boxes = raw_boxes[keep]
        masks = raw_masks[keep]
        ids = raw_ids[keep]
        scores = raw_scores[keep]

    return {"boxes": boxes, "masks": masks, "ids": ids, "scores": scores}


def inference_worker(state, session_holder, video_model, video_processor, device, using_gpu,
                      cargo_classes, scale, infer_w, infer_h,
                      confidence_threshold, session_reset_every_n_infer,
                      log_gpu_memory_every_n_infer):
    infer_cycle = 0

    while not state.stop_flag:
        try:
            frame, _ = state.read_frame()

            if frame is None:
                if state.video_ended.is_set():
                    break
                time.sleep(0.005)
                continue

            if infer_cycle > 0 and infer_cycle % session_reset_every_n_infer == 0:
                print(f"[cycle {infer_cycle}] Resetting all tracking sessions to bound memory growth...")
                session_holder.sessions = start_new_sessions(video_processor, device, cargo_classes)
                torch.cuda.empty_cache()

            if infer_cycle % log_gpu_memory_every_n_infer == 0 and using_gpu:
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                print(f"[cycle {infer_cycle}] GPU mem: {allocated:.0f}MB allocated / {reserved:.0f}MB reserved")

            small_bgr = cv2.resize(frame, (infer_w, infer_h))
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            inputs = video_processor(images=small_rgb, device=device, return_tensors="pt")

            new_result = {}
            for code, session in session_holder.sessions.items():
                new_result[code] = _run_one_class(
                    video_model, video_processor, session, inputs, scale, confidence_threshold
                )

            state.write_result(new_result)
            infer_cycle += 1

        except Exception:
            print("!! Inference worker crashed, attempting recovery:")
            traceback.print_exc()
            try:
                torch.cuda.empty_cache()
                session_holder.sessions = start_new_sessions(video_processor, device, cargo_classes)
                print("Recovered: started fresh tracking sessions.")
            except Exception:
                print("!! Recovery failed too. Stopping inference worker.")
                traceback.print_exc()
                break
            time.sleep(0.5)

    print("Inference worker stopped.")
