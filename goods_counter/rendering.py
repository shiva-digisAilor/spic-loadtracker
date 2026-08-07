"""
Thread 3 (main): ONLY renders -- draws overlays, imshow, waitKey,
MLflow logging. Never touches cap.read() or the model.
"""

import time
import cv2
import numpy as np

from goods_counter import mlflow_tracking


def _mask_centroid(mask):
    """Center-of-mass of a boolean mask, in (x, y) pixel coords. Falls back
    to None if the mask is somehow empty (shouldn't normally happen)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def _draw_id_badge(overlay, center, obj_id, color):
    """Draws a small, compact id badge (filled circle + centered number) at
    `center`. Kept deliberately tiny so it sits inside a mask without
    covering much of the object or colliding with a neighboring object's
    badge."""
    cx, cy = center
    text = str(obj_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    radius = max(text_w, text_h) // 2 + 6

    # Filled circle in the class color, thin white ring for contrast against
    # any background color/mask tint.
    cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)

    text_x = cx - text_w // 2
    text_y = cy + text_h // 2
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detections(overlay, width, height, cargo_classes, result, counter, mask_alpha):
    """Draws masks/boxes for every detected object, updates the counter's
    grand total (running max of how many were ever visible at once, see
    counting.py), and marks each object with a small id badge placed at
    the CENTER of its own mask -- purely for visual reference, the badge
    number is NOT what drives the grand total anymore. Mutates `overlay`
    in place."""
    for code, spec in cargo_classes.items():
        color = spec["color"]
        class_result = result[code]
        boxes = class_result["boxes"]
        masks = class_result["masks"]
        ids = class_result["ids"]

        # Grand total = running max of "how many were visible at once" --
        # decided once per class per frame, independent of individual ids.
        counter.update(code, len(ids))

        for i in range(len(ids)):
            obj_id = int(ids[i])
            x1, y1, x2, y2 = boxes[i].astype(int)

            badge_center = None

            if i < len(masks):
                mask = cv2.resize(masks[i].astype(np.uint8), (width, height)) > 0
                overlay[mask] = (overlay[mask] * (1 - mask_alpha) + np.array(color) * mask_alpha).astype(np.uint8)
                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                badge_center = _mask_centroid(mask)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            if badge_center is None:
                # No mask (or an empty one) -- fall back to the box's center.
                badge_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            _draw_id_badge(overlay, badge_center, obj_id, color)


def draw_hud(overlay, cargo_classes, counter, result, render_fps_display, capture_fps_display):
    """HUD shows, per class: currently visible / total ever counted.
    e.g. "box: 0/1" -- one box has been counted so far, but zero are in
    frame right now. Starts at "0/0" before anything's been seen."""
    cv2.putText(overlay, f"Render FPS: {render_fps_display:.1f}  Capture FPS: {capture_fps_display:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    hud_y = 55
    currently_visible_total = 0
    for code, spec in cargo_classes.items():
        visible_now = len(result[code]["ids"])
        total_ever = counter.total_count[code]
        currently_visible_total += visible_now
        cv2.putText(overlay, f"{code}: {visible_now}/{total_ever}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, spec["color"], 2)
        hud_y += 28
    cv2.putText(overlay, f"TOTAL: {currently_visible_total}/{counter.grand_total()}", (10, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def render_loop(state, counter, width, height, cargo_classes, mask_alpha,
                 window_name, mlflow_log_every_n_frames, using_gpu, device):
    frame_idx = 0
    render_fps_t0 = time.time()
    render_fps_count = 0
    render_fps_display = 0.0
    last_rendered_idx = -1

    while True:
        frame_bgr, captured_idx = state.read_frame()

        if frame_bgr is None:
            if state.video_ended.is_set():
                break
            time.sleep(0.005)
            continue

        if captured_idx == last_rendered_idx:
            # No new frame yet -- don't burn CPU re-drawing the same image,
            # but stay responsive to ESC and to end-of-stream.
            if state.video_ended.is_set():
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.002)
            continue
        last_rendered_idx = captured_idx

        result = state.read_result()
        overlay = frame_bgr.copy()

        draw_detections(overlay, width, height, cargo_classes, result, counter, mask_alpha)

        render_fps_count += 1
        if time.time() - render_fps_t0 >= 1.0:
            render_fps_display = render_fps_count / (time.time() - render_fps_t0)
            render_fps_count = 0
            render_fps_t0 = time.time()

        draw_hud(overlay, cargo_classes, counter, result, render_fps_display, state.capture_fps_display)

        if frame_idx % mlflow_log_every_n_frames == 0:
            mlflow_tracking.log_frame_metrics(
                frame_idx, cargo_classes, counter, result,
                render_fps_display, state.capture_fps_display, using_gpu, device,
            )

        cv2.imshow(window_name, overlay)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    return frame_idx