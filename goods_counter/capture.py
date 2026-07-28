"""
Thread 1: capture -- ONLY reads frames. Nothing else touches cap.read().
"""

import time


def capture_worker(cap, state):
    local_idx = 0
    fps_t0 = time.time()
    fps_count = 0

    while not state.stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Capture worker: end of video / failed to read frame.")
            state.video_ended.set()
            break

        state.write_frame(frame, local_idx)

        local_idx += 1
        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            state.capture_fps_display = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

    print("Capture worker stopped.")
