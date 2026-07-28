"""
The single object that lets the three threads (capture, inference, render)
talk to each other without stepping on each other's toes.

Design mirrors the original script: each stage always reads/writes the
*latest* frame or result under a short-held lock, so a slow stage never
blocks the others -- it just causes its consumer to reuse/skip stale data.
"""

import threading
import numpy as np


def _empty_class_result():
    return {
        "boxes": np.zeros((0, 4)),
        "masks": [],
        "ids": np.zeros((0,), dtype=int),
        "scores": np.zeros((0,)),
    }


class SharedState:
    def __init__(self, cargo_classes):
        self.frame_lock = threading.Lock()   # guards latest_frame_bgr / latest_frame_idx
        self.result_lock = threading.Lock()  # guards latest_result

        self.latest_frame_bgr = None
        self.latest_frame_idx = -1  # monotonically increasing capture counter

        self.latest_result = {code: _empty_class_result() for code in cargo_classes}

        self.video_ended = threading.Event()  # set by capture thread when source runs out
        self.stop_flag = False                # set by main thread to shut everything down

        self.capture_fps_display = 0.0

    # ---- frame access ----

    def write_frame(self, frame, idx):
        with self.frame_lock:
            self.latest_frame_bgr = frame
            self.latest_frame_idx = idx

    def read_frame(self):
        """Returns (frame_copy_or_None, idx)."""
        with self.frame_lock:
            frame = None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy()
            idx = self.latest_frame_idx
        return frame, idx

    # ---- result access ----

    def write_result(self, result):
        with self.result_lock:
            self.latest_result = result

    def read_result(self):
        with self.result_lock:
            return self.latest_result
