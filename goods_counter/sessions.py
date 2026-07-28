"""
Owns the lifecycle of SAM3 video-tracking sessions -- one per cargo class.
Kept separate because sessions get periodically reset (to bound memory
growth) independently of everything else in the inference loop.
"""

import torch


def start_new_sessions(video_processor, device, cargo_classes):
    """Creates one fresh tracking session per cargo class, each seeded with
    its text prompt. Returns {class_code: session}."""
    sessions = {}
    for code, spec in cargo_classes.items():
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
