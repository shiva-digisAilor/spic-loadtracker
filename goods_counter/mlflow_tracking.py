"""
Everything that talks to MLflow for *this session's* run (params, metrics,
traces, run lifecycle). Separate from model_loader.py, which only talks to
the MLflow *Model Registry*.
"""

import time
import torch
import mlflow


def start_run(experiment_name):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=f"goods-registry-{int(time.time())}")
    print(f"MLflow run started: {run.info.run_id} (experiment: {experiment_name})")
    return run


def log_run_params(model_registry_name, latest_version, device, using_gpu,
                    infer_width, confidence_threshold, video_source, cargo_classes):
    mlflow.log_params({
        "model_source": "mlflow_registry",
        "model_registry_name": model_registry_name,
        "model_registry_version": latest_version,
        "device": str(device),
        "using_gpu": using_gpu,
        "infer_width": infer_width,
        "confidence_threshold": confidence_threshold,
        "video_source": str(video_source),
        **{f"prompt_{code}": spec["prompt"] for code, spec in cargo_classes.items()},
    })


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


def log_frame_metrics(frame_idx, cargo_classes, counter, result,
                       render_fps_display, capture_fps_display, using_gpu, device):
    """Logs one @mlflow.trace call plus a batch of scalar metrics for the
    current frame. Swallows/reports its own errors so a logging hiccup
    never takes down the render loop."""
    try:
        grand_total = counter.grand_total()
        log_frame_trace(
            frame_idx, counter.total_count, render_fps_display,
            sum(len(result[c]["ids"]) for c in cargo_classes),
        )

        metrics_to_log = {
            "render_fps": render_fps_display,
            "capture_fps": capture_fps_display,
            "grand_total_count": grand_total,
        }
        for code in cargo_classes:
            metrics_to_log[f"count_{code}"] = counter.total_count[code]
            metrics_to_log[f"tracked_now_{code}"] = len(result[code]["ids"])
        if using_gpu:
            metrics_to_log["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            metrics_to_log["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        mlflow.log_metrics(metrics_to_log, step=frame_idx)
    except Exception as e:
        print(f"MLflow logging failed (continuing anyway): {e}")


def log_final_summary(frame_idx, counter):
    try:
        mlflow.log_metric("total_frames_rendered", frame_idx)
        mlflow.log_metric("final_grand_total", counter.grand_total())
        for code, count in counter.total_count.items():
            mlflow.log_metric(f"final_count_{code}", count)
        mlflow.end_run()
        print("MLflow run closed. View results with: mlflow ui --workers 1")
    except Exception as e:
        print(f"MLflow cleanup failed: {e}")
