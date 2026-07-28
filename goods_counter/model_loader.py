"""
Everything related to getting the model, its device, and its processor
ready to run. Nothing here touches the camera, threads, or MLflow *runs*
(only the MLflow *registry*, which is a different concern).
"""

import torch
from accelerate import Accelerator
from mlflow import MlflowClient
import mlflow.pytorch
from transformers import Sam3VideoProcessor


def setup_accelerator():
    """Returns (accelerator, device, using_gpu)."""
    accelerator = Accelerator()
    device = accelerator.device
    using_gpu = (device.type == "cuda") if hasattr(device, "type") else ("cuda" in str(device))
    if not using_gpu:
        print("!! WARNING: no CUDA GPU detected -- this will be slow.")
    return accelerator, device, using_gpu


def load_model_from_registry(model_registry_name: str, device):
    """
    Looks up the latest registered version of `model_registry_name` in the
    MLflow Model Registry, loads it, and moves it to `device` in bfloat16.

    Returns (video_model, latest_version).
    """
    print(f"Looking up latest registered version of '{model_registry_name}'...")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_registry_name}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions found for '{model_registry_name}'. "
            f"Run register_model.py first (see that file's instructions)."
        )
    latest_version = max(int(v.version) for v in versions)
    model_uri = f"models:/{model_registry_name}/{latest_version}"
    print(f"Loading model from MLflow registry: {model_uri}")
    print("(This loads from MLflow's local artifact store, not HuggingFace.)")

    video_model = mlflow.pytorch.load_model(model_uri)
    video_model = video_model.to(device, dtype=torch.bfloat16)
    video_model.eval()
    print("Model loaded from registry onto:", device)

    return video_model, latest_version


def load_processor():
    """
    The processor (tokenizer/preprocessing config) is lightweight and NOT
    part of the registered weights -- it still loads from HuggingFace, same
    as always.
    """
    return Sam3VideoProcessor.from_pretrained("facebook/sam3")
