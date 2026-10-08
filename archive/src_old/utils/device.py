"""
Device auto-detection: XPU (local Arc B580) → CUDA (Kaggle T4/P100) → CPU.
Import get_device() and get_amp_settings() everywhere instead of
hardcoding a device string -- the same training script then runs
unmodified on both machines.
"""
import torch


def get_device() -> torch.device:
    """XPU (local Arc B580) takes priority if present; falls back to CUDA
    (Kaggle T4/P100) or CPU. hasattr guards against older torch builds
    that don't ship the xpu module at all."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_amp_settings(device: torch.device):
    """Mixed-precision settings differ by hardware:
    - Arc B580 (xpu): bf16 autocast, no GradScaler needed
    - T4 / P100 (cuda): fp16 autocast + GradScaler
    - cpu: no autocast
    Returns (dtype_or_None, use_scaler: bool)
    """
    if device.type == "xpu":
        return torch.bfloat16, False
    if device.type == "cuda":
        return torch.float16, True
    return None, False
