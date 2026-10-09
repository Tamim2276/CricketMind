"""Pick the GPU, and the right precision for it.

Two machines: Arc B580 (XPU) here, RTX 3070 or Kaggle T4 elsewhere. Hardcoding
"cuda" breaks one of them, so nothing in this project names a backend.
"""
from typing import NamedTuple, Optional

import torch

__all__ = ["get_device", "get_amp_settings", "AmpSettings", "describe"]


def get_device() -> torch.device:
    """Best available: XPU, then CUDA, then CPU."""
    # a torch built without Intel support has no .xpu attribute at all
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


class AmpSettings(NamedTuple):
    dtype: Optional[torch.dtype]   # None = no autocast
    use_scaler: bool
    device_type: str               # what torch.autocast() wants


def get_amp_settings(device: torch.device) -> AmpSettings:
    """Autocast dtype, and whether a GradScaler is needed.

    bf16 keeps fp32's range, so gradients can't underflow and no scaler is
    needed. fp16 can't, and does. CUDA stays on fp16 because the Kaggle T4 has
    no real bf16 support.
    """
    if device.type == "xpu":
        return AmpSettings(torch.bfloat16, False, "xpu")   # a scaler errors here
    if device.type == "cuda":
        return AmpSettings(torch.float16, True, "cuda")
    return AmpSettings(None, False, "cpu")


def describe(device: Optional[torch.device] = None) -> str:
    """One line for the top of a training log."""
    device = device or get_device()
    amp = get_amp_settings(device)
    if amp.dtype is None:
        dtype = "off"
    else:
        dtype = str(amp.dtype).replace("torch.", "")

    if amp.use_scaler:
        scaler = "on"
    else:
        scaler = "off"

    name = "CPU"
    if device.type == "xpu":
        name = torch.xpu.get_device_name(device.index or 0)
    elif device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)

    return (f"device={device} ({name})  autocast={dtype}  "
            f"grad_scaler={scaler}  torch={torch.__version__}")


if __name__ == "__main__":
    print(describe())
