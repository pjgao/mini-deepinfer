"""Portable eager backend. No CUDA-to-NPU monkey-patching or custom kernels."""

import torch


def resolve_device(name: str) -> torch.device:
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU needs a matching torch_npu/CANN installation.") from exc
        if not torch.npu.is_available():
            raise RuntimeError("torch_npu is installed, but no NPU is available.")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda", "npu"}:
        raise ValueError("Choose cpu, cuda[:index], or npu[:index].")
    return device


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)
