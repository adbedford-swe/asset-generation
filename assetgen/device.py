"""Make a stage's device the current one, so code that asks for plain "cuda" lands on it too."""
from __future__ import annotations


def use_device(device: str) -> None:
    import torch

    if device.startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
