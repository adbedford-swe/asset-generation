"""Relative depth for every frame, sampled where the tracks are.

Depth Anything V2 (Small, Apache-2.0) estimates each frame's relative inverse depth. Its scale and
offset drift from frame to frame, so the fit only asks that the posed mesh's inverse depth at the
tracked points matches it up to a per-frame scale and offset: which parts are nearer than which.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def estimate(frames: np.ndarray, device: str = "cuda", batch: int = 8, log=print) -> np.ndarray:
    """Relative inverse depth (T, H, W) of ``frames`` (T, H, W, 3) uint8; larger is nearer."""
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    log(f"[capture] estimating depth for {len(frames)} frames ({MODEL}) on {dev}")
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval()
    H, W = frames.shape[1:3]
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            images = [Image.fromarray(f) for f in frames[i:i + batch]]
            inputs = processor(images=images, return_tensors="pt").to(dev)
            pred = model(**inputs).predicted_depth                      # (B, h, w)
            pred = torch.nn.functional.interpolate(pred[:, None], size=(H, W), mode="bilinear",
                                                   align_corners=False)[:, 0]
            out.append(pred.float().cpu().numpy())
    return np.concatenate(out)


def sample(depth: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Bilinear samples (T, N) of ``depth`` (T, H, W) at track positions (T, N, 2), pixel centres at +0.5."""
    T, H, W = depth.shape
    d = torch.as_tensor(depth)[:, None]
    xy = torch.as_tensor(tracks, dtype=torch.float32)
    grid = torch.stack([xy[..., 0] / W * 2 - 1, xy[..., 1] / H * 2 - 1], -1)[:, None]   # (T, 1, N, 2)
    return torch.nn.functional.grid_sample(d, grid, mode="bilinear", padding_mode="border",
                                           align_corners=False)[:, 0, 0].numpy()


def affine_residual(inv_depth: torch.Tensor, observed: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per frame, how far ``inv_depth`` (T, N) is from the best scale and offset of ``observed`` (T, N),
    as a share of its own spread: 0 when they agree on which points are nearer and by how much."""
    w = weight / weight.sum(-1, keepdim=True).clamp_min(1e-9)
    mo, mi = (w * observed).sum(-1, keepdim=True), (w * inv_depth).sum(-1, keepdim=True)
    co, ci = observed - mo, inv_depth - mi
    var_o = (w * co * co).sum(-1, keepdim=True).clamp_min(1e-12)
    scale = ((w * co * ci).sum(-1, keepdim=True) / var_o).clamp_min(0)
    resid = ci - scale * co
    var_i = (w * ci * ci).sum(-1).clamp_min(1e-12)
    return (w * resid * resid).sum(-1) / var_i.detach()
