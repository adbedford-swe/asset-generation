"""The start frame: where the camera stands, what it sees of the rest pose, and the points seeded on it.

The camera looks at the character from a three-quarter view, a little above, with room around it for
limbs that reach out and for jumps. Seeds are pixels on a grid over the character (away from its
outline, where tracks slide off), each remembered as a face and barycentric coordinates, so its
position on the posed mesh is known in every frame.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..splats import Camera, look_at


def start_camera(vertices: np.ndarray, width: int = 640, height: int = 640, yaw_deg: float = 35.0,
                 elevation_deg: float = 10.0, fov_deg: float = 30.0, margin: float = 1.6) -> Camera:
    """A camera on the rest pose (Y-up, facing +Z), its bounding sphere times ``margin`` filling the
    shorter side of the frame. ``fov_deg`` is the field of view across that side."""
    lo, hi = vertices.min(0), vertices.max(0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2 * margin
    distance = radius / math.sin(math.radians(fov_deg) / 2)
    yaw, el = math.radians(yaw_deg), math.radians(elevation_deg)
    eye = center + distance * np.array([math.sin(yaw) * math.cos(el), math.sin(el), math.cos(yaw) * math.cos(el)])
    f = 0.5 * min(width, height) / math.tan(math.radians(fov_deg) / 2)
    return Camera(look_at(torch.tensor(eye, dtype=torch.float32), torch.tensor(center, dtype=torch.float32)),
                  f, f, width / 2, height / 2, width, height)


def camera_from_dict(d: dict) -> Camera:
    return Camera(torch.tensor(d["view"], dtype=torch.float32), d["fx"], d["fy"], d["cx"], d["cy"],
                  int(d["width"]), int(d["height"]))


def project(points: torch.Tensor, cam: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel positions (..., 2) and depths (...) of world points (..., 3), in the rasterizer's
    convention (pixel (i, j) covers [j, j+1) x [i, i+1))."""
    view = cam.view.to(points)
    pc = points @ view[:3, :3].T + view[:3, 3]
    z = pc[..., 2]
    zs = z.clamp_min(1e-6)
    return torch.stack([cam.fx * pc[..., 0] / zs + cam.cx, cam.fy * pc[..., 1] / zs + cam.cy], -1), z


@dataclass
class Seeds:
    pixels: np.ndarray   # (N, 2) x, y at pixel centres
    faces: np.ndarray    # (N,) face index
    bary: np.ndarray     # (N, 3) barycentric coordinates in that face

    def save(self, path: Path) -> None:
        np.savez(path, pixels=self.pixels, faces=self.faces, bary=self.bary)

    @classmethod
    def load(cls, path: Path) -> "Seeds":
        d = np.load(path)
        return cls(d["pixels"], d["faces"], d["bary"])


def render_start(vertices: np.ndarray, faces: np.ndarray, cam: Camera, uvs=None, texture=None,
                 background=(1.0, 1.0, 1.0)):
    """The start frame (H, W, 3) in [0, 1], its mask, and each pixel's face and barycentric coordinates."""
    from ..preview import render_mesh

    rgb, mask, face, bary = render_mesh(vertices, faces, cam, uvs, texture, background=background,
                                       return_faces=True)
    return rgb.numpy(), mask.numpy() > 0.5, face.numpy(), bary.numpy()


def seed_points(mask: np.ndarray, face: np.ndarray, bary: np.ndarray, count: int = 1500,
                inset: int = 3) -> Seeds:
    """About ``count`` pixels on a regular grid over the character, at least ``inset`` pixels inside it."""
    from scipy.ndimage import binary_erosion

    inner = binary_erosion(mask, iterations=inset) if inset > 0 else mask
    if inner.sum() < count // 4:
        inner = mask
    area = int(inner.sum())
    step = max(1, int(round(math.sqrt(area / max(count, 1)))))
    ys, xs = np.nonzero(inner)
    on_grid = (ys % step == step // 2) & (xs % step == step // 2)
    ys, xs = ys[on_grid], xs[on_grid]
    return Seeds(np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float32), face[ys, xs].astype(np.int64),
                 bary[ys, xs].astype(np.float64))


def write_start(folder: Path, rgb: np.ndarray, cam: Camera, seeds: Seeds, mask: np.ndarray | None = None) -> None:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(folder / "start.png")
    if mask is not None:
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "start_mask.png")
    (folder / "camera.json").write_text(json.dumps(cam.to_dict(), indent=2))
    seeds.save(folder / "seeds.npz")
