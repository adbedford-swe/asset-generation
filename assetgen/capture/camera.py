"""The start frame: where the camera stands, what it sees of the rest pose, and the points seeded on it.

The camera looks at the character from a three-quarter view, a little above, with room around it for
limbs that reach out and for jumps. Seeds are pixels on a grid over the character (away from its
outline, where tracks slide off), each remembered as a face and barycentric coordinates, so its
position on the posed mesh is known in every frame. Each seed also gets a trust: how much texture
surrounds it. A point inside a patch of flat colour can slide across it without the picture changing,
so no tracker can hold it there; corners and detail it can.
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
                 elevation_deg: float = 10.0, fov_deg: float = 30.0, fill: float = 0.6) -> Camera:
    """A camera on the rest pose (Y-up, facing +Z), ``yaw_deg`` round from its front and ``elevation_deg``
    above, aimed so the character's silhouette is centred and spans ``fill`` of the frame's width or
    height, whichever it reaches first; the rest is room to move. ``fov_deg`` is the field of view across
    the frame's shorter side.

    The silhouette, not a bounding sphere: tracking and the video model both work with the pixels the
    character covers, and a long body's sphere would leave most of the frame empty."""
    vertices = np.asarray(vertices, dtype=np.float64)
    lo, hi = vertices.min(0), vertices.max(0)
    target = (lo + hi) / 2
    yaw, el = math.radians(yaw_deg), math.radians(elevation_deg)
    toward = np.array([math.sin(yaw) * math.cos(el), math.sin(el), math.cos(yaw) * math.cos(el)])
    distance = float(np.linalg.norm(hi - lo)) / 2 / math.sin(math.radians(fov_deg) / 2)
    f = 0.5 * min(width, height) / math.tan(math.radians(fov_deg) / 2)
    points = torch.as_tensor(vertices, dtype=torch.float32)

    def aimed(target, distance):
        eye = torch.tensor(target + distance * toward, dtype=torch.float32)
        return Camera(look_at(eye, torch.tensor(target, dtype=torch.float32)), f, f, width / 2, height / 2,
                      width, height)

    for _ in range(4):  # projected size goes as 1 / distance, near enough to settle in a few steps
        cam = aimed(target, distance)
        xy, z = project(points, cam)
        xy = xy.double().numpy()
        span = xy.max(0) - xy.min(0)
        distance *= max(span[0] / (fill * width), span[1] / (fill * height))
        # Shift the aim so the silhouette's centre lands on the image centre, at the character's depth.
        offset = (xy.max(0) + xy.min(0)) / 2 - np.array([width / 2, height / 2])
        depth = float(z.double().mean())
        R = cam.view[:3, :3].double().numpy()
        target = target + (offset[0] * R[0] + offset[1] * R[1]) * depth / f
    return aimed(target, distance)


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
    trust: np.ndarray | None = None  # (N,) in (0, 1]: how well a tracker can hold the point (see texture_trust)

    def weights(self) -> np.ndarray:
        return np.ones(len(self.pixels)) if self.trust is None else self.trust

    def save(self, path: Path) -> None:
        np.savez(path, pixels=self.pixels, faces=self.faces, bary=self.bary, trust=self.weights())

    @classmethod
    def load(cls, path: Path) -> "Seeds":
        d = np.load(path)
        return cls(d["pixels"], d["faces"], d["bary"], d["trust"] if "trust" in d else None)


def render_start(vertices: np.ndarray, faces: np.ndarray, cam: Camera, uvs=None, texture=None,
                 background=(1.0, 1.0, 1.0)):
    """The start frame (H, W, 3) in [0, 1], its mask, and each pixel's face and barycentric coordinates."""
    from ..preview import render_mesh

    rgb, mask, face, bary = render_mesh(vertices, faces, cam, uvs, texture, background=background,
                                       return_faces=True)
    return rgb.numpy(), mask.numpy() > 0.5, face.numpy(), bary.numpy()


def texture_trust(rgb: np.ndarray, pixels: np.ndarray, window: int = 9, floor: float = 0.05) -> np.ndarray:
    """Per pixel, how much texture surrounds it: the smaller eigenvalue of the colour structure tensor over a
    ``window`` (Shi and Tomasi's measure of a good feature to track), relative to the seeds' upper quartile
    and clipped to [floor, 1]. Low inside flat colour, where a track can only drift; high at corners and
    detail."""
    from scipy.ndimage import uniform_filter

    img = np.asarray(rgb, dtype=np.float64)
    gy, gx = np.gradient(img, axis=(0, 1))
    a = uniform_filter((gx * gx).sum(-1), window)
    b = uniform_filter((gx * gy).sum(-1), window)
    c = uniform_filter((gy * gy).sum(-1), window)
    smallest = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b * b)
    xs, ys = np.floor(pixels).astype(int).T
    score = np.maximum(smallest[ys, xs], 0)
    top = np.percentile(score, 75) if len(score) else 0
    return np.clip(score / top, floor, 1.0) if top > 0 else np.ones(len(score))


def seed_points(mask: np.ndarray, face: np.ndarray, bary: np.ndarray, count: int = 1500,
                inset: int = 3, rgb: np.ndarray | None = None) -> Seeds:
    """About ``count`` pixels on a regular grid over the character, at least ``inset`` pixels inside it,
    with their :func:`texture_trust` when the start frame ``rgb`` is given."""
    from scipy.ndimage import binary_erosion

    inner = binary_erosion(mask, iterations=inset) if inset > 0 else mask
    if inner.sum() < count // 4:
        inner = mask
    area = int(inner.sum())
    step = max(1, int(round(math.sqrt(area / max(count, 1)))))
    ys, xs = np.nonzero(inner)
    on_grid = (ys % step == step // 2) & (xs % step == step // 2)
    ys, xs = ys[on_grid], xs[on_grid]
    pixels = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float32)
    return Seeds(pixels, face[ys, xs].astype(np.int64), bary[ys, xs].astype(np.float64),
                 texture_trust(rgb, pixels) if rgb is not None else None)


def write_start(folder: Path, rgb: np.ndarray, cam: Camera, seeds: Seeds, mask: np.ndarray | None = None) -> None:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(folder / "start.png")
    if mask is not None:
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "start_mask.png")
    (folder / "camera.json").write_text(json.dumps(cam.to_dict(), indent=2))
    seeds.save(folder / "seeds.npz")
