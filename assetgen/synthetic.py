"""Idol-like test inputs from a textured mesh with a known pose.

Samples the surface into flat Gaussians, writes them the way Idol does (Y-down,
at a chosen yaw) and renders an Idol-style preprocessed input image (object
cropped to its bounding box, composited on black) from a chosen view. Used by
the tests and for checking the pipeline without Idol's gated encoders.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .orient import IDOL_UP_FIX
from .splats import Gaussians, axis_angle_matrix, make_camera, render


def load_textured_mesh(path):
    """All triangles of a glTF/OBJ scene in world space, as one trimesh with per-face colours sampled."""
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry[name].copy()
        geom.apply_transform(transform)
        parts.append(geom)
    return parts


def _surface_colour(mesh, points: np.ndarray, face_index: np.ndarray) -> np.ndarray:
    """Base colour (RGB in [0, 1]) at surface points, from the texture when there is one."""
    import trimesh

    visual = mesh.visual
    material = getattr(visual, "material", None)
    factor = np.ones(4)
    image = None
    if material is not None:
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if getattr(material, "baseColorFactor", None) is not None:
            factor = np.asarray(material.baseColorFactor, dtype=np.float64)
            factor = factor / 255.0 if factor.max() > 1 else factor
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None:
        tri = mesh.triangles[face_index]
        bary = trimesh.triangles.points_to_barycentric(tri, points)
        uvp = (uv[mesh.faces[face_index]] * bary[..., None]).sum(1)
        img = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        h, w = img.shape[:2]
        x = np.clip((uvp[:, 0] % 1.0) * w, 0, w - 1).astype(int)
        y = np.clip((1 - uvp[:, 1] % 1.0) * h, 0, h - 1).astype(int)
        return img[y, x] * factor[:3]
    if hasattr(visual, "vertex_colors") and visual.kind == "vertex":
        return visual.vertex_colors[mesh.faces[face_index]].mean(1)[:, :3] / 255.0
    return np.tile(factor[:3] * 0.8, (len(points), 1))


def mesh_to_gaussians(parts, n: int = 200_000, seed: int = 0) -> Gaussians:
    import trimesh

    rng = np.random.default_rng(seed)
    areas = np.array([p.area for p in parts])
    counts = np.maximum(1, (n * areas / areas.sum()).astype(int))
    pts, nrm, col = [], [], []
    for p, k in zip(parts, counts):
        s, fi = trimesh.sample.sample_surface(p, int(k), seed=int(rng.integers(1 << 31)))
        pts.append(s)
        nrm.append(p.face_normals[fi])
        col.append(_surface_colour(p, s, fi))
    pts = np.concatenate(pts)
    nrm = np.concatenate(nrm)
    col = np.concatenate(col)
    sigma = math.sqrt(areas.sum() / len(pts)) * 0.6

    normals = F.normalize(torch.tensor(nrm, dtype=torch.float32), dim=-1)
    helper = torch.where(normals[:, :1].abs() < 0.9, torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0]))
    t1 = F.normalize(torch.linalg.cross(normals, helper), dim=-1)
    t2 = torch.linalg.cross(normals, t1)
    frames = torch.stack([t1, t2, normals], dim=2)  # columns: tangent, bitangent, normal
    quats = _batch_quat(frames)
    scales = torch.tensor([sigma, sigma, sigma * 0.1]).expand(len(pts), 3).clone()
    return Gaussians(torch.tensor(pts, dtype=torch.float32), quats, scales,
                     torch.full((len(pts),), 0.95), torch.tensor(col, dtype=torch.float32))


def _batch_quat(R: torch.Tensor) -> torch.Tensor:
    w = torch.sqrt((1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp_min(1e-12)) / 2
    x = torch.sqrt((1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp_min(0)) / 2
    y = torch.sqrt((1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]).clamp_min(0)) / 2
    z = torch.sqrt((1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]).clamp_min(0)) / 2
    x = torch.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = torch.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = torch.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    return F.normalize(torch.stack([w, x, y, z], dim=1), dim=-1)


def normalise_unit_box(g: Gaussians, extent: float = 0.9) -> Gaussians:
    lo, hi = g.means.min(0).values, g.means.max(0).values
    s = extent / float((hi - lo).max())
    return g.transformed(torch.eye(3), -s * (lo + hi) / 2, s)


def idol_like(g_canonical: Gaussians, yaw_deg: float) -> Gaussians:
    """A canonical (Y-up, facing +Z) splat turned to face ``yaw_deg`` and written Y-down like Idol."""
    R = IDOL_UP_FIX @ axis_angle_matrix([0.0, 1.0, 0.0], yaw_deg)
    return normalise_unit_box(g_canonical).transformed(R, torch.zeros(3))


def input_image(g_canonical: Gaussians, view_yaw_deg: float, pitch_deg: float = 10.0,
                size: int = 512, margin: float = 1.2) -> np.ndarray:
    """Idol-style preprocessed image of the canonical splat seen from ``view_yaw_deg``."""
    g = normalise_unit_box(g_canonical)
    d = torch.tensor([math.sin(math.radians(view_yaw_deg)) * math.cos(math.radians(pitch_deg)),
                      math.sin(math.radians(pitch_deg)),
                      math.cos(math.radians(view_yaw_deg)) * math.cos(math.radians(pitch_deg))])
    r = render(g, make_camera(2.5 * d, torch.zeros(3), 30.0, size * 2))
    r = type(r)(r.rgb.cpu(), r.alpha.cpu(), r.depth.cpu())
    ys, xs = torch.nonzero(r.alpha > 0.5, as_tuple=True)
    cy, cx = (ys.min() + ys.max()).item() / 2, (xs.min() + xs.max()).item() / 2
    half = max(int(ys.max() - ys.min()), int(xs.max() - xs.min())) / 2 * margin
    rgb = (r.rgb * r.alpha[..., None]).clamp(0, 1).permute(2, 0, 1)[None]
    gy = torch.linspace(cy - half, cy + half, size) / (size * 2) * 2 - 1
    gx = torch.linspace(cx - half, cx + half, size) / (size * 2) * 2 - 1
    grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), dim=-1)[None]
    out = F.grid_sample(rgb, grid, align_corners=True)[0].permute(1, 2, 0)
    return (out.numpy() * 255).astype(np.uint8)
