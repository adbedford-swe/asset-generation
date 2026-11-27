"""Point tracks through a clip: (T, N, 2) pixel positions and (T, N) visibility for the start frame's seeds.

``exact`` follows the seeds through a known animation (the benchmark's ground truth); ``cotracker``
tracks them through a video with CoTracker3 (offline), which follows any surface, so it suits any
anatomy. CoTracker runs from a pinned checkout (:mod:`assetgen.thirdparty`); its code and weights are
CC BY-NC 4.0.
"""
from __future__ import annotations

import numpy as np
import torch

from .camera import Seeds, project


def exact(rig, rotation: torch.Tensor, translation: torch.Tensor, seeds: Seeds, cam,
          tolerance: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Where the seeds are in each frame of a posed sequence, and whether the camera sees them (a depth
    test against the posed mesh, ``tolerance`` of the asset size)."""
    from ..preview import render_mesh

    with torch.no_grad():
        points = rig.pose_points(rig.point_basis(seeds.faces, seeds.bary), rotation, translation)
        verts = rig.pose_points(rig.vertex_basis(), rotation, translation)
        xy, z = project(points, cam)
    xy, z = xy.cpu().numpy(), z.cpu().numpy()
    visible = np.zeros(z.shape, dtype=bool)
    for t in range(len(xy)):
        _, _, depth = render_mesh(verts[t].cpu().numpy(), rig.faces, cam, shade=False, return_depth=True)
        depth = depth.numpy()
        px = np.floor(xy[t]).astype(int)
        inside = (px[:, 0] >= 0) & (px[:, 0] < cam.width) & (px[:, 1] >= 0) & (px[:, 1] < cam.height)
        d = np.full(len(px), -np.inf)
        d[inside] = depth[px[inside, 1], px[inside, 0]]
        visible[t] = inside & (z[t] <= d + tolerance * rig.size)
    return xy.astype(np.float32), visible


def cotracker(frames: np.ndarray, seeds: Seeds, device: str = "cuda", chunk: int = 2048,
              log=print) -> tuple[np.ndarray, np.ndarray]:
    """CoTracker3 (offline) on ``frames`` (T, H, W, 3) uint8, queried at the seeds on frame 0."""
    import sys

    from ..thirdparty import cotracker_dir, require_capture

    checkpoint = require_capture(log)
    if str(cotracker_dir()) not in sys.path:
        sys.path.insert(0, str(cotracker_dir()))
    from cotracker.predictor import CoTrackerPredictor

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    log(f"[capture] tracking {len(seeds.pixels)} points through {len(frames)} frames with CoTracker3 on {dev}")
    model = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False)
    model.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(dev).eval()
    video = torch.as_tensor(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None].float().to(dev)
    tracks, visible = [], []
    # CoTracker's coordinates put pixel centres on integers.
    q = torch.as_tensor(seeds.pixels - 0.5, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(q), chunk):
            part = q[i:i + chunk]
            queries = torch.cat([torch.zeros(len(part), 1), part], 1)[None].to(dev)
            xy, vis = model(video, queries=queries)
            tracks.append(xy[0].cpu() + 0.5)
            visible.append(vis[0].cpu())
    return torch.cat(tracks, 1).numpy().astype(np.float32), torch.cat(visible, 1).numpy().astype(bool)
