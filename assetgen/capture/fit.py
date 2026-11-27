"""Fit the rig's motion to point tracks.

Unknowns: every joint's rotation in every frame, the hub's translation, and the travel (below). The fit
minimises the distance between where each tracked surface point lands on the posed mesh and where its
track says it is (a robust loss, over visible points), plus:

- ground: the character stands on the ground its rest pose stands on. Nothing sinks below it, and a
  point resting on it does not slide. This is what pins depth: one camera cannot tell a limb swung
  towards it from one swung away, but a foot on the ground has one place along its pixel's ray.
- travel: a point resting on the ground may move by the frame's travel, a velocity along the ground
  shared by the whole body. A video whose camera follows the character shows planted feet sliding
  back at the character's speed; travel takes that up, and is that speed.
- smoothness: joint rotations, the hub's path and the travel accelerate little from frame to frame;
- economy: of the poses that explain the tracks, the one closest to rest.

The start frame is the render the tracks were seeded on, so it is pinned to the rest pose; so is the
last when the video was asked to end on that render too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .camera import Seeds, project


@dataclass
class FitOptions:
    iterations: int = 2500
    lr: float = 0.03
    huber: float = 1.5           # pixels: the robust loss is quadratic below this, linear above
    smooth: float = 0.3          # acceleration of joint rotations (radians per frame squared)
    smooth_path: float = 0.3     # acceleration of the hub's path and of travel, in asset sizes
    economy: float = 0.02        # squared rotation away from rest
    depth: float = 2.0           # disagreement with the frames' relative depth (see depth.py)
    ground: float = 200.0        # sinking below the ground, in asset sizes squared
    slide: float = 200.0         # sliding of points resting on the ground, in asset sizes squared
    slide_scale: float = 0.004   # a slide per frame, in asset sizes, past which the penalty only grows slowly
    contact_band: float = 0.015  # how close to the ground a point rests, as a share of the asset's size
    anchor_start: bool = True
    anchor_end: bool = False
    stages: int = 3              # coarse to fine in time: fit every 4th, 2nd, then every frame
    refine: int = 300            # L-BFGS iterations after the last stage (0: none)
    ground_points: int = 1500


@dataclass
class Fit:
    rotation: torch.Tensor       # (T, J, 3)
    translation: torch.Tensor    # (T, 3) the hub's, in its parent's frame
    travel: torch.Tensor         # (T, 2) ground velocity (x, z) per frame, world units per frame
    residual: np.ndarray         # (T,) mean pixel distance over visible points
    report: dict = field(default_factory=dict)


def _accel(x: torch.Tensor) -> torch.Tensor:
    return x[2:] - 2 * x[1:-1] + x[:-2] if len(x) > 2 else x[:0]


def _interpolate(x: torch.Tensor, frames: np.ndarray, T: int) -> torch.Tensor:
    """Values at ``frames`` (sorted, including 0 and T-1) linearly interpolated to every frame."""
    out = torch.empty(T, *x.shape[1:], dtype=x.dtype, device=x.device)
    for a, b, xa, xb in zip(frames[:-1], frames[1:], x[:-1], x[1:]):
        w = torch.linspace(0, 1, int(b - a) + 1, dtype=x.dtype, device=x.device)[:-1]
        out[a:b] = xa[None] + w.view(-1, *[1] * (x.dim() - 1)) * (xb - xa)[None]
    out[frames[-1]] = x[-1]
    return out


def ground_vertices(rig, count: int, seed: int = 0) -> np.ndarray:
    """Vertices that stand for the body against the ground: all of the lowest tenth, and a spread of
    the rest (anything may touch the ground: a knee, a tail, a body lying down)."""
    with torch.no_grad():
        rest = rig.pose_points(rig.vertex_basis(), *rig.zeros(1))[0].cpu().numpy()
    y = rest[:, 1]
    low = np.nonzero(y <= y.min() + 0.1 * np.ptp(y))[0]
    rng = np.random.default_rng(seed)
    if len(low) > count // 2:
        low = rng.choice(low, count // 2, replace=False)
    rest_idx = np.setdiff1d(np.arange(len(y)), low)
    spread = rng.choice(rest_idx, min(len(rest_idx), count - len(low)), replace=False)
    return np.concatenate([low, spread])


class Objective:
    """The fit's loss, by term, for a pose sequence at some of the clip's frames."""

    def __init__(self, rig, seeds: Seeds, tracks: np.ndarray, visible: np.ndarray, cam, opts: FitOptions,
                 depth: np.ndarray | None = None):
        dev, dt = rig.device, rig.dtype
        self.rig, self.cam, self.opts = rig, cam, opts
        self.T = len(tracks)
        self.basis = rig.point_basis(seeds.faces, seeds.bary)
        self.ground_basis = rig.vertex_basis(ground_vertices(rig, opts.ground_points))
        self.target = torch.as_tensor(tracks, dtype=dt, device=dev)
        self.weight = torch.as_tensor(visible, dtype=dt, device=dev)
        self.observed = torch.as_tensor(depth, dtype=dt, device=dev) if depth is not None else None
        self.scale = rig.size
        with torch.no_grad():
            self.floor = float(rig.pose_points(self.ground_basis, *rig.zeros(1))[0, :, 1].min())
        self.band = opts.contact_band * self.scale

    def terms(self, r: torch.Tensor, t: torch.Tensor, trav: torch.Tensor, frames: np.ndarray) -> dict:
        """Loss terms for rotations ``r``, hub translations ``t`` and travel ``trav`` at ``frames``."""
        o, rig, scale = self.opts, self.rig, self.scale
        stride = float(np.diff(frames).mean()) if len(frames) > 1 else 1.0
        gaps = torch.as_tensor(np.diff(frames), dtype=r.dtype, device=r.device)[:, None]
        tgt, w = self.target[frames], self.weight[frames]
        xy, z = project(rig.pose_points(self.basis, r, t), self.cam)
        d = (xy - tgt).norm(dim=-1)
        robust = torch.where(d < o.huber, 0.5 * d * d / o.huber, d - 0.5 * o.huber)
        out = {"tracks": (robust * w).sum() / w.sum().clamp_min(1)}
        # Smoothness is per frame of the clip: a stride of s between fitted frames scales acceleration by s^2.
        out["smooth"] = o.smooth * (_accel(r) / stride ** 2).pow(2).sum(-1).mean()
        out["smooth_path"] = o.smooth_path * ((_accel(t / scale) / stride ** 2).pow(2).sum(-1).mean()
                                              + (_accel(trav / scale) / stride).pow(2).sum(-1).mean())
        out["economy"] = o.economy * r.pow(2).sum(-1).mean()
        if self.observed is not None and o.depth > 0:
            from .depth import affine_residual

            out["depth"] = o.depth * affine_residual(1 / z.clamp_min(1e-6), self.observed[frames], w).mean()
        g = rig.pose_points(self.ground_basis, r, t)
        height = g[..., 1] - self.floor
        out["ground"] = o.ground * (torch.relu(-height) / scale).pow(2).sum(-1).mean()
        # A point resting on the ground in two successive fitted frames moves only by the travel between them.
        rest_w = torch.exp(-torch.relu(height) / self.band)
        pair = torch.minimum(rest_w[1:], rest_w[:-1])
        step = g[1:, :, [0, 2]] - g[:-1, :, [0, 2]] - (trav[:-1] * gaps)[:, None]
        # Robust: a small slide costs its square, a large one only its logarithm, so where the tracks plainly
        # show a resting point sliding (a skid, a shuffle) they win rather than the fit hiding it in depth.
        s2 = (o.slide_scale / stride) ** 2
        slid = s2 * torch.log1p((step / scale).pow(2).sum(-1) / s2)
        out["slide"] = o.slide * (pair * slid).sum(-1).mean() / max(1.0, float(len(self.ground_basis)) / 100)
        return out


def fit(rig, seeds: Seeds, tracks: np.ndarray, visible: np.ndarray, cam, opts: FitOptions | None = None,
        depth: np.ndarray | None = None, log=print) -> Fit:
    """``depth``: optional relative inverse depth (T, N) at the tracks (:mod:`assetgen.capture.depth`)."""
    opts = opts or FitOptions()
    objective = Objective(rig, seeds, tracks, visible, cam, opts, depth)
    T = objective.T
    dev, dt = rig.device, rig.dtype
    rotation, translation = rig.zeros(T)
    travel = torch.zeros(T, 2, dtype=dt, device=dev)

    levels = [2 ** k for k in reversed(range(max(1, opts.stages)))]
    for li, stride in enumerate(levels):
        frames = np.unique(np.r_[np.arange(0, T, stride), T - 1])
        n = len(frames)
        rot = rotation[frames].clone().requires_grad_(True)
        tr = translation[frames].clone().requires_grad_(True)
        trav = travel[frames].clone().requires_grad_(True)
        free = torch.ones(n, 1, dtype=dt, device=dev)
        if opts.anchor_start:
            free[0] = 0
        if opts.anchor_end:
            free[-1] = 0
        optim = torch.optim.Adam([rot, tr, trav], lr=opts.lr)
        steps = opts.iterations if li == len(levels) - 1 else opts.iterations // 2
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps, eta_min=opts.lr * 0.05)
        for _ in range(steps):
            optim.zero_grad()
            loss = sum(objective.terms(rot * free[..., None], tr * free, trav, frames).values())
            loss.backward()
            optim.step()
            sched.step()
        if opts.refine and li == len(levels) - 1:
            lbfgs = torch.optim.LBFGS([rot, tr, trav], lr=1.0, max_iter=opts.refine, history_size=50,
                                      line_search_fn="strong_wolfe", tolerance_change=1e-12, tolerance_grad=1e-10)

            def closure():
                lbfgs.zero_grad()
                loss = sum(objective.terms(rot * free[..., None], tr * free, trav, frames).values())
                loss.backward()
                return loss

            lbfgs.step(closure)
        with torch.no_grad():
            r, t = rot * free[..., None], tr * free
            rotation = _interpolate(r.detach(), frames, T)
            translation = _interpolate(t.detach(), frames, T)
            travel = _interpolate(trav.detach(), frames, T)

    with torch.no_grad():
        xy, _ = project(rig.pose_points(objective.basis, rotation, translation), cam)
        d = (xy - objective.target).norm(dim=-1).cpu().numpy()
        final = {k: float(v) for k, v in objective.terms(rotation, translation, travel, np.arange(T)).items()}
    vis = np.asarray(visible, dtype=bool)
    residual = np.array([d[t][vis[t]].mean() if vis[t].any() else np.nan for t in range(T)])
    report = {"frames": T, "points": int(len(seeds.pixels)), "loss": final,
              "residual_px": {"mean": float(np.nanmean(residual)), "max": float(np.nanmax(residual)),
                              "p90_points": float(np.percentile(d[vis], 90)) if vis.any() else None}}
    log(f"[capture] fitted {T} frames: tracks missed by {report['residual_px']['mean']:.2f} px on average")
    return Fit(rotation, translation, travel, residual, report)
