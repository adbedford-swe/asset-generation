"""3D Gaussian splat loading and a pure-PyTorch forward rasterizer.

The rasterizer follows the reference 3DGS forward pass (EWA projection, 0.3 px
low-pass dilation, 3-sigma screen extent, front-to-back compositing per
16x16 tile) without any compiled extension, so it runs unchanged on ROCm,
CUDA and CPU. Tiles are composited in batches of similar list length, which
keeps padding small and avoids a Python loop per tile.

Conventions
-----------
* PLY layout is the INRIA one: ``rot_0..3`` is a (w, x, y, z) quaternion,
  ``scale_*`` are log scales, ``opacity`` is a logit, ``f_dc_*`` are degree-0
  SH coefficients.
* Cameras are OpenCV style: world -> camera ``view`` matrix, camera looks down
  +Z with X right and Y down in the image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    means: torch.Tensor      # (N, 3)
    quats: torch.Tensor      # (N, 4) unit, (w, x, y, z)
    scales: torch.Tensor     # (N, 3) linear
    opacities: torch.Tensor  # (N,)
    colors: torch.Tensor     # (N, 3) in [0, 1]

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    def to(self, device) -> "Gaussians":
        return Gaussians(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def subset(self, index) -> "Gaussians":
        return Gaussians(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def transformed(self, rotation: torch.Tensor, translation: torch.Tensor, scale: float = 1.0) -> "Gaussians":
        """Apply x -> scale * R x + t to the splat (R a proper rotation)."""
        rotation = rotation.to(self.means)
        q = matrix_to_quat(rotation)
        return Gaussians(
            means=scale * self.means @ rotation.T + translation.to(self.means),
            quats=F.normalize(quat_multiply(q.expand_as(self.quats), self.quats), dim=-1),
            scales=self.scales * scale,
            opacities=self.opacities,
            colors=self.colors,
        )

    def covariances(self) -> torch.Tensor:
        R = quat_to_matrix(self.quats)
        S2 = self.scales ** 2
        return torch.einsum("nij,nj,nkj->nik", R, S2, R)


def load_ply(path: str | Path, device="cpu") -> Gaussians:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]

    def cols(*names):
        return torch.from_numpy(np.stack([np.asarray(vertex[n], dtype=np.float32) for n in names], axis=1))

    means = cols("x", "y", "z")
    quats = F.normalize(cols("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1)
    scales = torch.exp(cols("scale_0", "scale_1", "scale_2"))
    opacities = torch.sigmoid(cols("opacity")[:, 0])
    colors = (SH_C0 * cols("f_dc_0", "f_dc_1", "f_dc_2") + 0.5).clamp(0.0, 1.0)
    g = Gaussians(means, quats, scales, opacities, colors)
    finite = torch.isfinite(torch.cat([means, quats, scales, opacities[:, None], colors], dim=1)).all(dim=1)
    return g.subset(finite).to(device)


def save_ply(g: Gaussians, path: str | Path) -> None:
    from plyfile import PlyData, PlyElement

    eps = 1e-6
    op = g.opacities.clamp(eps, 1 - eps)
    attrs = torch.cat([
        g.means,
        torch.zeros_like(g.means),
        (g.colors - 0.5) / SH_C0,
        torch.log(op / (1 - op))[:, None],
        torch.log(g.scales.clamp_min(eps)),
        g.quats,
    ], dim=1).detach().cpu().numpy().astype(np.float32)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.empty(attrs.shape[0], dtype=[(n, "f4") for n in names])
    for i, n in enumerate(names):
        data[n] = attrs[:, i]
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


# --------------------------------------------------------------------------- #
# Rotations
# --------------------------------------------------------------------------- #
def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (3, 3) -> unit quaternion (w, x, y, z)."""
    m = R.double()
    t = m.trace()
    if t > 0:
        s = torch.sqrt(t + 1.0) * 2
        q = torch.stack([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = torch.stack([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = torch.stack([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = torch.stack([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return F.normalize(q, dim=-1).to(R.dtype)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def axis_angle_matrix(axis, degrees: float) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    a = math.radians(degrees)
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=torch.float64)
    return (torch.eye(3, dtype=torch.float64) + math.sin(a) * K + (1 - math.cos(a)) * K @ K).float()


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    view: torch.Tensor  # (4, 4) world -> camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def position(self) -> torch.Tensor:
        R, t = self.view[:3, :3], self.view[:3, 3]
        return -R.T @ t

    def to_dict(self) -> dict:
        return {"view": self.view.tolist(), "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height}


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> torch.Tensor:
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)
    forward = F.normalize(target - eye, dim=-1)
    right = torch.linalg.cross(forward, up)
    if right.norm() < 1e-6:  # looking straight along `up`
        right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0]))
    right = F.normalize(right, dim=-1)
    down = torch.linalg.cross(forward, right)
    view = torch.eye(4)
    view[:3, :3] = torch.stack([right, down, forward])
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def make_camera(eye, target, fov_deg: float, size: int, up=(0.0, 1.0, 0.0)) -> Camera:
    f = 0.5 * size / math.tan(math.radians(fov_deg) / 2)
    return Camera(look_at(eye, target, up), f, f, size / 2, size / 2, size, size)


def sphere_directions(n: int) -> torch.Tensor:
    """Fibonacci-sphere unit directions (n, 3), Y-up."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    y = 1 - 2 * i / n
    r = torch.sqrt(1 - y * y)
    phi = math.pi * (3 - math.sqrt(5)) * i
    return torch.stack([r * torch.cos(phi), y, r * torch.sin(phi)], dim=1).float()


def orbit_cameras(center, radius: float, n_views: int, fov_deg: float, size: int,
                  min_elevation_deg: float = -60.0) -> list[Camera]:
    """Cameras spread evenly over the sphere around ``center``, skipping views from far below."""
    center = torch.as_tensor(center, dtype=torch.float32)
    lo = math.sin(math.radians(min_elevation_deg))
    dirs = sphere_directions(n_views * 2)
    dirs = dirs[dirs[:, 1] >= lo]
    idx = torch.linspace(0, len(dirs) - 1, n_views).round().long()
    cams = []
    for d in dirs[idx]:
        up = (0.0, 1.0, 0.0) if abs(float(d[1])) < 0.99 else (0.0, 0.0, 1.0)
        cams.append(make_camera(center + radius * d, center, fov_deg, size, up))
    return cams


def framing_radius(extent_radius: float, fov_deg: float, fill: float = 0.9) -> float:
    """Camera distance at which a sphere of ``extent_radius`` fills ``fill`` of the view."""
    return extent_radius / math.sin(math.radians(fov_deg) / 2 * fill)


# --------------------------------------------------------------------------- #
# Rasterizer
# --------------------------------------------------------------------------- #
@dataclass
class Render:
    rgb: torch.Tensor     # (H, W, 3)
    alpha: torch.Tensor   # (H, W)
    depth: torch.Tensor   # (H, W) median depth (where accumulated opacity crosses 0.5), 0 if never


def default_budget(device) -> int:
    """Elements per (tiles, pixels, gaussians) batch tensor; about a dozen such tensors are live at once."""
    device = torch.device(device)
    if device.type != "cuda":
        return 1 << 24
    free, _ = torch.cuda.mem_get_info(device)
    return int(max(1 << 20, min(1 << 27, free * 0.5 / (4 * 12))))


def render(g: Gaussians, cam: Camera, background=(0.0, 0.0, 0.0), tile: int = 16,
           near: float = 0.01, budget: int | None = None) -> Render:
    """Render one view. ``budget`` caps the elements of the per-batch (tiles, pixels, gaussians) tensors."""
    dev = g.device
    budget = budget or default_budget(dev)
    H, W = cam.height, cam.width
    R = cam.view[:3, :3].to(dev)
    t = cam.view[:3, 3].to(dev)

    mean_c = g.means @ R.T + t
    z = mean_c[:, 2]
    front = z > near
    idx = torch.nonzero(front).squeeze(1)
    mean_c = mean_c[idx]
    z = z[idx]
    cov_c = R @ g.covariances()[idx] @ R.T

    inv_z = 1.0 / z
    u = cam.fx * mean_c[:, 0] * inv_z + cam.cx
    v = cam.fy * mean_c[:, 1] * inv_z + cam.cy

    # Clamp the Jacobian at 1.3x the frustum like the reference implementation.
    lim_x = 1.3 * (0.5 * W / cam.fx)
    lim_y = 1.3 * (0.5 * H / cam.fy)
    tx = (mean_c[:, 0] * inv_z).clamp(-lim_x, lim_x) * z
    ty = (mean_c[:, 1] * inv_z).clamp(-lim_y, lim_y) * z
    J = torch.zeros(len(idx), 2, 3, device=dev)
    J[:, 0, 0] = cam.fx * inv_z
    J[:, 0, 2] = -cam.fx * tx * inv_z ** 2
    J[:, 1, 1] = cam.fy * inv_z
    J[:, 1, 2] = -cam.fy * ty * inv_z ** 2
    cov2 = J @ cov_c @ J.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    ok = det > 1e-12
    half_tr = 0.5 * (a + c)
    lam = half_tr + torch.sqrt((half_tr ** 2 - det).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam))
    on_screen = ok & (u + radius > 0) & (u - radius < W) & (v + radius > 0) & (v - radius < H)
    keep = torch.nonzero(on_screen).squeeze(1)

    rgb = torch.zeros(H * W, 3, device=dev)
    alpha = torch.zeros(H * W, device=dev)
    depth = torch.zeros(H * W, device=dev)
    bg = torch.as_tensor(background, dtype=torch.float32, device=dev)
    if keep.numel() == 0:
        return Render(bg.expand(H, W, 3).clone(), alpha.view(H, W), depth.view(H, W))

    u, v, z, radius = u[keep], v[keep], z[keep], radius[keep]
    conic = torch.stack([c, -b, a], dim=1)[keep] / det[keep, None]
    opac = g.opacities[idx[keep]]
    col = g.colors[idx[keep]]

    # Front-to-back: sort by depth once, then a stable sort by tile keeps depth order inside each tile.
    order = torch.argsort(z)
    u, v, z, radius, conic, opac, col = (x[order] for x in (u, v, z, radius, conic, opac, col))

    ntx, nty = math.ceil(W / tile), math.ceil(H / tile)
    x0 = ((u - radius) / tile).floor().clamp(0, ntx - 1).long()
    x1 = ((u + radius) / tile).floor().clamp(0, ntx - 1).long()
    y0 = ((v - radius) / tile).floor().clamp(0, nty - 1).long()
    y1 = ((v + radius) / tile).floor().clamp(0, nty - 1).long()
    nx = x1 - x0 + 1
    count = nx * (y1 - y0 + 1)
    gid = torch.repeat_interleave(torch.arange(len(u), device=dev), count)
    start = torch.cumsum(count, 0) - count
    local = torch.arange(len(gid), device=dev) - start[gid]
    tile_id = (y0[gid] + local // nx[gid]) * ntx + x0[gid] + local % nx[gid]
    tile_id, perm = torch.sort(tile_id, stable=True)
    gid = gid[perm]

    n_tiles = ntx * nty
    per_tile = torch.bincount(tile_id, minlength=n_tiles)
    tile_start = torch.cumsum(per_tile, 0) - per_tile
    occupied = torch.nonzero(per_tile).squeeze(1)
    occupied = occupied[torch.argsort(per_tile[occupied], descending=True)]

    P = tile * tile
    py, px = torch.meshgrid(torch.arange(tile, device=dev), torch.arange(tile, device=dev), indexing="ij")
    px = px.reshape(-1).float() + 0.5
    py = py.reshape(-1).float() + 0.5

    i = 0
    while i < len(occupied):
        L = int(per_tile[occupied[i]])
        B = max(1, min(len(occupied) - i, budget // (P * L)))
        tiles = occupied[i:i + B]
        i += B
        lens = per_tile[tiles]
        slot = torch.arange(L, device=dev)
        valid = slot[None, :] < lens[:, None]                                   # (B, L)
        gsel = gid[(tile_start[tiles][:, None] + slot[None, :]).clamp_max(len(gid) - 1)]
        tx0 = (tiles % ntx).float() * tile
        ty0 = (tiles // ntx).float() * tile
        dx = (tx0[:, None] + px[None, :])[:, :, None] - u[gsel][:, None, :]    # (B, P, L)
        dy = (ty0[:, None] + py[None, :])[:, :, None] - v[gsel][:, None, :]
        cg = conic[gsel]
        power = -0.5 * (cg[:, None, :, 0] * dx * dx + cg[:, None, :, 2] * dy * dy) - cg[:, None, :, 1] * dx * dy
        al = (opac[gsel][:, None, :] * torch.exp(power.clamp_max(0))).clamp_max(0.99)
        al = torch.where((al >= 1 / 255) & valid[:, None, :], al, torch.zeros((), device=dev))
        T = torch.cumprod(1 - al, dim=2)
        T_prev = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=2)
        w = T_prev * al
        out_rgb = torch.einsum("bpl,blc->bpc", w, col[gsel])
        out_a = 1 - T[..., -1]
        # Median depth: the first gaussian at which transmittance drops to 0.5 or below.
        crossed = T <= 0.5
        first = torch.argmax(crossed.int(), dim=2)
        hit = crossed.any(dim=2)
        out_d = torch.where(hit, torch.gather(z[gsel], 1, first), torch.zeros((), device=dev))

        pix_x = (tx0[:, None] + px[None, :] - 0.5).long()
        pix_y = (ty0[:, None] + py[None, :] - 0.5).long()
        inside = (pix_x < W) & (pix_y < H)
        flat = (pix_y * W + pix_x)[inside]
        rgb[flat] = out_rgb[inside]
        alpha[flat] = out_a[inside]
        depth[flat] = out_d[inside]

    rgb = rgb + (1 - alpha)[:, None] * bg
    return Render(rgb.view(H, W, 3), alpha.view(H, W), depth.view(H, W))
