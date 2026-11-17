"""Pure-PyTorch 3D Gaussian Splatting forward renderer (no C++/HIP, no gsplat).

Renders a standard 3DGS .ply from a set of orbit cameras to RGB + depth +
coverage, for downstream TSDF mesh fusion. Forward inference only — no
differentiable backward, so a vectorized alpha-compositing splat on the
ROCm PyTorch build is all we need.

Convention
----------
* World: Y-up, object centered near the origin.
* Camera: looks down +Z (into the scene), X right, Y down. A point in front
  of the camera has camera-space Z > 0.
* Depth output: expected depth  E[z] = sum(T_i a_i z_i) / sum(T_i a_i),
  which is smooth and well suited to TSDF fusion.

Quaternion storage follows the INRIA 3DGS .ply layout: rot_0..3 = (x, y, z, w).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# PLY loading
# --------------------------------------------------------------------------- #
def load_gaussians(ply_path: str | Path, device: str = "cpu") -> dict:
    """Load a standard 3DGS .ply into a dict of tensors on `device`."""
    import plyfile

    ply = plyfile.PlyData.read(str(ply_path))
    el = ply["vertex"]
    n = len(el)

    def col(name: str) -> torch.Tensor:
        return torch.tensor(np.asarray(el[name], dtype=np.float32), device=device)

    positions = torch.stack([col("x"), col("y"), col("z")], dim=1)          # (N,3)
    log_scales = torch.stack([col("scale_0"), col("scale_1"), col("scale_2")], dim=1)
    quats = torch.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], dim=1)
    opacities = torch.sigmoid(col("opacity"))                                # (N,)
    sh0 = torch.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], dim=1)  # (N,3)

    # 0th-order SH -> base RGB albedo.
    colors = (0.28209476 * sh0 + 0.5).clamp(0.0, 1.0)

    quats = F.normalize(quats, dim=1)
    scales = torch.exp(log_scales)

    # 3D covariance  Sigma = R S S^T R^T   (world space)
    cov3d = _covariance_from_scale_quat(scales, quats)

    return {
        "positions": positions,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
        "cov3d": cov3d,
        "n": n,
    }


def _covariance_from_scale_quat(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """scales (N,3), quats (N,4) as (x,y,z,w) -> cov3d (N,3,3)."""
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]

    # Rotation matrix rows (standard quaternion -> matrix).
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1).reshape(-1, 3, 3)
    S = torch.diag_embed(torch.stack([sx * sx, sy * sy, sz * sz], dim=1))  # S S^T
    cov = torch.bmm(torch.bmm(R, S), R.transpose(1, 2))
    return cov


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    view: torch.Tensor      # (4,4) world -> camera
    intr: torch.Tensor      # (3,3)
    width: int
    height: int
    label: str = ""

    @property
    def eye(self) -> torch.Tensor:
        return -self.view[:3, :3].T @ self.view[:3, 3]


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """World->camera view matrix; camera looks down +Z, X right, Y down."""
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)
    forward = F.normalize(target - eye, dim=-1)          # +Z
    right = F.normalize(torch.cross(forward, up), dim=-1)  # +X
    down = torch.cross(forward, right)            # +Y (down)
    R = torch.stack([right, down, forward], dim=0)  # (3,3) rows
    t = -(R @ eye)
    view = torch.eye(4, dtype=torch.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def orbit_cameras(
    center: torch.Tensor,
    radius: float,
    fov_deg: float,
    n_views: int,
    image_size: int = 512,
    elevations: tuple[float, float] = (-20.0, 20.0),
    device: str = "cpu",
) -> list[Camera]:
    """Ring of cameras around `center` at `radius`, elevations swept over the ring."""
    center = torch.as_tensor(center, dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    f = (image_size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    intr = torch.tensor([[f, 0, image_size / 2.0],
                         [0, f, image_size / 2.0],
                         [0, 0, 1.0]], device=device)
    cams: list[Camera] = []
    for i in range(n_views):
        az = 2.0 * math.pi * i / n_views
        el = math.radians(elevations[0] + (elevations[1] - elevations[0]) * i / max(1, n_views - 1))
        # Spherical -> cartesian (Y-up).
        eye = center + radius * torch.tensor(
            [math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)],
            device=device,
        )
        view = look_at(eye, center, up)
        cams.append(Camera(view=view, intr=intr, width=image_size, height=image_size,
                           label=f"v{i:03d}"))
    return cams


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_view(g: dict, cam: Camera, tile: int = 16, near: float = 0.05) -> dict:
    """Render one camera. Returns rgb (H,W,3), depth (H,W), alpha (H,W)."""
    dev = g["positions"].device
    H, W = cam.height, cam.width
    R = cam.view[:3, :3].to(dev)
    t = cam.view[:3, 3].to(dev)
    f = cam.intr[0, 0].item()
    cx, cy = cam.intr[0, 2].item(), cam.intr[1, 2].item()

    pos = g["positions"]
    cov = g["cov3d"]
    colors = g["colors"]
    opac = g["opacities"]

    # Transform to camera space.
    mean_cam = pos @ R.T + t                                   # (N,3)
    cov_cam = torch.einsum("ij,njk,lk->nil", R, cov, R)        # R cov R^T  (N,3,3)

    z = mean_cam[:, 2]
    valid = z > near
    if not valid.any():
        return {"rgb": torch.zeros(H, W, 3, device=dev),
                "depth": torch.full((H, W), float("inf"), device=dev),
                "alpha": torch.zeros(H, W, device=dev)}

    x, y, zc = mean_cam[:, 0], mean_cam[:, 1], z
    inv_z = 1.0 / zc.clamp(min=near)

    # 2D projection.
    u = f * x * inv_z + cx
    v = f * y * inv_z + cy

    # 2D covariance via the projection Jacobian  C2d = J cov_cam J^T.
    Jm = torch.zeros(zc.shape[0], 2, 3, device=dev)
    Jm[:, 0, 0] = f * inv_z
    Jm[:, 0, 2] = -f * x * inv_z * inv_z
    Jm[:, 1, 1] = f * inv_z
    Jm[:, 1, 2] = -f * y * inv_z * inv_z
    C2d = torch.einsum("nij,njk,nlk->nil", Jm, cov_cam, Jm)     # (N,2,2)

    # PSD clamp + bounding radius + conic via eigen-decomposition.
    eigval, eigvec = torch.linalg.eigh(C2d)                     # eigval (N,2) asc
    eigval = eigval.clamp(min=1e-9)
    radius = 3.0 * torch.sqrt(eigval[:, 1])
    C2d_psd = torch.bmm(torch.bmm(eigvec, torch.diag_embed(eigval)), eigvec.transpose(1, 2))
    det = (C2d_psd[:, 0, 0] * C2d_psd[:, 1, 1] - C2d_psd[:, 0, 1] ** 2).clamp(min=1e-12)
    conic_a = C2d_psd[:, 1, 1] / det
    conic_b = -C2d_psd[:, 0, 1] / det
    conic_c = C2d_psd[:, 0, 0] / det

    # Keep Gaussians whose 2D bbox intersects the image.
    in_img = (u + radius >= 0) & (u - radius <= W) & (v + radius >= 0) & (v - radius <= H) & valid
    keep = torch.nonzero(in_img, as_tuple=False).squeeze(1)
    if keep.numel() == 0:
        return {"rgb": torch.zeros(H, W, 3, device=dev),
                "depth": torch.full((H, W), float("inf"), device=dev),
                "alpha": torch.zeros(H, W, device=dev)}

    u = u[keep]; v = v[keep]; zc = zc[keep]; radius = radius[keep]
    ca = conic_a[keep]; cb = conic_b[keep]; cc = conic_c[keep]
    col = colors[keep]; opa = opac[keep]

    # ---- Tile assignment (vectorized) ----
    assert W % tile == 0 and H % tile == 0, "image size must be a multiple of tile"
    nt_x, nt_y = W // tile, H // tile
    tx0 = torch.floor((u - radius) / tile).long().clamp(0, nt_x - 1)
    tx1 = torch.floor((u + radius) / tile).long().clamp(0, nt_x - 1)
    ty0 = torch.floor((v - radius) / tile).long().clamp(0, nt_y - 1)
    ty1 = torch.floor((v + radius) / tile).long().clamp(0, nt_y - 1)
    nx = (tx1 - tx0 + 1)
    ny = (ty1 - ty0 + 1)
    count = nx * ny
    total = int(count.sum())
    gids = torch.repeat_interleave(torch.arange(keep.numel(), device=dev), count)
    # Per-assignment tile coords: local index within this gaussian's tile block.
    start = torch.cumsum(count, 0) - count
    local = torch.arange(total, device=dev) - torch.repeat_interleave(start, count) if total else torch.zeros(0, dtype=torch.long, device=dev)
    tx = tx0[gids] + (local % nx[gids])
    ty = ty0[gids] + (local // nx[gids])
    tile_id = ty * nt_x + tx

    # Sort by (tile, depth) so each tile's list is front-to-back.
    order = torch.argsort(tile_id * 1000000 + zc[gids], stable=True)
    tile_id = tile_id[order]; gids = gids[order]; zc_s = zc[gids]

    # ---- Per-tile compositing (Python loop over tiles, vectorized inside) ----
    rgb = torch.zeros(H, W, 3, device=dev)
    depth = torch.full((H, W), float("inf"), device=dev)
    alpha = torch.zeros(H, W, device=dev)

    # Per-tile compositing (Python loop over tiles, vectorized inside).
    local_x = torch.arange(tile, device=dev)
    local_y = torch.arange(tile, device=dev)

    tile_bounds = torch.searchsorted(tile_id, torch.arange(nt_x * nt_y, device=dev))
    tile_bounds = torch.cat([tile_bounds, torch.tensor([tile_id.numel()], device=dev)])

    rgb_flat = rgb.view(-1, 3)
    alpha_flat = alpha.view(-1)
    depth_flat = depth.view(-1)

    for tid in range(int(tile_bounds.numel()) - 1):
        s, e = int(tile_bounds[tid]), int(tile_bounds[tid + 1])
        if e - s < 1:
            continue
        gi = gids[s:e]
        tr = tid // nt_x
        tc = tid % nt_x
        # Tile pixel coordinates (tile*tile,).
        py_t = (tr * tile + local_y)[:, None].expand(tile, tile).reshape(-1).float()
        px_t = (tc * tile + local_x)[None, :].expand(tile, tile).reshape(-1).float()
        # Offsets of tile pixels from each gaussian center.
        du = px_t[:, None] - u[gi][None, :]          # (P, G)
        dv = py_t[:, None] - v[gi][None, :]          # (P, G)
        power = -(ca[gi][None, :] * du * du + 2 * cb[gi][None, :] * du * dv + cc[gi][None, :] * dv * dv)
        a = (opa[gi][None, :] * torch.exp(power)).clamp(0.0, 1.0)   # (P, G)
        # Front-to-back transmittance.
        T = torch.cumprod(1.0 - a, dim=1)                       # (P, G)
        T_prev = torch.cat([torch.ones_like(a[:, :1]), T[:, :-1]], dim=1)
        w = T_prev * a                                          # (P, G)
        col_g = col[gi].T                                       # (3, G)
        rgb_t = torch.einsum("pg,cg->pc", w, col_g)             # (P, 3)
        a_t = w.sum(dim=1)                                      # (P,)
        d_t = (w * zc[gi][None, :]).sum(dim=1) / a_t.clamp(min=1e-8)
        # Scatter into the image at the tile's flat pixel indices.
        flat = py_t.long() * W + px_t.long()
        rgb_flat[flat] = rgb_t
        alpha_flat[flat] = a_t
        depth_flat[flat] = d_t

    return {"rgb": rgb, "depth": depth, "alpha": alpha}


# --------------------------------------------------------------------------- #
# Batch render + save
# --------------------------------------------------------------------------- #
def render_views(g: dict, cameras: list[Camera], out_dir: str | Path,
                 tile: int = 16) -> list[dict]:
    """Render all cameras; save RGB PNGs, depth EXR/PNG, coverage, and poses.json."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for cam in cameras:
        res = render_view(g, cam, tile=tile)
        rgb8 = (res["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb8).save(out_dir / f"{cam.label}.png")

        depth = res["depth"].cpu().numpy()
        alpha = res["alpha"].cpu().numpy()
        # Save depth as 16-bit PNG (mm) with background = 0 where no coverage.
        d_mm = np.where(alpha > 0.0, depth * 1000.0, 0.0).astype(np.uint16)
        Image.fromarray(d_mm).save(out_dir / f"{cam.label}_depth.png")
        Image.fromarray((alpha * 255).astype(np.uint8)).save(out_dir / f"{cam.label}_mask.png")

        records.append({
            "label": cam.label,
            "view": cam.view.cpu().numpy().tolist(),
            "intr": cam.intr.cpu().numpy().tolist(),
            "width": cam.width,
            "height": cam.height,
        })
    (out_dir / "poses.json").write_text(json.dumps(records, indent=2))
    return records
