"""Put a splat in the glTF character frame: Y up, facing +Z, centred, standing on y = 0.

Idol writes its PLY Y-down (its z-up object turned +90 degrees about X), and the
object's yaw is whatever the input image implied. Both are recovered without
relying on that convention:

* Up and yaw: render the splat around the vertical axis, upright and flipped,
  and match each view against Idol's preprocessed input image (silhouette IoU
  plus colour agreement, both cropped to their bounding boxes so focal length
  and distance do not matter). The best view is where the input camera was.
* Facing: most characters are close to mirror symmetric. The vertical plane
  that best maps the splat onto itself contains the facing direction; of its
  two senses the one closer to the input camera wins. Without a usable plane
  the input camera direction itself is the facing.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .splats import Gaussians, axis_angle_matrix, framing_radius, make_camera, render

IDOL_UP_FIX = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # 180 degrees about X


@dataclass
class Orientation:
    rotation: list           # 3x3, applied to the input splat
    translation: list
    scale: float
    flipped: bool            # the input was upside down
    camera_yaw_deg: float    # where the input camera was, about +Y from +Z, after the up fix
    facing_yaw_deg: float    # facing direction found, same convention
    symmetry_error: float | None
    match_score: float | None
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


def _bbox_crop(mask: torch.Tensor, rgb: torch.Tensor, size: int = 96):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(ys) < 16:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    img = torch.cat([rgb, mask[..., None].float()], dim=-1).permute(2, 0, 1)[None]
    H, W = mask.shape
    # Square window around the bbox, sampled with zero padding outside the image.
    gy = torch.linspace(cy - side / 2, cy + side / 2, size)
    gx = torch.linspace(cx - side / 2, cx + side / 2, size)
    grid = torch.stack(torch.meshgrid(gx / W * 2 - 1, gy / H * 2 - 1, indexing="xy"), dim=-1)[None]
    out = F.grid_sample(img.float(), grid.to(img.device), align_corners=True)[0]
    return out[:3].permute(1, 2, 0), out[3] > 0.5


def reference_from_image(path: str, device="cpu"):
    """Idol's preprocessed input: object composited on black. Returns (rgb, mask) cropped to the object."""
    from PIL import Image
    from scipy.ndimage import binary_fill_holes, binary_opening

    img = Image.open(path)
    if img.mode == "RGBA":
        arr = np.asarray(img).astype(np.float32) / 255
        rgb, mask = arr[..., :3], arr[..., 3] > 0.5
    else:
        rgb = np.asarray(img.convert("RGB")).astype(np.float32) / 255
        mask = binary_fill_holes(binary_opening(rgb.max(-1) > 8 / 255, iterations=2))
    return _bbox_crop(torch.from_numpy(mask).to(device), torch.from_numpy(rgb).to(device))


def _view_score(g: Gaussians, yaw: float, pitch: float, center, radius, ref) -> float:
    d = torch.tensor([math.sin(math.radians(yaw)) * math.cos(math.radians(pitch)),
                      math.sin(math.radians(pitch)),
                      math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))])
    cam = make_camera(center + radius * d, center, 30.0, 160)
    r = render(g, cam)
    crop = _bbox_crop(r.alpha > 0.5, r.rgb)
    if crop is None:
        return -1.0
    rgb, mask = crop
    ref_rgb, ref_mask = ref
    inter = (mask & ref_mask).float()
    iou = float(inter.sum() / (mask | ref_mask).float().sum().clamp_min(1))
    colour = float(((rgb - ref_rgb).abs().mean(-1) * inter).sum() / inter.sum().clamp_min(1))
    return iou - colour


def find_input_camera(g: Gaussians, ref, pitch_deg: float | None = None, log=print):
    """Search up/down and yaw for the view that reproduces the input image."""
    lo = torch.quantile(g.means.cpu(), 0.01, dim=0)
    hi = torch.quantile(g.means.cpu(), 0.99, dim=0)
    center = ((lo + hi) / 2).to(g.device)
    radius = framing_radius(float((hi - lo).norm()) / 2, 30.0)
    pitches = [pitch_deg] if pitch_deg is not None else [0.0, 15.0, 30.0]
    best = (-2.0, False, 0.0, 0.0)
    for flipped in (False, True):
        gg = g.transformed(IDOL_UP_FIX, torch.zeros(3)) if flipped else g
        c = IDOL_UP_FIX.to(center) @ center if flipped else center
        for pitch in pitches:
            for yaw in range(0, 360, 15):
                s = _view_score(gg, yaw, pitch, c.cpu(), radius, ref)
                if s > best[0]:
                    best = (s, flipped, float(yaw), pitch)
    score, flipped, yaw, pitch = best
    gg = g.transformed(IDOL_UP_FIX, torch.zeros(3)) if flipped else g
    c = (IDOL_UP_FIX.to(center) @ center if flipped else center).cpu()
    for step in (5.0, 1.0):
        for cand_yaw in np.arange(yaw - 3 * step, yaw + 3 * step + 1e-6, step):
            for cand_pitch in ([pitch] if pitch_deg is not None else [pitch - step, pitch, pitch + step]):
                s = _view_score(gg, float(cand_yaw), float(cand_pitch), c, radius, ref)
                if s > score:
                    score, yaw, pitch = s, float(cand_yaw) % 360, float(cand_pitch)
    log(f"[orient] input view: flipped={flipped} yaw={yaw:.0f} pitch={pitch:.0f} score={score:.3f}")
    return flipped, yaw, pitch, score


def symmetry_plane(g: Gaussians, samples: int = 20000, seed: int = 0):
    """Yaw (degrees) of the vertical mirror plane's in-plane horizontal direction, and its error.

    The error is the median nearest-neighbour distance after mirroring, relative to the
    object's size, so a tail or a weapon in one hand does not dominate it."""
    from scipy.spatial import cKDTree

    pts = g.means[g.opacities > 0.3].cpu().numpy()
    rng = np.random.default_rng(seed)
    if len(pts) > samples:
        pts = pts[rng.choice(len(pts), samples, replace=False)]
    pts = pts - np.median(pts, axis=0)
    size = np.linalg.norm(np.percentile(pts, 99, axis=0) - np.percentile(pts, 1, axis=0))
    tree = cKDTree(pts)

    def error(theta_deg: float) -> float:
        n = np.array([math.cos(math.radians(theta_deg)), 0.0, -math.sin(math.radians(theta_deg))])
        mirrored = pts - 2 * (pts @ n)[:, None] * n[None]
        d, _ = tree.query(mirrored, k=1)
        return float(np.median(d)) / size

    # The plane normal at angle theta (from +X towards -Z) holds the facing direction yaw = theta.
    coarse = [(error(t), t) for t in np.arange(0, 180, 3.0)]
    err, theta = min(coarse)
    for step in (1.0, 0.25):
        err, theta = min((error(t), t) for t in np.arange(theta - 3 * step, theta + 3 * step + 1e-6, step))
    spread = float(np.median([e for e, _ in coarse]))
    return theta % 180, err, spread


def orient(g: Gaussians, reference_image: str | None = None, pitch_deg: float | None = None,
           height: float | None = None, use_symmetry: bool = True, log=print):
    """Return (oriented splat, Orientation)."""
    if reference_image is not None:
        ref = reference_from_image(reference_image, g.device)
        flipped, cam_yaw, _, score = find_input_camera(g, ref, pitch_deg, log)
        method = "image"
    else:
        flipped, cam_yaw, score, method = True, 0.0, None, "idol-default"
        log("[orient] no input image: assuming Idol's Y-down PLY, facing +Z")

    up = IDOL_UP_FIX if flipped else torch.eye(3)
    upright = g.transformed(up, torch.zeros(3))

    facing, sym_err = cam_yaw, None
    if use_symmetry:
        theta, sym_err, spread = symmetry_plane(upright)
        log(f"[orient] symmetry plane yaw {theta:.1f} deg, error {sym_err:.4f} (median over planes {spread:.4f})")
        if sym_err < 0.6 * spread:
            # Two senses along the plane; keep the one closest to where the camera was.
            cands = [theta, theta + 180]
            facing = min(cands, key=lambda a: abs((a - cam_yaw + 180) % 360 - 180)) % 360
            method += "+symmetry"
    # Rotate about +Y so the facing direction maps onto +Z.
    yaw_fix = axis_angle_matrix([0.0, 1.0, 0.0], -facing)
    R = yaw_fix @ up
    rotated = g.transformed(R, torch.zeros(3))

    sel = rotated.means[rotated.opacities > 0.2]
    lo = torch.quantile(sel.cpu(), 0.002, dim=0).to(g.device)
    hi = torch.quantile(sel.cpu(), 0.998, dim=0).to(g.device)
    scale = 1.0 if height is None else height / float(hi[1] - lo[1])
    t = -scale * torch.stack([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    result = rotated.transformed(torch.eye(3), t, scale)
    o = Orientation(rotation=R.tolist(), translation=t.tolist(), scale=float(scale), flipped=bool(flipped),
                    camera_yaw_deg=float(cam_yaw), facing_yaw_deg=float(facing),
                    symmetry_error=None if sym_err is None else float(sym_err),
                    match_score=None if score is None else float(score), method=method)
    log(f"[orient] facing yaw {facing:.1f} deg ({method}); height {(hi[1] - lo[1]) * scale:.3f}")
    return result, o
