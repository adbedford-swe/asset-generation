"""Textured-mesh rasterizer for previews and for scoring a mesh against its splat."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .splats import Camera, Gaussians, make_camera, render


def render_mesh(vertices, faces, cam: Camera, uvs=None, texture=None, vertex_colors=None,
                background=(1.0, 1.0, 1.0), shade: bool = True, chunk: int = 1 << 20, wrap: bool = False,
                return_depth: bool = False, return_faces: bool = False, light=None):
    """Z-buffered rasterization (on the GPU when there is one). Returns (rgb (H, W, 3), mask (H, W)) on CPU,
    then the depth along the view (inf where nothing was drawn) with ``return_depth``, then each pixel's
    face (-1 where nothing was drawn) and its barycentric coordinates (H, W, 3) with ``return_faces``.
    ``wrap`` repeats the texture for UVs outside [0, 1], as a tiling ground's are. ``light`` (a world
    direction towards the light) shades every surface, textured or not, by a directional light."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = torch.as_tensor(np.asarray(vertices), dtype=torch.float32, device=dev)
    Fc = torch.as_tensor(np.asarray(faces), dtype=torch.long, device=dev)
    H, W = cam.height, cam.width
    R, t = cam.view[:3, :3].to(dev), cam.view[:3, 3].to(dev)
    pc = V @ R.T + t
    z = pc[:, 2].clamp_min(1e-6)
    sx = cam.fx * pc[:, 0] / z + cam.cx
    sy = cam.fy * pc[:, 1] / z + cam.cy
    tri = torch.stack([sx, sy], dim=1)[Fc]
    tz = pc[:, 2][Fc]
    keep = (tz > 1e-3).all(1)

    zbuf = torch.full((H * W,), float("inf"), device=dev)
    fbuf = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    bbuf = torch.zeros(H * W, 3, device=dev)
    idx = torch.nonzero(keep).squeeze(1)
    x0 = tri[idx, :, 0].min(1).values.floor().clamp(0, W - 1).long()
    x1 = tri[idx, :, 0].max(1).values.ceil().clamp(0, W - 1).long()
    y0 = tri[idx, :, 1].min(1).values.floor().clamp(0, H - 1).long()
    y1 = tri[idx, :, 1].max(1).values.ceil().clamp(0, H - 1).long()
    nx = x1 - x0 + 1
    count = nx * (y1 - y0 + 1)
    start_all = torch.cumsum(count, 0) - count
    # Process faces in groups whose pixel candidates fit in `chunk`.
    bounds = torch.searchsorted(start_all, torch.arange(0, int(count.sum()) + chunk, chunk, device=dev))
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        if b <= a:
            continue
        c = count[a:b]
        fid = torch.repeat_interleave(torch.arange(a, b, device=dev), c)
        local = torch.arange(len(fid), device=dev) - (torch.cumsum(c, 0) - c)[fid - a]
        px = (x0[fid] + local % nx[fid]).float() + 0.5
        py = (y0[fid] + local // nx[fid]).float() + 0.5
        f = idx[fid]
        p0, p1, p2 = tri[f, 0], tri[f, 1], tri[f, 2]
        den = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
        ok = den.abs() > 1e-12
        den = torch.where(ok, den, torch.ones_like(den))
        w0 = ((p1[:, 1] - p2[:, 1]) * (px - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (py - p2[:, 1])) / den
        w1 = ((p2[:, 1] - p0[:, 1]) * (px - p2[:, 0]) + (p0[:, 0] - p2[:, 0]) * (py - p2[:, 1])) / den
        w2 = 1 - w0 - w1
        inside = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        # Perspective-correct depth.
        inv = w0 / tz[f, 0] + w1 / tz[f, 1] + w2 / tz[f, 2]
        depth = 1 / inv.clamp_min(1e-9)
        pix = (py.long() * W + px.long())[inside]
        d = depth[inside]
        zbuf.scatter_reduce_(0, pix, d, reduce="amin")
        win = d <= zbuf[pix]
        fbuf[pix[win]] = f[inside][win]
        bw = torch.stack([w0 / tz[f, 0], w1 / tz[f, 1], w2 / tz[f, 2]], 1)[inside][win] * d[win, None]
        bbuf[pix[win]] = bw

    mask = fbuf >= 0
    rgb = torch.tensor(background, dtype=torch.float32, device=dev).repeat(H * W, 1)
    hit = torch.nonzero(mask).squeeze(1)
    f = Fc[fbuf[hit]]
    w = bbuf[hit]
    if texture is not None and uvs is not None:
        UV = torch.as_tensor(np.asarray(uvs), dtype=torch.float32, device=dev)
        uv = (UV[f] * w[..., None]).sum(1)
        if wrap:
            uv = uv.remainder(1.0)
        tex = torch.as_tensor(np.asarray(texture), dtype=torch.float32, device=dev) / 255.0
        grid = (uv * 2 - 1).view(1, 1, -1, 2)  # glTF uv: v down, matches image rows
        col = F.grid_sample(tex.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False,
                            padding_mode="border" if wrap else "zeros")[0, :, 0].T
    elif vertex_colors is not None:
        VC = torch.as_tensor(np.asarray(vertex_colors), dtype=torch.float32, device=dev)
        col = (VC[f] * w[..., None]).sum(1)
    else:
        col = torch.full((len(hit), 3), 0.8, device=dev)
    if light is not None:
        n = F.normalize(torch.linalg.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]]), dim=-1)
        n = torch.where(((n @ R[2]) > 0)[:, None], -n, n)  # face the camera
        to_light = F.normalize(torch.as_tensor(light, dtype=torch.float32, device=dev), dim=0)
        col = col * (0.3 + 0.7 * (n @ to_light).clamp_min(0)[:, None])
    elif shade and texture is None and vertex_colors is None:
        n = F.normalize(torch.linalg.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]]), dim=-1)
        col = col * (0.35 + 0.65 * (n @ R[2]).abs()[:, None])
    rgb[hit] = col
    out = [rgb.view(H, W, 3).cpu(), mask.view(H, W).float().cpu()]
    if return_depth:
        out.append(zbuf.view(H, W).cpu())
    if return_faces:
        out += [fbuf.view(H, W).cpu(), bbuf.view(H, W, 3).cpu()]
    return tuple(out)


def turntable_cameras(center, radius: float, n: int = 6, size: int = 384, elevation_deg: float = 10.0,
                      fov_deg: float = 30.0, offset_deg: float = 0.0):
    cams = []
    for i in range(n):
        yaw = math.radians(offset_deg + 360.0 * i / n)
        el = math.radians(elevation_deg)
        eye = torch.as_tensor(center, dtype=torch.float32) + radius * torch.tensor(
            [math.sin(yaw) * math.cos(el), math.sin(el), math.cos(yaw) * math.cos(el)])
        cams.append(make_camera(eye, center, fov_deg, size))
    return cams


def compare_mesh_to_splat(g: Gaussians, vertices, faces, uvs, texture, n: int = 8, size: int = 384):
    """Colour PSNR where both silhouettes overlap and silhouette IoU over a turntable
    offset from the capture cameras. Returns (metrics dict, contact-sheet image uint8)."""
    v = np.asarray(vertices)
    lo, hi = v.min(0), v.max(0)
    center = torch.tensor((lo + hi) / 2, dtype=torch.float32)
    radius = float(np.linalg.norm(hi - lo)) / 2 / math.sin(math.radians(15) * 0.9)
    cams = turntable_cameras(center, radius, n, size, elevation_deg=12.0, offset_deg=11.0)
    psnrs, ious, rows = [], [], []
    for cam in cams:
        sr = render(g, cam, background=(1.0, 1.0, 1.0))
        sr = type(sr)(sr.rgb.cpu(), sr.alpha.cpu(), sr.depth.cpu())
        mr, mm = render_mesh(vertices, faces, cam, uvs, texture)
        sm = sr.alpha > 0.5
        union = sm | (mm > 0.5)
        both = sm & (mm > 0.5)
        mse = ((sr.rgb.clamp(0, 1) - mr) ** 2).mean(-1)[both].mean()
        psnrs.append(float(-10 * torch.log10(mse.clamp_min(1e-10))))
        ious.append(float((sm & (mm > 0.5)).sum() / union.sum().clamp_min(1)))
        rows.append(torch.cat([sr.rgb.clamp(0, 1), mr], dim=0))
    sheet = (torch.cat(rows, dim=1).numpy() * 255).astype(np.uint8)
    return {"psnr": float(np.mean(psnrs)), "iou": float(np.mean(ious)),
            "psnr_min": float(np.min(psnrs)), "iou_min": float(np.min(ious))}, sheet


def rig_preview(vertices, faces, uvs, texture, joints, parents, size: int = 512):
    """Front and side views of the mesh with the skeleton drawn over it (uint8 image)."""
    from PIL import Image, ImageDraw

    v = np.asarray(vertices)
    lo, hi = v.min(0), v.max(0)
    center = torch.tensor((lo + hi) / 2, dtype=torch.float32)
    radius = float(np.linalg.norm(hi - lo)) / 2 / math.sin(math.radians(15) * 0.9)
    panels = []
    for cam in turntable_cameras(center, radius, 4, size, elevation_deg=0.0)[:2]:
        rgb, _ = render_mesh(vertices, faces, cam, uvs, texture)
        img = Image.fromarray((rgb.numpy() * 160 + 95).astype(np.uint8))  # faded, so bones stand out
        draw = ImageDraw.Draw(img)
        J = torch.as_tensor(np.asarray(joints), dtype=torch.float32)
        pc = J @ cam.view[:3, :3].T + cam.view[:3, 3]
        px = (cam.fx * pc[:, 0] / pc[:, 2] + cam.cx).tolist()
        py = (cam.fy * pc[:, 1] / pc[:, 2] + cam.cy).tolist()
        for j, p in enumerate(parents):
            if p >= 0:
                draw.line([(px[p], py[p]), (px[j], py[j])], fill=(20, 90, 220), width=3)
        for j in range(len(px)):
            colour = (220, 40, 40) if parents[j] < 0 else (250, 200, 30)
            draw.ellipse([px[j] - 4, py[j] - 4, px[j] + 4, py[j] + 4], fill=colour, outline=(0, 0, 0))
        panels.append(np.asarray(img))
    return np.concatenate(panels, axis=1)
