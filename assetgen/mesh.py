"""Gaussian splat -> closed, decimated, UV-textured triangle mesh.

1. Render the splat from cameras spread over the sphere (RGB, alpha, median depth).
2. Fuse into a dense truncated signed distance volume. Pixels the splat does not
   cover carve the voxels they see as empty, voxels never seen in front of a
   surface stay inside, so the zero level set is closed.
3. Marching cubes, drop small floating islands, Taubin smoothing, quadric
   decimation to a face budget.
4. xatlas UV atlas, then a texture baked by projecting every texel into the
   views that see it unoccluded (depth test against the median depth maps),
   weighted towards frontal views. Texels no view sees take the colour of the
   nearest Gaussians. Chart borders are dilated so mipmaps do not bleed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .splats import Camera, Gaussians, Render, framing_radius, orbit_cameras, render


@dataclass
class MeshSettings:
    n_views: int = 64
    render_size: int = 768
    fov_deg: float = 30.0
    resolution: int = 256          # voxels along the longest side
    trunc_voxels: float = 4.0
    carve_alpha: float = 0.05      # below this coverage a pixel is empty space
    surface_alpha: float = 0.5     # above this its median depth is trusted
    min_island: float = 0.02       # islands smaller than this share of the largest are dropped
    smooth_iters: int = 8
    target_faces: int = 40000
    texture_size: int = 2048
    view_power: float = 4.0


@dataclass
class TexturedMesh:
    vertices: np.ndarray   # (V, 3) float32
    faces: np.ndarray      # (F, 3) int32
    uvs: np.ndarray        # (V, 2) float32, glTF convention (v down)
    texture: np.ndarray    # (S, S, 3) uint8


def splat_bounds(g: Gaussians, min_opacity: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust axis-aligned bounds of the visible part of the splat."""
    pts = g.means[g.opacities > min_opacity]
    if len(pts) < 100:
        pts = g.means
    lo = torch.quantile(pts.float().cpu(), 0.001, dim=0)
    hi = torch.quantile(pts.float().cpu(), 0.999, dim=0)
    return lo.to(g.device), hi.to(g.device)


def capture_views(g: Gaussians, s: MeshSettings, log=print) -> tuple[list[Camera], list[Render]]:
    lo, hi = splat_bounds(g)
    center = (lo + hi) / 2
    radius = framing_radius(float((hi - lo).norm()) / 2, s.fov_deg)
    cams = orbit_cameras(center.cpu(), radius, s.n_views, s.fov_deg, s.render_size)
    renders = []
    for i, cam in enumerate(cams):
        renders.append(render(g, cam))
        if (i + 1) % 16 == 0 or i + 1 == len(cams):
            log(f"  rendered {i + 1}/{len(cams)} views")
    return cams, renders


def _project(points: torch.Tensor, cam: Camera):
    R = cam.view[:3, :3].to(points)
    t = cam.view[:3, 3].to(points)
    pc = points @ R.T + t
    z = pc[:, 2]
    zs = z.clamp_min(1e-6)
    u = cam.fx * pc[:, 0] / zs + cam.cx
    v = cam.fy * pc[:, 1] / zs + cam.cy
    return u, v, z


def _sample(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    """Sample an (H, W) or (H, W, C) image at pixel coordinates (u, v)."""
    H, W = image.shape[:2]
    img = image if image.dim() == 3 else image[..., None]
    grid = torch.stack([u / W * 2 - 1, v / H * 2 - 1], dim=-1).view(1, 1, -1, 2)
    out = F.grid_sample(img.permute(2, 0, 1)[None], grid, mode=mode, align_corners=False)
    out = out[0, :, 0].T
    return out if image.dim() == 3 else out[:, 0]


def fuse_tsdf(cams: list[Camera], renders: list[Render], lo: torch.Tensor, hi: torch.Tensor,
              s: MeshSettings, chunk: int = 1 << 22):
    """Return (sdf grid (X, Y, Z) numpy, origin, voxel size); negative is inside."""
    dev = renders[0].depth.device
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    voxel = float((hi - lo).max()) / s.resolution
    dims = torch.ceil((hi - lo) / voxel).long() + 1
    trunc = s.trunc_voxels * voxel
    gx, gy, gz = (torch.arange(int(n), device=dev, dtype=torch.float32) for n in dims)
    n_vox = int(dims.prod())
    sdf = torch.zeros(n_vox, device=dev)
    weight = torch.zeros(n_vox, device=dev)
    for start in range(0, n_vox, chunk):
        ids = torch.arange(start, min(start + chunk, n_vox), device=dev)
        iz = ids % dims[2]
        iy = (ids // dims[2]) % dims[1]
        ix = ids // (dims[1] * dims[2])
        pts = lo + voxel * torch.stack([gx[ix], gy[iy], gz[iz]], dim=1)
        acc = torch.zeros(len(ids), device=dev)
        wsum = torch.zeros(len(ids), device=dev)
        for cam, r in zip(cams, renders):
            u, v, z = _project(pts, cam)
            inside = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            a = _sample(r.alpha, u, v)
            d = _sample(r.depth, u, v)
            empty = inside & (a < s.carve_alpha)
            surf = inside & (a > s.surface_alpha) & (d > 0)
            diff = d - z
            seen = surf & (diff > -trunc)
            val = torch.where(empty, torch.ones_like(z), (diff / trunc).clamp(-1, 1))
            w = (empty | seen).float()
            acc += w * val
            wsum += w
        sdf[ids] = torch.where(wsum > 0, acc / wsum.clamp_min(1e-6), -torch.ones_like(acc))
        weight[ids] = wsum
    grid = sdf.view(*[int(n) for n in dims]).cpu().numpy()
    return grid, lo.cpu().numpy(), voxel


def extract_surface(grid: np.ndarray, origin: np.ndarray, voxel: float):
    from skimage.measure import marching_cubes

    padded = np.pad(grid, 1, constant_values=1.0)  # close the surface at the volume border
    verts, faces, _, _ = marching_cubes(padded, level=0.0, spacing=(voxel, voxel, voxel))
    verts = verts - voxel + origin
    # skimage winds faces so normals point towards increasing values (outside here) when using
    # a gradient-direction convention; make outward orientation explicit below.
    return verts.astype(np.float64), faces.astype(np.int64)


def clean_mesh(verts: np.ndarray, faces: np.ndarray, s: MeshSettings, log=print):
    import trimesh

    mesh = trimesh.Trimesh(verts, faces, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        areas = np.array([p.area for p in parts])
        keep = [p for p, a in zip(parts, areas) if a >= s.min_island * areas.max()]
        log(f"  kept {len(keep)} of {len(parts)} connected parts")
        mesh = trimesh.util.concatenate(keep)
    if mesh.volume < 0:
        mesh.invert()
    if s.smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=s.smooth_iters)
    if len(mesh.faces) > s.target_faces:
        import fast_simplification

        v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64),
                                            target_count=s.target_faces)
        mesh = trimesh.Trimesh(v, f, process=True)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def unwrap(vertices: np.ndarray, faces: np.ndarray, texture_size: int):
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = texture_size
    pack.padding = 4
    pack.bilinear = True
    pack.blockAlign = True
    atlas.generate(xatlas.ChartOptions(), pack)
    vmap, new_faces, uvs = atlas[0]
    return vmap.astype(np.int64), new_faces.astype(np.int64), uvs.astype(np.float32)


def rasterize_uv(uvs: torch.Tensor, faces: torch.Tensor, size: int, chunk: int = 1 << 15):
    """Per texel: face index (-1 if none) and barycentric weights. UVs in [0, 1], v up."""
    dev = uvs.device
    face_of = torch.full((size * size,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros(size * size, 3, device=dev)
    tri = uvs[faces] * size                                                     # (F, 3, 2) texel units
    for s in range(0, len(faces), chunk):
        t = tri[s:s + chunk]
        x0 = t[..., 0].min(1).values.floor().clamp(0, size - 1).long()
        x1 = t[..., 0].max(1).values.ceil().clamp(0, size - 1).long()
        y0 = t[..., 1].min(1).values.floor().clamp(0, size - 1).long()
        y1 = t[..., 1].max(1).values.ceil().clamp(0, size - 1).long()
        nx = x1 - x0 + 1
        count = nx * (y1 - y0 + 1)
        fid = torch.repeat_interleave(torch.arange(len(t), device=dev), count)
        start = torch.cumsum(count, 0) - count
        local = torch.arange(len(fid), device=dev) - start[fid]
        px = (x0[fid] + local % nx[fid]).float() + 0.5
        py = (y0[fid] + local // nx[fid]).float() + 0.5
        a, b, c = t[fid, 0], t[fid, 1], t[fid, 2]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        good = den.abs() > 1e-12
        den = torch.where(good, den, torch.ones_like(den))
        w0 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / den
        w1 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / den
        w2 = 1 - w0 - w1
        hit = good & (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        texel = (py.long() * size + px.long())[hit]
        face_of[texel] = fid[hit] + s
        bary[texel] = torch.stack([w0, w1, w2], dim=1)[hit].clamp_min(0)
    bary = bary / bary.sum(1, keepdim=True).clamp_min(1e-8)
    return face_of, bary


def bake_texture(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray, normals: np.ndarray,
                 cams: list[Camera], renders: list[Render], g: Gaussians, s: MeshSettings,
                 depth_tol: float, chunk: int = 1 << 18) -> np.ndarray:
    dev = renders[0].rgb.device
    size = s.texture_size
    V = torch.as_tensor(vertices, dtype=torch.float32, device=dev)
    Fc = torch.as_tensor(faces, dtype=torch.long, device=dev)
    N = torch.as_tensor(normals, dtype=torch.float32, device=dev)
    UV = torch.as_tensor(uvs, dtype=torch.float32, device=dev)
    face_of, bary = rasterize_uv(UV, Fc, size)
    texels = torch.nonzero(face_of >= 0).squeeze(1)
    color = torch.zeros(size * size, 3, device=dev)
    known = torch.zeros(size * size, dtype=torch.bool, device=dev)
    cam_pos = [c.position.to(dev) for c in cams]

    for start in range(0, len(texels), chunk):
        tx = texels[start:start + chunk]
        f = Fc[face_of[tx]]
        w = bary[tx]
        p = (V[f] * w[..., None]).sum(1)
        n = F.normalize((N[f] * w[..., None]).sum(1), dim=-1)
        acc = torch.zeros(len(tx), 3, device=dev)
        wsum = torch.zeros(len(tx), device=dev)
        for cam, r, cp in zip(cams, renders, cam_pos):
            u, v, z = _project(p, cam)
            inside = (z > 0) & (u >= 0.5) & (u < cam.width - 0.5) & (v >= 0.5) & (v < cam.height - 0.5)
            d = _sample(r.depth, u, v)
            a = _sample(r.alpha, u, v)
            visible = inside & (a > s.surface_alpha) & ((d - z).abs() < depth_tol)
            cosine = (F.normalize(cp - p, dim=-1) * n).sum(-1)
            wt = torch.where(visible & (cosine > 0.05), cosine.clamp_min(0) ** s.view_power, torch.zeros_like(z))
            acc += wt[:, None] * _sample(r.rgb, u, v, mode="bilinear")
            wsum += wt
        seen = wsum > 1e-6
        color[tx[seen]] = acc[seen] / wsum[seen, None]
        known[tx[seen]] = True

    # Texels no view sees: colour of the nearest opaque Gaussians.
    missing = texels[~known[texels]]
    if len(missing):
        from scipy.spatial import cKDTree

        sel = g.opacities > 0.2
        tree = cKDTree(g.means[sel].cpu().numpy())
        gcol = g.colors[sel].cpu().numpy()
        f = Fc[face_of[missing]]
        p = (V[f] * bary[missing][..., None]).sum(1).cpu().numpy()
        dist, nn = tree.query(p, k=8)
        wt = 1.0 / np.maximum(dist, 1e-6)
        fill = (gcol[nn] * wt[..., None]).sum(1) / wt.sum(1, keepdims=True)
        color[missing] = torch.as_tensor(fill, dtype=torch.float32, device=dev)
        known[missing] = True

    tex = dilate(color.view(size, size, 3), known.view(size, size), iterations=16)
    # Rasterised with v up; images are stored top row first.
    return (tex.flip(0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def dilate(image: torch.Tensor, mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Grow known texels into the gutters by averaging known neighbours."""
    img = (image * mask[..., None]).permute(2, 0, 1)[None]
    m = mask.float()[None, None]
    kernel = torch.ones(1, 1, 3, 3, device=image.device)
    for _ in range(iterations):
        s = F.conv2d(img, kernel.expand(3, 1, 3, 3), padding=1, groups=3)
        c = F.conv2d(m, kernel, padding=1)
        grow = (c > 0) & (m == 0)
        img = torch.where(grow, s / c.clamp_min(1), img)
        m = torch.where(grow, torch.ones_like(m), m)
    return img[0].permute(1, 2, 0)


def splat_to_mesh(g: Gaussians, s: MeshSettings | None = None, log=print) -> TexturedMesh:
    s = s or MeshSettings()
    log(f"[mesh] rendering {s.n_views} views at {s.render_size}px")
    cams, renders = capture_views(g, s, log)
    lo, hi = splat_bounds(g)
    log(f"[mesh] fusing at {s.resolution} voxels across the longest side")
    grid, origin, voxel = fuse_tsdf(cams, renders, lo, hi, s)
    verts, faces = extract_surface(grid, origin, voxel)
    log(f"[mesh] marching cubes: {len(verts)} vertices, {len(faces)} faces")
    mesh = clean_mesh(verts, faces, s, log)
    log(f"[mesh] cleaned and decimated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    vmap, ufaces, uvs = unwrap(mesh.vertices, mesh.faces, s.texture_size)
    vertices = mesh.vertices[vmap]
    normals = mesh.vertex_normals[vmap]
    log(f"[mesh] UV atlas: {len(vertices)} vertices after seams; baking {s.texture_size}px texture")
    texture = bake_texture(vertices, ufaces, uvs, normals, cams, renders, g, s, depth_tol=3 * voxel)
    uv_gltf = np.stack([uvs[:, 0], 1 - uvs[:, 1]], axis=1)
    return TexturedMesh(vertices.astype(np.float32), ufaces.astype(np.int32), uv_gltf.astype(np.float32), texture)


def export_glb(m: TexturedMesh, path) -> None:
    import trimesh
    from PIL import Image

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(m.texture), metallicFactor=0.0, roughnessFactor=0.9, doubleSided=False)
    # trimesh takes OpenGL-style UVs (v up) and flips them into glTF's convention on export.
    uv_gl = np.stack([m.uvs[:, 0], 1 - m.uvs[:, 1]], axis=1)
    mesh = trimesh.Trimesh(m.vertices, m.faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv_gl, material=material))
    mesh.export(str(path))
