"""Stage 1: 3DGS .ply -> textured watertight .glb.

Pipeline:
  1. Render N orbit views (RGB + expected-depth + poses) with the pure-PyTorch
     splat renderer (splat_render).
  2. Fuse the depth maps into a watertight mesh via Open3D ScalableTSDFVolume
     (marching cubes).
  3. UV-unwrap with xatlas.
  4. Bake an albedo texture by re-projecting the rendered RGB views onto the UV
     map (visibility-weighted multi-view projection).
  5. Export a self-contained .glb (mesh + UVs + texture) via trimesh.

Usage:
    python pipeline/mesh_extract.py --ply outputs/idol/fig-viking.ply \
        --out outputs/mesh --n_views 32 --image_size 512 --tex_size 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from splat_render import load_gaussians, orbit_cameras, render_views  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Render
# --------------------------------------------------------------------------- #
def render(ply_path: str, out_dir: Path, n_views: int, image_size: int,
           fov_deg: float, radius: float, device: str) -> None:
    g = load_gaussians(ply_path, device=device)
    center = torch.zeros(3, device=device)
    cams = orbit_cameras(center, radius, fov_deg=fov_deg, n_views=n_views,
                         image_size=image_size, device=device)
    render_views(g, cams, out_dir)
    print(f"[mesh] rendered {n_views} views -> {out_dir}")


# --------------------------------------------------------------------------- #
# 2. TSDF fusion -> mesh
# --------------------------------------------------------------------------- #
def tsdf_fuse(render_dir: Path, voxel_size: float, sdf_trunc: float):
    import open3d as o3d
    from open3d.pipelines import integration as o3d_integ
    from PIL import Image

    poses = json.loads((render_dir / "poses.json").read_text())
    volume = o3d_integ.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d_integ.TSDFVolumeColorType.RGB8,
    )
    for rec in poses:
        color_img = o3d.geometry.Image(np.asarray(Image.open(
            render_dir / f"{rec['label']}.png")))
        depth_mm = np.asarray(Image.open(
            render_dir / f"{rec['label']}_depth.png")).astype(np.float32)
        depth_img = o3d.geometry.Image(depth_mm)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1000.0, depth_trunc=3.0,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            width=rec["width"], height=rec["height"],
            fx=rec["intr"][0][0], fy=rec["intr"][1][1],
            cx=rec["intr"][0][2], cy=rec["intr"][1][2],
        )
        extr = np.linalg.inv(np.asarray(rec["view"], dtype=np.float64))  # cam->world
        volume.integrate(rgbd, intr, extr)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_non_manifold_edges()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    non_manifold = mesh.get_non_manifold_vertices()
    if len(non_manifold) > 0:
        mesh.remove_vertices_by_index(non_manifold)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    return mesh


# --------------------------------------------------------------------------- #
# 3. UV unwrap (xatlas)
# --------------------------------------------------------------------------- #
def unwrap_uv(vertices: np.ndarray, triangles: np.ndarray):
    import xatlas

    atlas = xatlas.Atlas()
    xatlas.add_mesh(atlas, vertices.astype(np.float32), triangles.astype(np.int32))
    xatlas.generate(atlas)
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    vertices, uvs, triangles = xatlas.update_mesh(
        atlas, vertices.astype(np.float32), uvs, triangles.astype(np.int32))
    return vertices, uvs, triangles


# --------------------------------------------------------------------------- #
# 4. Texture bake (multi-view projection)
# --------------------------------------------------------------------------- #
def bake_texture(vertices, triangles, uvs, render_dir: Path, tex_size: int) -> np.ndarray:
    """Visibility-weighted multi-view projection of the rendered RGB onto UVs.

    The only Python loop is over triangles; per-triangle barycentric rasterization,
    projection into every view, RGB gather and visibility weighting are batched
    numpy ops.
    """
    from PIL import Image

    poses = json.loads((render_dir / "poses.json").read_text())
    v_uv = uvs[triangles]            # (M,3,2)
    v_pos = vertices[triangles]      # (M,3,3)

    # Stacked per-view projection data.
    V_all = np.stack([np.asarray(r["view"], dtype=np.float64)[:3, :3] for r in poses])  # (V,3,3)
    t_all = np.stack([np.asarray(r["view"], dtype=np.float64)[:3, 3] for r in poses])   # (V,3)
    K00 = np.array([r["intr"][0][0] for r in poses])
    K11 = np.array([r["intr"][1][1] for r in poses])
    K02 = np.array([r["intr"][0][2] for r in poses])
    K12 = np.array([r["intr"][1][2] for r in poses])
    W = poses[0]["width"]; H = poses[0]["height"]
    rgb_all = np.stack([
        np.asarray(Image.open(render_dir / f"{r['label']}.png")).astype(np.float32) / 255.0
        for r in poses])                                                    # (V,H,W,3)
    rgb_flat = rgb_all.reshape(len(poses), -1, 3)
    cam_all = -np.einsum("vij,vj->vi", V_all, t_all)                        # (V,3)

    acc = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    wsum = np.zeros((tex_size, tex_size), dtype=np.float64)

    # UV bboxes per triangle.
    min_u = v_uv[..., 0].min(axis=1); max_u = v_uv[..., 0].max(axis=1)
    min_v = v_uv[..., 1].min(axis=1); max_v = v_uv[..., 1].max(axis=1)
    x0 = np.clip((min_u * tex_size).astype(int), 0, tex_size - 1)
    x1 = np.clip((max_u * tex_size).astype(int) + 1, 0, tex_size)
    y0 = np.clip((min_v * tex_size).astype(int), 0, tex_size - 1)
    y1 = np.clip((max_v * tex_size).astype(int) + 1, 0, tex_size)

    p0 = v_uv[:, 0]; p1 = v_uv[:, 1]; p2 = v_uv[:, 2]
    den = ((p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0])
           + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1]))                # (M,)
    nrm = np.cross(v_pos[:, 1] - v_pos[:, 0], v_pos[:, 2] - v_pos[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)   # (M,3)

    view_idx = np.arange(len(poses))[None, :]                              # (1,V)

    for m in range(len(triangles)):
        if x1[m] <= x0[m] or y1[m] <= y0[m] or abs(den[m]) < 1e-12:
            continue
        tx = np.arange(x0[m], x1[m]); ty = np.arange(y0[m], y1[m])
        TX, TY = np.meshgrid(tx, ty, indexing="xy")
        u = (TX + 0.5) / tex_size; v = (TY + 0.5) / tex_size
        w2 = ((p1[m, 1] - p2[m, 1]) * (u - p2[m, 0]) + (p2[m, 0] - p1[m, 0]) * (v - p2[m, 1])) / den[m]
        w3 = ((p2[m, 1] - p0[m, 1]) * (u - p2[m, 0]) + (p0[m, 0] - p2[m, 0]) * (v - p2[m, 1])) / den[m]
        w1 = 1.0 - w2 - w3
        inside = (w1 >= -1e-4) & (w2 >= -1e-4) & (w3 >= -1e-4)
        if not inside.any():
            continue
        w1 = np.clip(w1[inside], 0, None).ravel()
        w2 = np.clip(w2[inside], 0, None).ravel()
        w3 = np.clip(w3[inside], 0, None).ravel()
        ty_in = TY[inside].ravel(); tx_in = TX[inside].ravel()
        pos = (w1[:, None] * v_pos[m, 0] + w2[:, None] * v_pos[m, 1]
               + w3[:, None] * v_pos[m, 2])                                # (P,3)
        pc = np.einsum("pd,vde->pve", pos, V_all) + t_all[None, :, :]      # (P,V,3)
        z = pc[:, :, 2]
        px = K00[None, :] * pc[:, :, 0] / np.maximum(z, 1e-6) + K02[None, :]
        py = K11[None, :] * pc[:, :, 1] / np.maximum(z, 1e-6) + K12[None, :]
        inimg = (z > 0.05) & (px >= 0) & (py >= 0) & (px < W - 1) & (py < H - 1)
        vdir = cam_all[None, :, :] - pos[:, None, :]
        vdir /= np.maximum(np.linalg.norm(vdir, axis=2, keepdims=True), 1e-12)
        vis = np.einsum("pvd,d->pv", vdir, nrm[m])                         # (P,V)
        vis = np.clip(np.where(inimg, vis, 0.0), 0, None)
        if vis.sum() <= 0:
            continue
        pxi = np.clip(px.astype(int), 0, W - 1)
        pyi = np.clip(py.astype(int), 0, H - 1)
        flat = pyi * W + pxi                                               # (P,V)
        rgb_s = rgb_flat[view_idx, flat]                                   # (P,V,3)
        per_texel = (vis[:, :, None] * rgb_s).sum(axis=1)                  # (P,3)
        w_per = vis.sum(axis=1)                                            # (P,)
        np.add.at(acc, (ty_in, tx_in), per_texel)
        np.add.at(wsum, (ty_in, tx_in), w_per)

    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    mask = wsum > 1e-6
    tex[mask] = acc[mask] / wsum[mask, None]
    return np.clip(tex, 0, 1)


# --------------------------------------------------------------------------- #
# 5. Export .glb
# --------------------------------------------------------------------------- #
def export_glb(vertices, triangles, uvs, texture, out_path: Path) -> None:
    import trimesh
    from PIL import Image

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    tex_pil = Image.fromarray((texture * 255).astype(np.uint8))
    material = trimesh.visual.material.SimpleMaterial(
        baseColorTexture=tex_pil, metallic=0.0, roughness=1.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh.export(str(out_path))
    print(f"[mesh] exported {out_path}  ({len(vertices)} verts, {len(triangles)} tris)")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", default="outputs/mesh")
    ap.add_argument("--n_views", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--fov_deg", type=float, default=40.0)
    ap.add_argument("--radius", type=float, default=1.6)
    ap.add_argument("--voxel_size", type=float, default=0.005)
    ap.add_argument("--sdf_trunc", type=float, default=0.02)
    ap.add_argument("--tex_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out)
    render_dir = out / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    render(args.ply, render_dir, args.n_views, args.image_size,
           args.fov_deg, args.radius, args.device)

    mesh = tsdf_fuse(render_dir, args.voxel_size, args.sdf_trunc)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    print(f"[mesh] TSDF mesh: {len(vertices)} verts, {len(triangles)} tris")

    vertices, uvs, triangles = unwrap_uv(vertices, triangles)
    print(f"[mesh] UV unwrapped: {uvs.shape}")

    texture = bake_texture(vertices, triangles, uvs, render_dir, args.tex_size)
    print(f"[mesh] texture baked: {texture.shape}")

    export_glb(vertices, triangles, uvs, texture, out / "mesh.glb")


if __name__ == "__main__":
    main()
