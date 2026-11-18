"""Stage 1: splat -> upright, front-facing, textured mesh (.glb), with a quality report."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np


def run(ply: Path, out_dir: Path, reference: Path | None, device: str, settings, height: float | None = None,
        log=print) -> dict:
    import torch
    from PIL import Image

    from .mesh import export_glb, splat_to_mesh
    from .orient import orient
    from .preview import compare_mesh_to_splat, render_mesh, turntable_cameras
    from .splats import load_ply, save_ply

    out_dir.mkdir(parents=True, exist_ok=True)
    g = load_ply(ply, device=device)
    log(f"[mesh] {len(g)} gaussians from {ply}")
    g, o = orient(g, str(reference) if reference else None, height=height, log=log)
    save_ply(g, out_dir / "canonical.ply")
    (out_dir / "orientation.json").write_text(json.dumps(o.to_dict(), indent=2))

    m = splat_to_mesh(g, settings, log)
    export_glb(m, out_dir / "mesh.glb")

    metrics, sheet = compare_mesh_to_splat(g, m.vertices, m.faces, m.uvs, m.texture)
    Image.fromarray(sheet).save(out_dir / "compare.png")
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    center = torch.tensor((lo + hi) / 2, dtype=torch.float32)
    radius = float(np.linalg.norm(hi - lo)) / 2 / math.sin(math.radians(15) * 0.9)
    front = render_mesh(m.vertices, m.faces, turntable_cameras(center, radius, 1, 512, 5.0)[0], m.uvs, m.texture)[0]
    Image.fromarray((front.numpy() * 255).astype(np.uint8)).save(out_dir / "front.png")
    report = {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
              "extent": (hi - lo).tolist(), **metrics, "orientation": o.to_dict(), "settings": asdict(settings)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"[mesh] {len(m.faces)} faces; vs splat: colour PSNR {metrics['psnr']:.2f} dB, "
        f"silhouette IoU {metrics['iou']:.3f}")
    return report


def main(argv=None) -> None:
    from .mesh import MeshSettings

    d = MeshSettings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="Idol's preprocessed input image, to recover up and facing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--height", type=float, default=None, help="scale so the asset is this tall")
    for name in ("n_views", "render_size", "resolution", "target_faces", "texture_size"):
        ap.add_argument(f"--{name}", type=int, default=getattr(d, name))
    a = ap.parse_args(argv)
    from .device import use_device

    use_device(a.device)
    s = MeshSettings(n_views=a.n_views, render_size=a.render_size, resolution=a.resolution,
                     target_faces=a.target_faces, texture_size=a.texture_size)
    run(Path(a.ply), Path(a.out), Path(a.reference) if a.reference else None, a.device, s, a.height)


if __name__ == "__main__":
    main()
