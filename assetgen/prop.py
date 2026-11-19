"""Finalize a static asset: a prop (a barrel, a tree, a rock) or a diorama (a whole set piece).

Characters are rigged and animated after the mesh stage; static assets only need to
be made ready to place. This stage scales the mesh to the height asked for, puts its
pivot at the centre of its base (on y = 0, so it stands where it is placed), and
works out what a game or a scene layout needs to know about it:

* a collider: a convex hull for props, the mesh itself for dioramas (you walk on
  those), or a box; engines build it on import;
* a footprint: the outline of the asset seen from above, its radius, and the radius
  of its base (a tree's trunk rather than its canopy), so layouts can space assets
  without them overlapping;

and writes ``<name>.glb``, the prop file ``<name>.assetgen`` (format ``assetgen.prop``,
schema ``schema/prop.schema.json``) and a four-view preview ``<name>.png``.

    python -m assetgen.prop --mesh run/1_mesh/mesh.glb --out run/barrel.glb --height 1.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .mesh import TexturedMesh

FORMAT = "assetgen.prop"
VERSION = 1
COLLIDERS = ("auto", "convex", "box", "mesh", "none")
SCHEMA_URL = "https://raw.githubusercontent.com/adbedford-swe/asset-generation/main/schema/prop.schema.json"
BASE_SHARE = 0.1  # the slice of the asset's height that counts as its base


def place_on_ground(m: TexturedMesh, height: float | None = None) -> tuple[TexturedMesh, float]:
    """Scale to ``height`` (metres) and move the pivot to the centre of the base, on y = 0.
    Returns the mesh and the scale applied."""
    v = m.vertices.astype(np.float64)
    lo, hi = v.min(0), v.max(0)
    scale = 1.0 if height is None else float(height) / max(float(hi[1] - lo[1]), 1e-9)
    offset = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    placed = ((v - offset) * scale).astype(np.float32)
    return TexturedMesh(placed, m.faces, m.uvs, m.texture), scale


def outline(points: np.ndarray, max_points: int = 16) -> np.ndarray:
    """The convex outline of 2D points (counter-clockwise), with at most ``max_points`` corners: the
    corners whose removal changes the area least go first."""
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return np.asarray(points, dtype=np.float64)
    try:
        hull = points[ConvexHull(points).vertices]
    except Exception:  # degenerate (all on a line)
        lo, hi = points.min(0), points.max(0)
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    hull = list(map(tuple, hull))
    while len(hull) > max_points:
        def lost(i):
            (ax, ay), (bx, by), (cx, cy) = hull[i - 1], hull[i], hull[(i + 1) % len(hull)]
            return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        del hull[min(range(len(hull)), key=lost)]
    return np.array(hull)


def footprint(vertices: np.ndarray) -> dict:
    """The asset seen from above: outline (x, z), its radius about the pivot, and the base's radius."""
    v = np.asarray(vertices, dtype=np.float64)
    xz = v[:, [0, 2]]
    height = float(v[:, 1].max() - v[:, 1].min())
    base = xz[v[:, 1] <= v[:, 1].min() + BASE_SHARE * max(height, 1e-9)]
    poly = outline(xz)
    return {"outline": [[round(float(x), 4), round(float(z), 4)] for x, z in poly],
            "radius": round(float(np.linalg.norm(xz, axis=1).max()), 4),
            "base_radius": round(float(np.linalg.norm(base, axis=1).max()) if len(base) else 0.0, 4)}


def collider_for(kind: str, collider: str, vertices: np.ndarray) -> dict:
    """How engines should collide with the asset. Shapes are built on import from the model; the box is
    given so anything can fall back to it."""
    if collider not in COLLIDERS:
        raise ValueError(f"unknown collider {collider!r}; one of {', '.join(COLLIDERS)}")
    if collider == "auto":
        collider = "mesh" if kind == "diorama" else "convex"
    lo, hi = vertices.min(0), vertices.max(0)
    return {"type": collider,
            "box": {"center": [round(float(c), 4) for c in (lo + hi) / 2],
                    "size": [round(float(s), 4) for s in hi - lo]}}


def preview(m: TexturedMesh, size: int = 384) -> np.ndarray:
    """Four views around the asset, from slightly above (uint8 image)."""
    import torch

    from .preview import render_mesh, turntable_cameras

    lo, hi = m.vertices.min(0), m.vertices.max(0)
    center = torch.tensor((lo + hi) / 2, dtype=torch.float32)
    radius = float(np.linalg.norm(hi - lo)) / 2 / math.sin(math.radians(15) * 0.9)
    views = []
    for cam in turntable_cameras(center, radius, 4, size, elevation_deg=20.0, offset_deg=-30.0):
        rgb, _ = render_mesh(m.vertices, m.faces, cam, m.uvs, m.texture)
        views.append((rgb.numpy() * 255).astype(np.uint8))
    return np.concatenate(views, axis=1)


def _version() -> str:
    from .character import _version as version

    return version()


def describe(glb: Path, m: TexturedMesh, kind: str, collider: dict, source: dict | None = None) -> dict:
    """The prop file for ``glb`` (holding ``m``)."""
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    data = {
        "$schema": SCHEMA_URL,
        "format": FORMAT,
        "version": VERSION,
        "name": Path(glb).stem,
        "kind": kind,
        "generator": {"name": "assetgen", "version": _version()},
        "model": {
            "file": Path(glb).name, "units": "meters", "up": "+Y", "forward": "+Z", "pivot": "base",
            "height": round(float(hi[1] - lo[1]), 4),
            "size": [round(float(s), 4) for s in hi - lo],
            "bounds": {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi]},
            "vertices": int(len(m.vertices)), "triangles": int(len(m.faces)),
            "texture": list(m.texture.shape[1::-1]),
        },
        "collider": collider,
        "footprint": footprint(m.vertices),
        "static": True,
    }
    if source:
        data["source"] = {k: v for k, v in source.items() if v not in (None, "", [])}
    if Path(glb).with_suffix(".png").exists():
        data["preview"] = Path(glb).with_suffix(".png").name
    return data


def finalize(mesh_glb: Path, out: Path, kind: str = "prop", height: float | None = None, collider: str = "auto",
             source: dict | None = None, log=print) -> Path:
    """mesh.glb -> ``out`` (.glb), its prop file and preview. Returns the prop file's path."""
    from PIL import Image

    from .character import run_source
    from .gltf import load_textured_glb
    from .mesh import export_glb

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m, scale = place_on_ground(load_textured_glb(mesh_glb), height)
    export_glb(m, out, name=out.stem)
    Image.fromarray(preview(m)).save(out.with_suffix(".png"))
    if source is None:
        source = run_source(out.parent)
    data = describe(out, m, kind, collider_for(kind, collider, m.vertices), source)
    path = out.with_suffix(".assetgen")
    path.write_text(json.dumps(data, indent=2))
    size = data["model"]["size"]
    log(f"[prop] {out.name}: {kind}, {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m (scaled x{scale:.3g}), "
        f"{data['model']['triangles']} triangles, {data['collider']['type']} collider, "
        f"footprint radius {data['footprint']['radius']:.2f} m")
    return path


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not an assetgen prop file")
    if data.get("version", 0) > VERSION:
        raise ValueError(f"{path} is prop format v{data['version']}; this assetgen reads up to v{VERSION}")
    return data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", required=True, help="the mesh stage's mesh.glb")
    ap.add_argument("--out", required=True, help="<name>.glb to write")
    ap.add_argument("--kind", default="prop", choices=["prop", "diorama"])
    ap.add_argument("--height", type=float, default=None, help="height in metres (default: as meshed)")
    ap.add_argument("--collider", default="auto", choices=COLLIDERS)
    a = ap.parse_args(argv)
    finalize(Path(a.mesh), Path(a.out), a.kind, a.height, a.collider)


if __name__ == "__main__":
    main()
