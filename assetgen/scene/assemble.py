"""Put a laid-out scene together: one glTF, the props it is made of, and the scene file.

``<name>.glb`` holds the ground and every placed object. Each object's mesh is stored
once and every placement is a node pointing at it (glTF instancing), so a forest of
forty pines costs one pine. Nodes are named ``<object>_<k>``, the ground ``ground``.
``assets/`` holds each object as a prop of its own (``<id>.glb`` + ``<id>.assetgen``,
textures harmonized like the scene's), for engines to make prefabs from. The scene
file ``<name>.assetgen`` (format ``assetgen.scene``) lists the objects and every
placement, so an engine can give each node its collider, or rebuild the scene from
the props.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from ..mesh import TexturedMesh
from .layout import Layout
from .spec import SceneSpec
from .terrain import ground_mesh

FORMAT = "assetgen.scene"
VERSION = 1
SCHEMA_URL = "https://raw.githubusercontent.com/adbedford-swe/asset-generation/main/schema/scene.schema.json"


def instance_matrix(position, rotation, scale) -> np.ndarray:
    import trimesh

    x, y, z, w = rotation
    m = trimesh.transformations.quaternion_matrix([w, x, y, z])
    m[:3, :3] *= scale
    m[:3, 3] = position
    return m


def ground_textured(layout: Layout, spec: SceneSpec, texture: np.ndarray | None) -> TexturedMesh:
    vertices, faces, _, uvs = ground_mesh(layout.ground, spec.ground.tile)
    if texture is None:
        texture = np.full((4, 4, 3), np.clip(np.array(spec.ground.color) * 255, 0, 255), dtype=np.uint8)
        uvs = np.full_like(uvs, 0.5)
    return TexturedMesh(vertices, faces.astype(np.int32), uvs, texture)


def write_glb(path: Path, spec: SceneSpec, layout: Layout, meshes: dict[str, TexturedMesh],
              ground: TexturedMesh) -> None:
    import trimesh

    from ..mesh import to_trimesh

    scene = trimesh.Scene()
    scene.add_geometry(to_trimesh(ground, "ground"), node_name="ground", geom_name="ground")
    added = set()
    for inst in layout.instances:
        matrix = instance_matrix(inst.position, inst.rotation, inst.scale)
        if inst.object not in added:
            scene.add_geometry(to_trimesh(meshes[inst.object], inst.object), node_name=inst.node,
                               geom_name=inst.object, transform=matrix)
            added.add(inst.object)
        else:
            scene.graph.update(frame_to=inst.node, frame_from=scene.graph.base_frame, matrix=matrix,
                               geometry=inst.object)
    scene.export(str(path))


def preview(spec: SceneSpec, layout: Layout, meshes: dict[str, TexturedMesh], ground: TexturedMesh,
            size: int = 640) -> np.ndarray:
    """The scene from above at three quarters, beside a view straight down (uint8 image)."""
    import torch

    from ..preview import render_mesh
    from ..splats import make_camera

    width, depth = spec.ground.size
    top = max((i.position[1] + meshes[i.object].vertices[:, 1].max() * i.scale for i in layout.instances),
              default=1.0)
    radius = math.hypot(width, depth) / 2
    distance = 1.1 * radius / math.sin(math.radians(20))  # the whole ground fits a 40 degree view
    oblique = np.array([0.5, 0.62, 0.6]) / np.linalg.norm([0.5, 0.62, 0.6])
    views = [(torch.tensor(oblique * distance, dtype=torch.float32), (0.0, 1.0, 0.0)),
             (torch.tensor([0.0, distance + top, 1e-3]), (0.0, 0.0, -1.0))]
    panels = []
    for eye, up in views:
        cam = make_camera(eye, torch.tensor([0.0, top * 0.15, 0.0]), 40.0, size, up=up)
        rgb, mask, depth_buf = render_mesh(ground.vertices, ground.faces, cam, ground.uvs, ground.texture,
                                           background=(0.86, 0.89, 0.93), wrap=True, return_depth=True)
        for inst in layout.instances:
            m = meshes[inst.object]
            T = instance_matrix(inst.position, inst.rotation, inst.scale)
            v = m.vertices @ T[:3, :3].T + T[:3, 3]
            r, k, d = render_mesh(v, m.faces, cam, m.uvs, m.texture, return_depth=True)
            nearer = (k > 0.5) & (d < depth_buf)
            rgb[nearer] = r[nearer]
            depth_buf = torch.where(nearer, d, depth_buf)
        panels.append((rgb.clamp(0, 1).numpy() * 255).astype(np.uint8))
    return np.concatenate(panels, axis=1)


def _version() -> str:
    from ..character import _version as version

    return version()


def describe(path: Path, spec: SceneSpec, layout: Layout, objects: dict[str, dict], ground_texture: str | None,
             looks: dict | None = None) -> dict:
    counts = {}
    for inst in layout.instances:
        counts[inst.object] = counts.get(inst.object, 0) + 1
    heights = layout.ground.heights
    data = {
        "$schema": SCHEMA_URL,
        "format": FORMAT,
        "version": VERSION,
        "name": spec.name,
        "generator": {"name": "assetgen", "version": _version()},
        "model": {"file": Path(path).name, "units": "meters", "up": "+Y", "forward": "+Z",
                  "size": [spec.ground.size[0], spec.ground.size[1]]},
        "style": spec.style,
        "ground": {"node": "ground", "size": spec.ground.size, "collider": "mesh",
                   "height_range": [round(float(heights.min()), 4), round(float(heights.max()), 4)],
                   "texture": ground_texture, "tile": spec.ground.tile},
        "objects": [],
        "instances": [{"object": i.object, "node": i.node, "position": i.position, "rotation": i.rotation,
                       "yaw": i.yaw, "scale": i.scale, "placed": i.placed} for i in layout.instances],
    }
    for o in spec.objects:
        prop = objects[o.id]
        entry = {"id": o.id, "kind": prop["kind"], "file": f"assets/{o.id}.glb", "asset": f"assets/{o.id}.assetgen",
                 "height": prop["model"]["height"], "collider": prop["collider"]["type"],
                 "box": prop["collider"]["box"],
                 "footprint": {"radius": prop["footprint"]["radius"],
                               "base_radius": prop["footprint"]["base_radius"]},
                 "count": counts.get(o.id, 0)}
        if o.prompt:
            entry["prompt"] = o.prompt
        if looks and o.id in looks:
            entry["look"] = looks[o.id]
        data["objects"].append(entry)
    if layout.warnings:
        data["warnings"] = layout.warnings
    if spec.concept:
        data["source"] = {"concept": Path(spec.concept).name}
    preview_path = Path(path).with_suffix(".png")
    if preview_path.exists():
        data["preview"] = preview_path.name
    return data


def write_assets(folder: Path, objects: dict[str, dict], meshes: dict[str, TexturedMesh], sources: dict[str, Path]):
    """``assets/<id>.glb`` and ``.assetgen`` for each object: its prop, with the scene's textures."""
    from ..mesh import export_glb

    assets = Path(folder) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for object_id, data in objects.items():
        export_glb(meshes[object_id], assets / f"{object_id}.glb", name=object_id)
        data = dict(data, name=object_id, model=dict(data["model"], file=f"{object_id}.glb"))
        preview = sources[object_id].with_suffix(".png")
        data.pop("preview", None)
        if preview.exists():
            shutil.copy2(preview, assets / f"{object_id}.png")
            data["preview"] = f"{object_id}.png"
        (assets / f"{object_id}.assetgen").write_text(json.dumps(data, indent=2))


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not an assetgen scene file")
    if data.get("version", 0) > VERSION:
        raise ValueError(f"{path} is scene format v{data['version']}; this assetgen reads up to v{VERSION}")
    return data
