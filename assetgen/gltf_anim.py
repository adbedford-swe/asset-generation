"""Evaluate a skinned, animated glTF on the CPU: posed vertices per frame, and contact sheets.

Used to check every clip after the motion and finalize stages without a 3D viewer.
"""
from __future__ import annotations

import io
import math

import numpy as np

from .finalize import read_accessor


def _quat_to_matrix(q):  # glTF order (x, y, z, w)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
    else:
        th = math.acos(d)
        q = (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)
    return q / np.linalg.norm(q)


class AnimatedGLB:
    def __init__(self, path):
        import pygltflib as gl
        from PIL import Image

        self.doc = doc = gl.GLTF2.load_binary(str(path))
        blob = doc.binary_blob()
        self.blob = blob
        self.parent = {c: i for i, n in enumerate(doc.nodes) for c in (n.children or [])}
        mesh_node = next(i for i, n in enumerate(doc.nodes) if n.mesh is not None and n.skin is not None)
        self.skin = doc.skins[doc.nodes[mesh_node].skin]
        prim = doc.meshes[doc.nodes[mesh_node].mesh].primitives[0]
        a = prim.attributes
        self.positions = read_accessor(doc, blob, a.POSITION).astype(np.float64)
        self.faces = read_accessor(doc, blob, prim.indices).reshape(-1, 3).astype(np.int64)
        self.uvs = read_accessor(doc, blob, a.TEXCOORD_0).astype(np.float32) if a.TEXCOORD_0 is not None else None
        self.joints = read_accessor(doc, blob, a.JOINTS_0).astype(np.int64)
        self.weights = read_accessor(doc, blob, a.WEIGHTS_0).astype(np.float64)
        self.ibm = read_accessor(doc, blob, self.skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.texture = None
        mat = doc.materials[prim.material] if prim.material is not None else None
        tex_info = mat.pbrMetallicRoughness.baseColorTexture if mat and mat.pbrMetallicRoughness else None
        if tex_info is not None:
            img = doc.images[doc.textures[tex_info.index].source]
            view = doc.bufferViews[img.bufferView]
            data = blob[view.byteOffset or 0:(view.byteOffset or 0) + view.byteLength]
            self.texture = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        self.animations = [a.name or f"anim_{i}" for i, a in enumerate(doc.animations)]

    def duration(self, anim: int) -> float:
        a = self.doc.animations[anim]
        return max(float(read_accessor(self.doc, self.blob, s.input).max()) for s in a.samplers)

    def _local_trs(self, anim: int | None, t: float):
        trs = {}
        for i, n in enumerate(self.doc.nodes):
            trs[i] = [np.array(n.translation or [0, 0, 0], float), np.array(n.rotation or [0, 0, 0, 1], float),
                      np.array(n.scale or [1, 1, 1], float)]
        if anim is None:
            return trs
        a = self.doc.animations[anim]
        for c in a.channels:
            s = a.samplers[c.sampler]
            times = read_accessor(self.doc, self.blob, s.input)[:, 0]
            vals = read_accessor(self.doc, self.blob, s.output).astype(float)
            k = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 1))
            k1 = min(k + 1, len(times) - 1)
            u = 0.0 if k1 == k else float(np.clip((t - times[k]) / (times[k1] - times[k]), 0, 1))
            if s.interpolation == "STEP":
                u = 0.0
            slot = {"translation": 0, "rotation": 1, "scale": 2}.get(c.target.path)
            if slot is None:
                continue
            if slot == 1:
                trs[c.target.node][1] = _slerp(vals[k], vals[k1], u)
            else:
                trs[c.target.node][slot] = vals[k] + u * (vals[k1] - vals[k])
        return trs

    def world_matrices(self, anim: int | None = None, t: float = 0.0) -> dict:
        trs = self._local_trs(anim, t)
        cache = {}

        def world(i):
            if i in cache:
                return cache[i]
            T, R, S = trs[i]
            m = np.eye(4)
            m[:3, :3] = _quat_to_matrix(R) * S[None, :]
            m[:3, 3] = T
            if i in self.parent:
                m = world(self.parent[i]) @ m
            cache[i] = m
            return m

        return {i: world(i) for i in range(len(self.doc.nodes))}

    def posed(self, anim: int | None = None, t: float = 0.0) -> np.ndarray:
        W = self.world_matrices(anim, t)
        skin = np.stack([W[j] for j in self.skin.joints]) @ self.ibm             # (J, 4, 4)
        M = (skin[self.joints] * self.weights[..., None, None]).sum(1)           # (V, 4, 4)
        p = np.concatenate([self.positions, np.ones((len(self.positions), 1))], 1)
        return np.einsum("vij,vj->vi", M, p)[:, :3]

    def joint_positions(self, anim: int | None = None, t: float = 0.0) -> np.ndarray:
        W = self.world_matrices(anim, t)
        return np.stack([W[j][:3, 3] for j in self.skin.joints])


def contact_sheet(path, frames: int = 6, size: int = 256, yaw_deg: float = 35.0) -> np.ndarray:
    """One row per animation: ``frames`` evenly spaced poses, seen from a fixed three-quarter camera."""
    import torch

    from .preview import render_mesh
    from .splats import make_camera

    g = AnimatedGLB(path)
    rows = []
    rest = g.posed(None)
    lo, hi = rest.min(0), rest.max(0)
    for ai in range(len(g.animations)):
        poses = [g.posed(ai, g.duration(ai) * k / max(1, frames - 1)) for k in range(frames)]
        allp = np.concatenate(poses)
        plo, phi = np.minimum(lo, allp.min(0)), np.maximum(hi, allp.max(0))
        center = torch.tensor((plo + phi) / 2, dtype=torch.float32)
        radius = float(np.linalg.norm(phi - plo)) / 2 / math.sin(math.radians(15) * 0.9)
        y = math.radians(yaw_deg)
        eye = center + radius * torch.tensor([math.sin(y) * 0.95, 0.3, math.cos(y) * 0.95])
        cam = make_camera(eye, center, 30.0, size)
        row = [render_mesh(p, g.faces, cam, g.uvs, g.texture, background=(0.86, 0.87, 0.9))[0].numpy() for p in poses]
        rows.append(np.concatenate(row, axis=1))
    return (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
