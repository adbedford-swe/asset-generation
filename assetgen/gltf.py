"""Read the textured mesh back from a .glb, and write a skinned, textured .glb."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from .mesh import TexturedMesh


@dataclass
class Rig:
    joints: np.ndarray        # (J, 3) rest positions, world space
    parents: np.ndarray       # (J,) parent index, -1 for the root
    names: list[str]
    weights: np.ndarray       # (V, J) skin weights, rows sum to 1


def load_textured_glb(path) -> TexturedMesh:
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    geoms = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        g = scene.geometry[name].copy()
        g.apply_transform(transform)
        geoms.append(g)
    if len(geoms) != 1:
        raise ValueError(f"{path}: expected one mesh, found {len(geoms)}")
    m = geoms[0]
    material = m.visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    uv = np.asarray(m.visual.uv, dtype=np.float32)
    return TexturedMesh(
        vertices=np.asarray(m.vertices, dtype=np.float32),
        faces=np.asarray(m.faces, dtype=np.int32),
        uvs=np.stack([uv[:, 0], 1 - uv[:, 1]], axis=1),  # trimesh flips v on load; back to glTF
        texture=np.array(image.convert("RGB")),
    )


def top_k_weights(weights: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    J = weights.shape[1]
    if J < k:  # glTF skins always carry four influences per vertex
        weights = np.pad(weights, ((0, 0), (0, k - J)))
    idx = np.argsort(-weights, axis=1)[:, :k]
    w = np.take_along_axis(weights, idx, axis=1).clip(min=0)
    idx = np.where(idx < J, idx, 0)  # padding influences must still name a real joint
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-8)
    return idx.astype(np.uint16), w.astype(np.float32)


def write_rigged_glb(mesh: TexturedMesh, rig: Rig, path, name: str = "asset") -> None:
    import pygltflib as gl
    from PIL import Image

    V = mesh.vertices.astype(np.float32)
    Fc = mesh.faces.astype(np.uint32)
    normals = _vertex_normals(V, Fc)
    joint_idx, joint_w = top_k_weights(rig.weights)

    blob = bytearray()
    views, accessors = [], []

    def add(data: np.ndarray, target, comp, typ, minmax=False):
        data = np.ascontiguousarray(data)
        while len(blob) % 4:
            blob.append(0)
        views.append(gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=data.nbytes, target=target))
        blob.extend(data.tobytes())
        acc = gl.Accessor(bufferView=len(views) - 1, componentType=comp, count=len(data), type=typ)
        if minmax:
            acc.min = data.min(0).tolist()
            acc.max = data.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    a_pos = add(V, gl.ARRAY_BUFFER, gl.FLOAT, gl.VEC3, minmax=True)
    a_nrm = add(normals, gl.ARRAY_BUFFER, gl.FLOAT, gl.VEC3)
    a_uv = add(mesh.uvs.astype(np.float32), gl.ARRAY_BUFFER, gl.FLOAT, gl.VEC2)
    a_jnt = add(joint_idx, gl.ARRAY_BUFFER, gl.UNSIGNED_SHORT, gl.VEC4)
    a_wgt = add(joint_w, gl.ARRAY_BUFFER, gl.FLOAT, gl.VEC4)
    a_idx = add(Fc.reshape(-1), gl.ELEMENT_ARRAY_BUFFER, gl.UNSIGNED_INT, gl.SCALAR)

    J = len(rig.joints)
    ibm = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    ibm[:, :3, 3] = -rig.joints
    a_ibm = add(ibm.transpose(0, 2, 1).reshape(J, 16), None, gl.FLOAT, gl.MAT4)  # column-major

    png = io.BytesIO()
    Image.fromarray(mesh.texture).save(png, format="PNG")
    while len(blob) % 4:
        blob.append(0)
    views.append(gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(png.getvalue())))
    blob.extend(png.getvalue())
    image_view = len(views) - 1

    nodes = []
    children = [[] for _ in range(J)]
    for j, p in enumerate(rig.parents):
        if p >= 0:
            children[p].append(j)
    for j in range(J):
        p = rig.parents[j]
        local = rig.joints[j] - (rig.joints[p] if p >= 0 else 0)
        nodes.append(gl.Node(name=rig.names[j], translation=[float(x) for x in local], children=children[j]))
    roots = [j for j in range(J) if rig.parents[j] < 0]
    mesh_node = len(nodes)
    nodes.append(gl.Node(name=name, mesh=0, skin=0))
    armature = len(nodes)
    nodes.append(gl.Node(name="Armature", children=roots))

    doc = gl.GLTF2(
        asset=gl.Asset(generator="assetgen"),
        scene=0,
        scenes=[gl.Scene(nodes=[armature, mesh_node])],
        nodes=nodes,
        meshes=[gl.Mesh(name=name, primitives=[gl.Primitive(
            attributes=gl.Attributes(POSITION=a_pos, NORMAL=a_nrm, TEXCOORD_0=a_uv, JOINTS_0=a_jnt, WEIGHTS_0=a_wgt),
            indices=a_idx, material=0)])],
        skins=[gl.Skin(name="Armature", joints=list(range(J)), inverseBindMatrices=a_ibm, skeleton=roots[0])],
        materials=[gl.Material(name=name, pbrMetallicRoughness=gl.PbrMetallicRoughness(
            baseColorTexture=gl.TextureInfo(index=0), metallicFactor=0.0, roughnessFactor=0.9))],
        textures=[gl.Texture(source=0, sampler=0)],
        samplers=[gl.Sampler(magFilter=gl.LINEAR, minFilter=gl.LINEAR_MIPMAP_LINEAR)],
        images=[gl.Image(bufferView=image_view, mimeType="image/png")],
        accessors=accessors,
        bufferViews=views,
        buffers=[gl.Buffer(byteLength=len(blob))],
    )
    doc.set_binary_blob(bytes(blob))
    doc.save_binary(str(path))


def _vertex_normals(V: np.ndarray, Fc: np.ndarray) -> np.ndarray:
    """Area-weighted normals, shared by vertices that UV seams split at the same position."""
    _, weld = np.unique(V, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    fn = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])
    vn = np.zeros((weld.max() + 1, 3))
    for k in range(3):
        np.add.at(vn, weld[Fc[:, k]], fn)
    vn = vn[weld]
    return (vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)).astype(np.float32)
