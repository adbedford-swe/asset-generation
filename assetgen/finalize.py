"""Stage 4: one engine-ready .glb holding the rigged, textured asset and every clip.

The animated clips share one canonical asset, so the first clip's file is kept
and the other clips' animations are copied into it, with channels re-targeted
by node name. Each animation is named after its prompt. Optionally the
armature is scaled so the asset stands a given height.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def read_accessor(doc, blob: bytes, index: int) -> np.ndarray:
    acc = doc.accessors[index]
    view = doc.bufferViews[acc.bufferView]
    n = COMPONENTS[acc.type]
    item = BYTES[acc.componentType] * n
    stride = view.byteStride or item
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (acc.count - 1) + item, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(acc.count, item), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(DTYPES[acc.componentType]).reshape(acc.count, n)


def append_accessor(doc, blob: bytearray, data: np.ndarray, like) -> int:
    import pygltflib as gl

    while len(blob) % 4:
        blob.append(0)
    raw = np.ascontiguousarray(data).tobytes()
    doc.bufferViews.append(gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(raw)))
    blob.extend(raw)
    acc = gl.Accessor(bufferView=len(doc.bufferViews) - 1, componentType=like.componentType,
                      count=len(data), type=like.type, normalized=like.normalized)
    if like.min is not None:
        acc.min = data.min(0).tolist()
        acc.max = data.max(0).tolist()
    doc.accessors.append(acc)
    return len(doc.accessors) - 1


def merge_clips(clips: list[dict], out_path: Path, height: float | None = None, log=print) -> dict:
    import pygltflib as gl

    base = gl.GLTF2.load_binary(clips[0]["glb"])
    blob = bytearray(base.binary_blob())
    node_by_name = {n.name: i for i, n in enumerate(base.nodes)}
    used = set()

    def unique(name):
        k, out = 1, name
        while out in used:
            k += 1
            out = f"{name}_{k}"
        used.add(out)
        return out

    from .animate import clip_name

    animations = []
    for ci, clip in enumerate(clips):
        doc = base if ci == 0 else gl.GLTF2.load_binary(clip["glb"])
        src_blob = blob if ci == 0 else doc.binary_blob()
        for anim in doc.animations:
            if ci == 0:
                anim.name = unique(clip_name(clip["prompt"]))
                animations.append(anim)
                continue
            samplers, channels = [], []
            for s in anim.samplers:
                inp = append_accessor(base, blob, read_accessor(doc, src_blob, s.input), doc.accessors[s.input])
                out = append_accessor(base, blob, read_accessor(doc, src_blob, s.output), doc.accessors[s.output])
                samplers.append(gl.AnimationSampler(input=inp, output=out, interpolation=s.interpolation))
            for c in anim.channels:
                target = node_by_name.get(doc.nodes[c.target.node].name)
                if target is None:
                    raise ValueError(f"{clip['glb']}: node {doc.nodes[c.target.node].name!r} not in the base asset")
                channels.append(gl.AnimationChannel(sampler=c.sampler, target=gl.AnimationChannelTarget(
                    node=target, path=c.target.path)))
            animations.append(gl.Animation(name=unique(clip_name(clip["prompt"])), samplers=samplers,
                                           channels=channels))
    base.animations = animations

    info = {"clips": [a.name for a in animations]}
    if height is not None:
        info["scale"] = _scale_to_height(base, blob, height)
    base.buffers[0].byteLength = len(blob)
    base.set_binary_blob(bytes(blob))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save_binary(str(out_path))
    log(f"[finalize] {out_path}: {len(animations)} clips {info['clips']}")
    return info


def _scale_to_height(doc, blob, height: float) -> float:
    """Scale the scene's root nodes so the rest-pose mesh is ``height`` tall."""
    lo, hi = np.inf, -np.inf
    for mesh in doc.meshes:
        for p in mesh.primitives:
            acc = doc.accessors[p.attributes.POSITION]
            if acc.min is not None:
                lo, hi = min(lo, acc.min[1]), max(hi, acc.max[1])
            else:
                y = read_accessor(doc, bytes(blob), p.attributes.POSITION)[:, 1]
                lo, hi = min(lo, y.min()), max(hi, y.max())
    s = height / float(hi - lo)
    animated = {c.target.node for a in doc.animations for c in a.channels}
    for i in doc.scenes[doc.scene or 0].nodes:
        node = doc.nodes[i]
        if i in animated:
            raise ValueError(f"root node {node.name!r} is animated; cannot scale it")
        node.scale = [v * s for v in (node.scale or [1.0, 1.0, 1.0])]
        node.translation = [v * s for v in (node.translation or [0.0, 0.0, 0.0])]
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", required=True, help="clips.json written by the animate stage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=float, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    merge_clips(json.loads(Path(a.clips).read_text()), out, a.height)
    from PIL import Image

    from .gltf_anim import contact_sheet

    Image.fromarray(contact_sheet(out)).save(out.with_suffix(".png"))
    print(f"[finalize] clip contact sheet: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
