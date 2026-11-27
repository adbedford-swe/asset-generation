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


def _rest_world(doc, node: int) -> np.ndarray:
    """World matrix of ``node`` in the rest pose (from the nodes' own TRS)."""
    parent = {c: i for i, n in enumerate(doc.nodes) for c in (n.children or [])}
    chain = []
    while node is not None:
        chain.append(node)
        node = parent.get(node)
    m = np.eye(4)
    for i in reversed(chain):
        n = doc.nodes[i]
        x, y, z, w = n.rotation or [0.0, 0.0, 0.0, 1.0]
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        local = np.eye(4)
        local[:3, :3] = r * np.asarray(n.scale or [1.0, 1.0, 1.0])[None, :]
        local[:3, 3] = n.translation or [0.0, 0.0, 0.0]
        m = m @ local
    return m


def skeleton_root(doc) -> int:
    joints = set(doc.skins[0].joints)
    parent = {c: i for i, n in enumerate(doc.nodes) for c in (n.children or [])}
    return next(j for j in doc.skins[0].joints if parent.get(j) not in joints)


def loop_root_motion(doc, blob: bytearray, anim, in_place: bool) -> list[float]:
    """Horizontal root velocity of a looping clip (world units per second). With ``in_place`` the
    root's horizontal travel is removed, so the last frame equals the first in every channel."""
    root = skeleton_root(doc)
    parent = {c: i for i, n in enumerate(doc.nodes) for c in (n.children or [])}
    to_world = _rest_world(doc, parent[root])[:3, :3] if root in parent else np.eye(3)
    for ch in anim.channels:
        if ch.target.node != root or ch.target.path != "translation":
            continue
        sampler = anim.samplers[ch.sampler]
        times = read_accessor(doc, bytes(blob), sampler.input)[:, 0].astype(np.float64)
        local = read_accessor(doc, bytes(blob), sampler.output).astype(np.float64)
        duration = float(times[-1] - times[0]) or 1.0
        world = local @ to_world.T
        travel = world[-1] - world[0]
        travel[1] = 0.0  # height belongs to the motion (bobbing, crouching), not to travel
        velocity = (travel / duration).tolist()
        if in_place:
            ramp = ((times - times[0]) / duration)[:, None]
            local -= (ramp * travel[None, :]) @ np.linalg.inv(to_world).T
            sampler.output = append_accessor(doc, blob, local.astype(np.float32), doc.accessors[sampler.output])
        return velocity
    return [0.0, 0.0, 0.0]


def merge_clips(clips: list[dict], out_path: Path, height: float | None = None, root_motion: bool = False,
                log=print) -> dict:
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

    animations, sources = [], []
    for ci, clip in enumerate(clips):
        doc = base if ci == 0 else gl.GLTF2.load_binary(clip["glb"])
        src_blob = blob if ci == 0 else doc.binary_blob()
        for anim in doc.animations:
            if ci == 0:
                anim.name = unique(clip_name(clip["prompt"]))
                animations.append(anim)
                sources.append(clip)
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
            sources.append(clip)
    base.animations = animations

    velocities = {}
    for anim, clip in zip(animations, sources):
        if clip.get("loop"):
            velocities[anim.name] = loop_root_motion(base, blob, anim, in_place=not root_motion)
    info = {"clips": [a.name for a in animations]}
    scale = 1.0
    if height is not None:
        scale = info["scale"] = _scale_to_height(base, blob, height)
    for anim, clip in zip(animations, sources):
        extras = {"prompt": clip["prompt"], "loop": bool(clip.get("loop"))}
        if clip.get("loop"):
            extras["loop_frames"] = clip.get("loop_frames")
            extras["root_motion"] = "kept" if root_motion else "in_place"
            extras["root_velocity"] = [round(v * scale, 5) for v in velocities[anim.name]]
        extras.update(clip_timing(clip, anim.name, out_path))
        anim.extras = extras
    info["animations"] = {a.name: a.extras for a in animations}
    base.buffers[0].byteLength = len(blob)
    base.set_binary_blob(bytes(blob))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save_binary(str(out_path))
    log(f"[finalize] {out_path}: {len(animations)} clips {info['clips']}")
    return info


def clip_timing(clip: dict, name: str, out_path: Path) -> dict:
    """What a clip from video brings besides its animation: its frame rate, timed events (contacts,
    impacts), a one-shot's phases, and its sound, copied to ``sounds/<clip>.wav`` beside the asset."""
    import shutil

    out = {}
    for key in ("fps", "source", "events", "phases"):
        if clip.get(key) is not None:
            out[key] = clip[key]
    sound = clip.get("sound")
    if sound and Path(sound).exists():
        target = out_path.parent / "sounds" / f"{name}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(sound).resolve() != target.resolve():
            shutil.copy2(sound, target)
        out["sound"] = f"sounds/{name}.wav"
    return out


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


def ground(path: Path, log=print) -> float:
    """Raise or lower the asset so its feet meet y = 0 and save it again. Returns the shift.

    UniMate's canonical asset is not grounded, so the merged clips can stand well below the
    origin. The ground is the median over the looping clips' frames of the lowest vertex (the
    rest pose when there are no loops), so a walk's planted feet touch y = 0."""
    import pygltflib as gl

    from .gltf_anim import AnimatedGLB

    g = AnimatedGLB(path)
    lows = []
    for i, anim in enumerate(g.doc.animations):
        if (anim.extras or {}).get("loop"):
            end = g.duration(i)
            lows += [float(g.posed(i, end * k / 12)[:, 1].min()) for k in range(12)]
    rest = g.posed(None)
    floor = float(np.median(lows)) if lows else float(rest[:, 1].min())
    if abs(floor) < 1e-4 * float(np.ptp(rest[:, 1]) or 1.0):
        return 0.0
    doc = gl.GLTF2.load_binary(str(path))
    for i in doc.scenes[doc.scene or 0].nodes:
        node = doc.nodes[i]
        t = list(node.translation or [0.0, 0.0, 0.0])
        t[1] -= floor
        node.translation = t
    doc.save_binary(str(path))
    log(f"[finalize] grounded: moved the asset {-floor:+.3f} along y")
    return -floor


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", required=True, help="clips.json written by the animate stage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--root_motion", action="store_true",
                    help="keep the root travel of looping clips instead of making them loop in place")
    a = ap.parse_args(argv)
    out = Path(a.out)
    info = merge_clips(json.loads(Path(a.clips).read_text()), out, a.height, root_motion=a.root_motion)
    info["ground_offset"] = round(ground(out), 5)
    from PIL import Image

    from . import character
    from .gltf_anim import AnimatedGLB, contact_sheet, loop_seam

    g = AnimatedGLB(out)
    for i, anim_name in enumerate(g.animations):
        if info["animations"][anim_name].get("loop"):
            seam = loop_seam(g, i)
            info["animations"][anim_name]["seam"] = seam
            print(f"[finalize] loop {anim_name}: seam position {seam['position']:.1e} of the asset size, "
                  f"seam roughness {seam['roughness']:.2f} x the rest of the clip")
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    Image.fromarray(contact_sheet(out)).save(out.with_suffix(".png"))
    print(f"[finalize] clip contact sheet: {out.with_suffix('.png')}")
    print(f"[finalize] character file: {character.write(out, info)}")


if __name__ == "__main__":
    main()
