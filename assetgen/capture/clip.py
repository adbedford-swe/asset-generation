"""A fitted motion written onto the character's asset as a glTF animation, the way the motion stage's
clips are: a copy of the asset with one animation, which finalize merges by node name.

Every joint gets a rotation channel. The hub gets its translation; the travel goes on the skeleton's
root joint, which is where finalize reads a looping clip's speed from (and removes it, for loops that
play in place).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .motion import Motion
from .rig import matrix_to_quaternion


def _continuous(q: np.ndarray) -> np.ndarray:
    """Flip quaternions (T, ..., 4) so consecutive ones lie in the same hemisphere, for clean interpolation."""
    q = q.copy()
    for t in range(1, len(q)):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] *= -1
    return q


def write(asset: Path, rig, motion: Motion, fps: float, out: Path, name: str = "captured",
          extras: dict | None = None) -> Path:
    """``asset`` with ``motion`` as its only animation, saved to ``out``. ``rig`` was built from ``asset``."""
    import pygltflib as gl

    from ..finalize import append_accessor, read_accessor, skeleton_root

    doc = gl.GLTF2.load_binary(str(asset))
    blob = bytearray(doc.binary_blob())
    T = len(motion)
    times = (np.arange(T) / fps).astype(np.float32)[:, None]
    scalar = gl.Accessor(componentType=gl.FLOAT, type=gl.SCALAR, min=[0.0], max=[0.0])
    vec4 = gl.Accessor(componentType=gl.FLOAT, type=gl.VEC4)
    vec3 = gl.Accessor(componentType=gl.FLOAT, type=gl.VEC3)
    time_acc = append_accessor(doc, blob, times, scalar)

    samplers, channels = [], []

    def channel(node: int, path: str, values: np.ndarray, like) -> None:
        samplers.append(gl.AnimationSampler(input=time_acc, output=append_accessor(doc, blob, values.astype(np.float32),
                                                                                   like), interpolation="LINEAR"))
        channels.append(gl.AnimationChannel(sampler=len(samplers) - 1,
                                            target=gl.AnimationChannelTarget(node=node, path=path)))

    quats = _continuous(matrix_to_quaternion(rig.local_rotations(motion.rotation)))      # (T, J, 4)
    for j, node in enumerate(rig.nodes):
        channel(node, "rotation", quats[:, j], vec4)

    rest_t = rig.rest_translation.detach().cpu().double().numpy()
    hub_t = rest_t[rig.hub][None] + motion.translation.detach().cpu().double().numpy()
    root = skeleton_root(doc)
    root_j = rig.nodes.index(root)
    # Travel is a world offset; the root joint's translation is in its parent's frame, which nothing animates.
    to_parent = np.linalg.inv(rig.base[root_j].detach().cpu().double().numpy()[:3, :3])
    travel_local = motion.offsets() @ to_parent.T
    if root_j == rig.hub:
        channel(root, "translation", hub_t + travel_local, vec3)
    else:
        channel(rig.nodes[rig.hub], "translation", hub_t, vec3)
        channel(root, "translation", rest_t[root_j][None] + travel_local, vec3)

    doc.animations = [gl.Animation(name=name, samplers=samplers, channels=channels, extras=extras or {})]
    doc.buffers[0].byteLength = len(blob)
    doc.set_binary_blob(bytes(blob))
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save_binary(str(out))
    # Read back one channel to fail here, not in finalize, if the file came out wrong.
    check = gl.GLTF2.load_binary(str(out))
    read_accessor(check, check.binary_blob(), check.animations[0].samplers[0].output)
    return out
