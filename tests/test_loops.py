import types

import numpy as np
import pygltflib as gl
import torch

from assetgen.animate import split_prompt
from assetgen.finalize import append_accessor, merge_clips
from assetgen.gltf import write_rigged_glb
from assetgen.gltf_anim import AnimatedGLB, loop_seam
from assetgen.motion_sampler import dominant_period, flow_sample, loop_length, loop_period, periodic
from tests.test_gltf import column_asset


def test_periodic_tie_is_exact():
    x = torch.randn(2, 3, 4, 60)
    y = periodic(x, [24, None])
    assert torch.equal(y[1], x[1])
    assert torch.allclose(y[0, ..., :36], y[0, ..., 24:], atol=1e-6)


def test_flow_sample_loops_the_window():
    """With any velocity field, tying the predicted clean motion makes the result periodic."""
    transport = types.SimpleNamespace(train_eps=0.0, sample_eps=0.0,
                                      check_interval=lambda *a, **k: (0.0, 1.0))
    target = torch.sin(torch.linspace(0, 7, 60)).expand(1, 2, 3, 60)

    def model(x, t, cond=None):  # flows straight to an aperiodic target
        return (target - x) / (1 - t[0] + 1e-3)

    out = flow_sample(model, transport, {}, torch.randn(1, 2, 3, 60), steps=16, periods=[40])
    assert torch.allclose(out[..., :20], out[..., 40:], atol=1e-5)
    plain = flow_sample(model, transport, {}, torch.randn(1, 2, 3, 60), steps=16)
    assert not torch.allclose(plain[..., :20], plain[..., 40:], atol=1e-2)


def test_dominant_period_and_loop_length():
    t = np.arange(60)
    walk = np.stack([np.sin(2 * np.pi * t / 20), np.cos(2 * np.pi * t / 20)], 1)
    assert dominant_period(walk, max_period=30) == 20
    assert loop_length(walk, 60, "auto") == 40          # two whole cycles, 20 frames of overlap
    noise = np.random.default_rng(0).normal(size=(60, 8))
    assert dominant_period(noise, max_period=30) is None
    assert loop_length(noise, 60, "auto") == 48         # no cycle: the window less the margin
    assert loop_length(walk, 60, 30) == 30


def test_loop_period_recovers_the_tie():
    base = np.random.default_rng(1).normal(size=(36, 5, 12))
    motion = np.concatenate([base, base[:24]])          # period 36 over 60 frames
    assert loop_period(motion) == 36
    assert loop_period(np.random.default_rng(2).normal(size=(60, 5, 12))) is None


def test_once_marker():
    assert split_prompt("An object attacks forward. [once]") == ("An object attacks forward.", False)
    assert split_prompt("An object walks forward.") == ("An object walks forward.", True)


def walking_clip(path, frames=31, speed=0.5, fps=30.0):
    """The column asset's root bone walking along +Z while its upper bone swings once per clip."""
    doc = gl.GLTF2.load_binary(str(path))
    blob = bytearray(doc.binary_blob())
    root = next(i for i, n in enumerate(doc.nodes) if n.name == "root")
    upper = next(i for i, n in enumerate(doc.nodes) if n.name == "upper")
    t = (np.arange(frames) / fps).astype(np.float32)[:, None]
    phase = 2 * np.pi * np.arange(frames) / (frames - 1)
    trans = np.stack([np.zeros(frames), 0.02 * np.sin(2 * phase), speed * t[:, 0]], 1).astype(np.float32)
    angle = 0.4 * np.sin(phase)
    rot = np.stack([np.zeros(frames), np.zeros(frames), np.sin(angle / 2), np.cos(angle / 2)], 1).astype(np.float32)
    ti = append_accessor(doc, blob, t, gl.Accessor(componentType=gl.FLOAT, type=gl.SCALAR, min=[0.0], max=[1.0]))
    tr = append_accessor(doc, blob, trans, gl.Accessor(componentType=gl.FLOAT, type=gl.VEC3))
    ro = append_accessor(doc, blob, rot, gl.Accessor(componentType=gl.FLOAT, type=gl.VEC4))
    doc.animations = [gl.Animation(name="walk", samplers=[gl.AnimationSampler(input=ti, output=tr),
                                                          gl.AnimationSampler(input=ti, output=ro)],
                                   channels=[gl.AnimationChannel(sampler=0, target=gl.AnimationChannelTarget(
                                                 node=root, path="translation")),
                                             gl.AnimationChannel(sampler=1, target=gl.AnimationChannelTarget(
                                                 node=upper, path="rotation"))])]
    doc.buffers[0].byteLength = len(blob)
    doc.set_binary_blob(bytes(blob))
    doc.save_binary(str(path))


def test_in_place_loop_and_recorded_speed(tmp_path):
    mesh, rig = column_asset()
    write_rigged_glb(mesh, rig, tmp_path / "w.glb")
    walking_clip(tmp_path / "w.glb")
    clips = [{"glb": str(tmp_path / "w.glb"), "prompt": "An object walks forward.", "loop": True, "loop_frames": 30}]
    info = merge_clips(clips, tmp_path / "out.glb", log=lambda *_: None)
    extras = info["animations"]["walks_forward"]
    assert extras["loop"] and extras["root_motion"] == "in_place"
    assert np.allclose(extras["root_velocity"], [0, 0, 0.5], atol=1e-4)
    g = AnimatedGLB(tmp_path / "out.glb")
    seam = loop_seam(g, 0)
    assert seam["position"] < 1e-5 and seam["roughness"] < 1.5
    # The same clip with its root travel kept does not close: it is root motion.
    kept = merge_clips(clips, tmp_path / "kept.glb", root_motion=True, log=lambda *_: None)
    assert kept["animations"]["walks_forward"]["root_motion"] == "kept"
    assert loop_seam(AnimatedGLB(tmp_path / "kept.glb"), 0)["position"] > 0.1
