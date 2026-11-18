import types

import numpy as np
import pygltflib as gl
import torch

from assetgen.animate import split_prompt
from assetgen.finalize import append_accessor, merge_clips
from assetgen.gltf import write_rigged_glb
from assetgen.gltf_anim import AnimatedGLB, loop_seam
from assetgen.motion_sampler import (WINDOW, Strip, _select, dominant_period, loop_length, loop_period,
                                     strip_sample)
from tests.test_gltf import column_asset

TRANSPORT = types.SimpleNamespace(train_eps=0.0, sample_eps=0.0, check_interval=lambda *a, **k: (0.0, 1.0))


def smoothing_model(x, t, cond=None):
    """Flows each window towards a smoothed copy of itself: smooth output, no preferred phase."""
    k = torch.ones(1, 1, 9, device=x.device) / 9
    flat = x.reshape(-1, 1, x.shape[-1])
    smooth = torch.nn.functional.conv1d(torch.nn.functional.pad(flat, (4, 4), mode="replicate"), k)
    return (smooth.reshape(x.shape) - x) / (1 - t[:, None, None, None] + 0.05)


def second_difference(x):
    return (x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]).abs().amax(dim=(0, 1))


def test_long_ring_is_smooth_through_its_wrap():
    """A 150-frame ring (more than two windows) comes out as smooth across frame 149 -> 0 as anywhere."""
    torch.manual_seed(0)
    L = 150
    ring = strip_sample(smoothing_model, TRANSPORT, {}, [Strip(L, True)], [torch.randn(2, 3, L)], steps=24,
                        generator=torch.Generator().manual_seed(0))[0]
    assert ring.shape[-1] == L
    wrapped = torch.cat([ring[..., -3:], ring[..., :3]], -1)
    interior = second_difference(ring)
    assert float(second_difference(wrapped).max()) <= 2 * float(interior.quantile(0.95))


def test_single_window_strip_is_plain_euler():
    torch.manual_seed(1)
    noise = torch.randn(2, 3, WINDOW)
    out = strip_sample(smoothing_model, TRANSPORT, {}, [Strip(WINDOW, False)], [noise], steps=10)[0]
    x = noise.clone()[None]
    ts = torch.linspace(0, 1, 11)
    for i in range(10):
        x = x + (ts[i + 1] - ts[i]) * smoothing_model(x, torch.full((1,), float(ts[i])))
    assert torch.allclose(out, x[0], atol=1e-5)


def test_windows_cover_every_frame():
    ring = Strip(150, True)
    covered = {(st + j) % 150 for st in ring.starts for j in range(WINDOW)}
    assert covered == set(range(150)) and len(ring.starts) == 5
    assert Strip(120, False).starts == [0, 30, 60]
    assert Strip(60, False).starts == [0]


def test_loop_length_follows_the_cycle():
    t = np.arange(120)

    def with_slow_drift(sig):  # a large slow channel must not hide a small fast cycle
        return np.concatenate([sig * 0.05, 10 * np.linspace(0, 1, 120)[:, None]], 1)

    def cycle(period):
        return np.stack([np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)], 1)

    assert dominant_period(cycle(20)) == 20
    assert loop_length(cycle(20), "auto") == 20       # one cycle
    assert loop_length(cycle(15), "auto") == 30       # two cycles reach the 20-frame minimum
    assert loop_length(cycle(50), "auto") == 50       # longer than the old 48-frame cap
    assert dominant_period(with_slow_drift(cycle(24))) == 24
    noise = np.random.default_rng(0).normal(size=(120, 8))
    assert dominant_period(noise) is None
    assert loop_length(noise, "auto") == 90           # no cycle: three seconds
    assert loop_length(None, 200) == 200              # explicit length, beyond one window


def test_select_takes_per_sample_rows():
    cond = {"n_joints": torch.tensor([3, 5]), "caption": ["a", "b"], "scale": torch.tensor(2.0)}
    out = _select(cond, [1, 1, 0], 2)
    assert out["n_joints"].tolist() == [5, 5, 3] and out["caption"] == ["b", "b", "a"]
    assert out["scale"] is cond["scale"]


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
