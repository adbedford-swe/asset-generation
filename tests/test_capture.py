"""Motion from video (assetgen.capture), on a small creature whose true motion is known."""
import base64
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

from assetgen import animate, character, export
from assetgen.capture import bench, clip, motion, track
from assetgen.capture import run as capture_run
from assetgen.capture.camera import render_start, seed_points, start_camera
from assetgen.capture.fit import FitOptions, fit
from assetgen.capture.rig import (RigModel, axis_angle_to_matrix, from_animation, matrix_to_axis_angle,
                                  matrix_to_quaternion, quaternion_to_matrix)
from assetgen.capture.video import VideoSpec, h3_frames, video_prompt
from assetgen.finalize import ground, merge_clips
from assetgen.gltf import Rig, write_rigged_glb
from assetgen.gltf_anim import AnimatedGLB
from assetgen.mesh import TexturedMesh
from tests.test_character import add_travel_clip
from tests.test_gltf import column_asset
from tests.test_imagegen import StubImageServer, StubSdServer, catalog_files


def box(lo, hi):
    """Vertices and faces of an axis-aligned box, outward-facing."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def creature(path: Path) -> Path:
    """A body on two legs with an arm, each part rigid on its own bone; the body's joint (the hub) branches.
    Textured with noise, so every part of it is something to track."""
    parts = [((-0.3, 0.5, -0.15), (0.3, 0.9, 0.15), 0),     # body -> hub
             ((-0.25, 0.0, -0.08), (-0.1, 0.5, 0.08), 1),    # left leg
             ((0.1, 0.0, -0.08), (0.25, 0.5, 0.08), 2),      # right leg
             ((0.3, 0.75, -0.06), (0.75, 0.85, 0.06), 3)]    # arm
    verts, faces, owner = [], [], []
    for lo, hi, j in parts:
        v, f = box(lo, hi)
        faces.append(f + sum(len(x) for x in verts))
        verts.append(v)
        owner += [j] * len(v)
    v = np.concatenate(verts).astype(np.float32)
    f = np.concatenate(faces).astype(np.int32)
    weights = np.zeros((len(v), 4), np.float32)
    weights[np.arange(len(v)), owner] = 1
    rng = np.random.default_rng(0)
    uv = rng.random((len(v), 2)).astype(np.float32)
    texture = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    rig = Rig(np.array([[0, 0.5, 0], [-0.17, 0.5, 0], [0.17, 0.5, 0], [0.3, 0.8, 0]], np.float32),
              np.array([-1, 0, 0, 0]), ["hips", "leg_l", "leg_r", "arm"], weights)
    write_rigged_glb(TexturedMesh(v, f, uv, texture), rig, path)
    return path


def swing(rig: RigModel, frames: int, period: float = 16.0, travel: float = 0.0):
    """Legs swinging opposite each other about x, the arm waving about z; the hub moves by ``travel`` a frame."""
    t = torch.arange(frames, dtype=torch.float32)
    rot, tr = rig.zeros(frames)
    ramp = torch.clamp(t / 6, max=1)
    phase = torch.sin(2 * np.pi * t / period) * ramp
    j = {n: i for i, n in enumerate(rig.names)}
    rot[:, j["leg_l"], 0] = 0.5 * phase
    rot[:, j["leg_r"], 0] = -0.5 * phase
    rot[:, j["arm"], 2] = 0.6 * phase
    tr[:, 2] = travel * t
    return rot, tr


@pytest.fixture
def creature_glb(tmp_path):
    return creature(tmp_path / "creature.glb")


def test_rotation_conversions_round_trip():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(50, 3))
    w *= (rng.random((50, 1)) * 3.0) / np.linalg.norm(w, axis=1, keepdims=True)  # angles below pi: unique
    R = axis_angle_to_matrix(torch.as_tensor(w)).numpy()
    assert np.allclose(matrix_to_axis_angle(R), w, atol=1e-6)
    assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(R)), R, atol=1e-9)
    assert np.allclose(axis_angle_to_matrix(torch.zeros(3, dtype=torch.float64)).numpy(), np.eye(3))


def test_rig_model_is_the_gltf_skin(tmp_path):
    mesh, rig = column_asset()
    path = tmp_path / "c.glb"
    write_rigged_glb(mesh, rig, path)
    add_travel_clip(path, 0.4, lift=0.1)
    g = AnimatedGLB(path)
    model = RigModel(g, dtype=torch.float64)
    times = np.linspace(0, 1, 7)
    posed = model.pose_points(model.vertex_basis(), *from_animation(model, g, 0, times)).numpy()
    assert np.abs(posed - np.stack([g.posed(0, t) for t in times])).max() < 1e-6
    assert model.names[model.hub] == "root"


def test_hub_is_where_the_skeleton_branches(creature_glb):
    assert RigModel(AnimatedGLB(creature_glb)).names[0] == "hips"
    rig = RigModel(AnimatedGLB(creature_glb))
    assert rig.names[rig.hub] == "hips"


def _scene(glb, size=192):
    g = AnimatedGLB(glb)
    rig = RigModel(g)
    with torch.no_grad():
        rest = rig.pose_points(rig.vertex_basis(), *rig.zeros(1))[0].numpy()
    cam = start_camera(rest, size, size, margin=1.4)
    rgb, mask, face, bary = render_start(rest, rig.faces, cam, g.uvs, g.texture)
    return g, rig, cam, seed_points(mask, face, bary, 400, inset=1)


def test_seeds_lie_on_the_character(creature_glb):
    g, rig, cam, seeds = _scene(creature_glb)
    assert len(seeds.pixels) > 100
    assert np.allclose(seeds.bary.sum(1), 1, atol=1e-4) and (seeds.bary >= -1e-4).all()
    tracks, visible = track.exact(rig, *rig.zeros(1), seeds, cam)
    assert np.abs(tracks[0] - seeds.pixels).max() < 0.75   # a seed projects back onto its own pixel
    assert visible.mean() > 0.95


def test_fit_recovers_a_known_motion_from_its_tracks(creature_glb):
    g, rig, cam, seeds = _scene(creature_glb)
    truth = swing(rig, 24)
    tracks, visible = track.exact(rig, *truth, seeds, cam)
    result = fit(rig, seeds, tracks, visible, cam, FitOptions(iterations=250, refine=60), log=lambda *_: None)
    s = bench.score(rig, (result.rotation, result.translation), truth, cam)
    assert result.report["residual_px"]["mean"] < 1.0
    assert s["fit"]["vertex_mean"] < 0.5 * s["rest"]["vertex_mean"]
    assert s["fit"]["across_view"] < 0.01


def test_loop_takes_one_cycle_and_closes_it(creature_glb):
    rig = RigModel(AnimatedGLB(creature_glb))
    rot, tr = swing(rig, 90, period=20.0, travel=0.01)
    looped, cut = motion.loop(rig, motion.Motion(rot, tr, torch.zeros(90, 2)), fps=24, skip=0.3)
    assert cut["frames"] in (20, 40) and cut["seam"] < 0.05
    assert torch.allclose(looped.rotation[0], looped.rotation[-1], atol=1e-6)
    assert torch.allclose(looped.translation[0], looped.translation[-1], atol=1e-6)
    # the hub's 0.01 a frame along z is ground covered: it becomes travel, not a jump at the seam
    assert np.isclose(float(looped.travel[0, 1]), 0.01, atol=1e-4)


def test_contacts_and_footsteps(creature_glb):
    rig = RigModel(AnimatedGLB(creature_glb))
    rot, tr = swing(rig, 48, period=24.0)
    found = motion.contacts(rig, motion.Motion(rot, tr, torch.zeros(48, 2)), fps=24)
    assert {c["joint"] for c in found} <= {"leg_l", "leg_r", "hips"} and found
    steps = motion.footsteps(found)
    assert all(e["type"] == "contact" for e in steps) and steps == sorted(steps, key=lambda e: e["time"])


def test_phases_of_a_one_shot_and_sound_onsets(creature_glb):
    rig = RigModel(AnimatedGLB(creature_glb))
    T = 48
    rot, tr = rig.zeros(T)
    swing_up = torch.clamp((torch.arange(T) - 20) / 6, 0, 1)            # still, then a fast swing, then still
    rot[:, 3, 2] = 1.2 * swing_up
    m = motion.Motion(rot, tr, torch.zeros(T, 2))
    p = motion.phases(rig, m, fps=24)
    assert 18 / 24 <= p["active"][0] <= 21 / 24 and 24 / 24 <= p["active"][1] <= 27 / 24
    assert p["active"][0] <= p["impact"] <= p["active"][1] + 1 / 24
    rate = 16000
    audio = np.random.default_rng(0).normal(0, 0.002, rate * 2)
    for at in (0.5, 1.25):
        k = int(at * rate)
        audio[k:k + 400] += np.sin(np.arange(400) * 0.6) * np.linspace(1, 0, 400)
    onsets = motion.onsets(audio, rate)
    assert any(abs(o - 0.5) < 0.03 for o in onsets[:2]) and any(abs(o - 1.25) < 0.03 for o in onsets[:2])
    assert motion.phases(rig, m, fps=24, onsets=[1.0])["impact"] == 1.0


def test_a_clip_from_video_merges_with_others_into_the_character_file(creature_glb, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    rig = RigModel(AnimatedGLB(creature_glb))
    rot, tr = swing(rig, 24)
    m = motion.Motion(rot, tr, torch.zeros(24, 2))
    wave = tmp_path / "wave.wav"
    from assetgen.capture.video import write_audio

    write_audio(wave, np.zeros(16000, np.float32))
    first = clip.write(creature_glb, rig, m, 24, tmp_path / "a" / "walk.glb", name="walk")
    second = clip.write(creature_glb, rig, m, 24, tmp_path / "b" / "wave.glb", name="wave")
    reread = AnimatedGLB(first)
    with torch.no_grad():
        want = rig.pose_points(rig.vertex_basis(), rot[10:11], tr[10:11])[0].numpy()
    assert np.abs(reread.posed(0, 10 / 24) - want).max() < 1e-4
    clips = [{"glb": str(first), "prompt": "An object walks forward.", "loop": True, "loop_frames": 23, "fps": 24,
              "source": "video", "events": [{"time": 0.25, "type": "contact", "joint": "leg_l"}]},
             {"glb": str(second), "prompt": "An object waves.", "loop": False, "fps": 24, "source": "video",
              "phases": {"windup": [0, 0.2], "active": [0.2, 0.5], "recovery": [0.5, 0.96], "impact": 0.4},
              "events": [{"time": 0.4, "type": "impact"}], "sound": str(wave)}]
    out = tmp_path / "run" / "creature.glb"
    info = merge_clips(clips, out, log=lambda *_: None)
    info["ground_offset"] = ground(out, log=lambda *_: None)
    path = character.write(out, info)
    data = json.loads(path.read_text())
    jsonschema.validate(data, json.loads((export.ROOT / "schema" / "character.schema.json").read_text()))
    walk, wave_clip = data["clips"]
    assert walk["source"] == "video" and walk["fps"] == 24 and walk["frames"] == 24
    assert walk["events"][0]["joint"] == "leg_l"
    assert wave_clip["role"] == "action" and wave_clip["phases"]["impact"] == 0.4
    assert wave_clip["sound"] == "sounds/waves.wav" and (out.parent / "sounds" / "waves.wav").exists()
    assert "CoTracker3" in data["license"]["note"]
    exported = export.export(out, "godot", tmp_path / "exports")
    assert (exported / "sounds" / "waves.wav").exists()


def test_video_prompt_and_frame_counts():
    assert [h3_frames(s) for s in (0.1, 2.2, 4.2)] == [5, 56, 107]
    assert all((h3_frames(s) - 5) % 17 == 0 for s in np.linspace(0.2, 8, 30))
    text = video_prompt("An object jumps in place.", "a red fox")
    assert text.startswith("A red fox jumps in place.") and "static" in text


def test_video_tag_routes_a_prompt_to_capture():
    assert animate.split_prompt("An object roars. [once] [video]") == ("An object roars.", False)
    assert animate.split_prompt("An object walks forward. [video]") == ("An object walks forward.", True)
    assert animate.from_video("An object roars. [VIDEO] [once]")
    assert not animate.from_video("An object roars. [once]")


def test_pipeline_passes_video_options_only_with_video_prompts(tmp_path):
    from assetgen.pipeline import Options, stage_commands

    o = Options(out=tmp_path, image=tmp_path / "x.png", prompts=["An object walks forward."],
                video_args=["--video_url", "http://h:1"], text=None)
    assert "--video_url" not in stage_commands(o)["animate"][0]
    o.prompts.append("An object pounces. [once] [video]")
    anim = stage_commands(o)["animate"][0]
    assert anim[anim.index("--video_url") + 1] == "http://h:1"


class StubVideoServer(StubSdServer):
    """sd-server's native video API: a job that completes on the first poll."""

    def do_GET(self):
        if self.path.startswith("/sdcpp/v1/jobs/"):
            self._json({"id": "job_1", "kind": "vid_gen", "status": "completed",
                        "result": {"b64_json": base64.b64encode(b"WEBM").decode(), "mime_type": "video/webm",
                                   "output_format": "webm", "fps": 24, "frame_count": 56}})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/sdcpp/v1/vid_gen":
            length = int(self.headers.get("Content-Length") or 0)
            StubImageServer.requests.append((self.path, json.loads(self.rfile.read(length)), None))
            self._json({"id": "job_1", "kind": "vid_gen", "status": "queued", "poll_url": "/sdcpp/v1/jobs/job_1"},
                       202)
        else:
            super().do_POST()


H3_FILES = ["diffusion_models/minimax_h3_ref2va-Q4_K_M.gguf", "diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf",
            "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "vae/minimax_h3_audio_vae_fp32.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors"]


def test_sdcpp_video_job_selects_the_h3_files_and_unloads(tmp_path):
    from assetgen.capture.video import sdcpp_generate

    StubImageServer.requests = []
    StubSdServer.state = {"state": "empty", "recipe": None, "models": {}}
    StubSdServer.files = catalog_files(*H3_FILES)
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubVideoServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        start = tmp_path / "start.png"
        start.write_bytes(b"PNG")
        spec = VideoSpec(url=f"http://127.0.0.1:{server.server_address[1]}", steps=8)
        out = sdcpp_generate("a fox pounces", start, start, 56, spec, 3, tmp_path / "video.webm",
                             log=lambda *_: None, poll=0.01)
    finally:
        server.shutdown()
    assert out.read_bytes() == b"WEBM"
    paths = [p for p, _, _ in StubImageServer.requests]
    assert paths == ["/sdcpp/v1/vid_gen", "/sdcpp/v1/models/unload"]
    body = StubImageServer.requests[0][1]
    assert body["init_image"] == body["end_image"] == base64.b64encode(b"PNG").decode()
    assert body["video_frames"] == 56 and body["fps"] == 24 and body["sample_params"]["sample_steps"] == 8
    assert body["model"] == {"recipe": "minimax-h3-fl2va", "models": {
        "diffusion_model": "diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf",
        "llm": "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "vae": "vae/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors"}}


def test_command_backend_fills_the_template(tmp_path):
    from assetgen.capture.video import command_generate

    start = tmp_path / "start.png"
    start.write_bytes(b"PNG")
    script = tmp_path / "make.py"
    script.write_text("import shutil, sys\nshutil.copy(sys.argv[1], sys.argv[2])\n"
                      "open(sys.argv[2] + '.args', 'w').write(' '.join(sys.argv[3:]))\n")
    spec = VideoSpec(backend="command", command=f'"{sys.executable}" "{script}" {{init}} {{out}} {{frames}} {{prompt}}')
    out = command_generate("a fox", start, start, 56, spec, 0, tmp_path / "v.webm", log=lambda *_: None)
    assert out.read_bytes() == b"PNG"
    assert Path(str(out) + ".args").read_text() == "56 a fox"


def test_capture_from_a_video_file_end_to_end(creature_glb, tmp_path, monkeypatch):
    """A rendered video of a known motion in, a looping clip with travel and footsteps out. Tracks are the
    exact ones (CoTracker is a download), so this checks everything around the tracker."""
    import imageio.v3 as iio

    spec = VideoSpec(width=160, height=160)
    g, rig, cam, seeds = capture_run.prepare(creature_glb, tmp_path / "start", spec)
    truth = swing(rig, 72, period=16.0)
    frames = bench.render_video(rig, *truth, cam, g.uvs, g.texture)
    iio.imwrite(tmp_path / "given.mp4", frames, fps=24)
    exact = track.exact(rig, *truth, seeds, cam)
    calls = []
    monkeypatch.setattr(track, "cotracker", lambda *a, **k: calls.append(1) or exact)
    opts = FitOptions(iterations=200, refine=40)
    entry = capture_run.capture(creature_glb, "An object marches on the spot. [video]", tmp_path / "clip", spec,
                                video=tmp_path / "given.mp4", device="cpu", depth=False, opts=opts,
                                log=lambda *_: None)
    assert entry["source"] == "video" and entry["loop"] and entry["fps"] == 24
    assert entry["loop_frames"] in (16, 32, 48)
    assert Path(entry["glb"]).exists() and AnimatedGLB(entry["glb"]).animations == ["marches_on_the_spot"]
    assert all(e["joint"] in ("leg_l", "leg_r", "hips") for e in entry["events"] if e["type"] == "contact")
    report = json.loads((tmp_path / "clip" / "capture.json").read_text())
    assert report["fit"]["residual_px"]["mean"] < 1.0

    # Through the motion stage: made once, then reused.
    monkeypatch.setattr(capture_run, "generate", lambda *a, **k: tmp_path / "given.mp4")
    kwargs = dict(subject=None, seed=0, device="cpu", depth=False, log=lambda *_: None)
    real_capture = capture_run.capture
    monkeypatch.setattr(capture_run, "capture", lambda *a, **k: real_capture(*a, **{**k, "opts": opts}))
    first = animate.run_video(creature_glb, tmp_path / "video", ["An object marches on the spot. [video]"], spec,
                              **kwargs)
    again = animate.run_video(creature_glb, tmp_path / "video", ["An object marches on the spot. [video]"], spec,
                              **kwargs)
    assert first == again and len(calls) == 2
