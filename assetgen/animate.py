"""Stage 3: rigged mesh + text prompts -> animated clips, with UniMate or from video.

1. ``rig_preprocess`` turns the rig into a UniMate asset (joint labels, facing,
   canonical rest pose) without stopping for review. The facing pair is chosen
   here from geometry: stage 1 made the asset face +Z, so the right side of a
   mirror pair is the joint at -X.
2. The sampler generates one motion per prompt (and repetition).
3. Each motion drives the canonical mesh and is exported as a .glb.

A prompt tagged ``[video]`` is made from video instead (:mod:`assetgen.capture`): a
video model animates a picture of the character, and the rig is fitted to it. Its
clip is written onto the same asset as UniMate's, so finalize merges them all. When
every prompt is a video prompt, UniMate is not run at all.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import numpy as np

from .thirdparty import unimate_dir, unimate_exp_dir

# One UniMate step (preprocess, sampling, or driving one clip) never needs this long.
STEP_TIMEOUT = int(os.environ.get("ASSETGEN_STEP_TIMEOUT", 3600))

DEFAULT_PROMPTS = [
    "An object stands idle and breathes.",
    "An object walks forward.",
    "An object runs forward.",
    "An object attacks forward. [once]",
]


def depth_of(parents: np.ndarray) -> np.ndarray:
    depth = np.zeros(len(parents), dtype=int)
    for j in range(len(parents)):
        p = parents[j]
        while p >= 0:
            depth[j] += 1
            p = parents[p]
    return depth


def facing_pair(joints: np.ndarray, parents: np.ndarray, names: list[str]):
    """Pick (right, left, body_axis) bone names for an asset that faces +Z.

    A lateral pair is two joints mirrored across x = 0 (same height and depth,
    opposite x), as close to the root as possible and well apart. Without one
    (snakes, fish) the pair is head / tail along z."""
    joints = np.asarray(joints, dtype=np.float64)
    size = float(np.ptp(joints, axis=0).max()) or 1.0
    depth = depth_of(np.asarray(parents))
    children = np.bincount(np.asarray(parents)[np.asarray(parents) >= 0], minlength=len(joints))
    best, best_score = None, -np.inf
    for a in range(len(joints)):
        for b in range(len(joints)):
            if joints[a, 0] >= 0 or joints[b, 0] <= 0 or parents[a] < 0 or parents[b] < 0:
                continue
            pa, pb = joints[a], joints[b]
            sep = pb[0] - pa[0]
            mirror = abs(pa[0] + pb[0]) + abs(pa[1] - pb[1]) + abs(pa[2] - pb[2])
            if sep < 0.08 * size or mirror > 0.3 * sep:
                continue
            score = sep / size - 0.08 * (depth[a] + depth[b]) - mirror / size
            score += 0.05 * ((children[a] > 0) + (children[b] > 0))
            if score > best_score:
                best, best_score = (names[a], names[b], False), score
    if best is not None:
        return best
    head, tail = int(np.argmax(joints[:, 2])), int(np.argmin(joints[:, 2]))
    return names[head], names[tail], True


def _env(device: str) -> dict:
    env = dict(os.environ)
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    m = re.fullmatch(r"cuda:(\d+)", device)
    if m:
        # One mask only: ROCm builds read HIP_VISIBLE_DEVICES, CUDA builds CUDA_VISIBLE_DEVICES.
        from importlib.metadata import version

        key = "HIP_VISIBLE_DEVICES" if "rocm" in version("torch") else "CUDA_VISIBLE_DEVICES"
        env.pop("HIP_VISIBLE_DEVICES" if key == "CUDA_VISIBLE_DEVICES" else "CUDA_VISIBLE_DEVICES", None)
        env[key] = m.group(1)
    env["PYTHONPATH"] = str(unimate_dir()) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(args: list[str], env: dict, log=print, expect: Path | None = None, blender: bool = False) -> None:
    """Run a UniMate entry point. bpy, used as a module, can crash while the interpreter shuts
    down after its work is saved; with ``expect`` that file being written during this run counts
    as success."""
    log("[animate] $ " + " ".join(str(a) for a in args))
    started = time.time()
    prefix = ["-m", "assetgen.unimate_compat", *(["--blender"] if blender else [])]
    try:
        code = subprocess.run([sys.executable, *prefix, *map(str, args)], cwd=unimate_dir(), env=env,
                              timeout=STEP_TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        code = -1
        log(f"[animate] gave up after {STEP_TIMEOUT}s")
    if code == 0:
        return
    if expect is not None and expect.exists() and expect.stat().st_mtime >= started - 1:
        log(f"[animate] exited with {code} during shutdown, after writing {expect.name}")
        return
    raise subprocess.CalledProcessError(code, args)


def clip_name(prompt: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", prompt.lower()).split()
    if words[:2] == ["an", "object"]:
        words = words[2:]
    return "_".join(words[:4]) or "clip"


ONCE = re.compile(r"\s*\[once\]\s*$", re.IGNORECASE)
VIDEO = re.compile(r"\s*\[video\]", re.IGNORECASE)


def split_prompt(prompt: str) -> tuple[str, bool]:
    """A prompt and whether its clip loops: every clip loops unless the prompt ends in ``[once]``."""
    prompt = VIDEO.sub("", prompt)
    text = ONCE.sub("", prompt).strip()
    return text, text == prompt.strip()


def from_video(prompt: str) -> bool:
    """Whether a prompt is tagged ``[video]``: its clip is made from video, not by UniMate."""
    return bool(VIDEO.search(prompt))


def _file_key(path: Path, *extra) -> str:
    import hashlib

    h = hashlib.sha1(path.read_bytes())
    h.update(json.dumps([str(e) for e in extra]).encode())
    return h.hexdigest()


def preprocess(rigged: Path, asset_dir: Path, name: str, env: dict, annotate: str, llm: dict | None,
               log=print) -> dict:
    """UniMate's rig_preprocess, skipped when the same rig was already prepared the same way."""
    rig = json.loads(rigged.with_suffix(".json").read_text())
    right, left, body_axis = facing_pair(np.array(rig["joints"]), np.array(rig["parents"]), rig["names"])
    key = _file_key(rigged, name, right, left, body_axis, annotate, llm)
    stamp = asset_dir / ".assetgen_key"
    if stamp.exists() and stamp.read_text() == key and (asset_dir / "summary.json").exists():
        log("[animate] rig preprocessing up to date")
        return json.loads((asset_dir / "summary.json").read_text())
    log(f"[animate] facing pair: right={right} left={left}{' (head/tail)' if body_axis else ''}")
    pre = ["-m", "data_process.rig_preprocess", "run", "--input", rigged, "--output_dir", asset_dir,
           "--name", name, "--no_review", "--overwrite", "--face_r", right, "--face_l", left,
           "--annotate", "llm" if llm else annotate]
    if body_axis:
        pre.append("--body_axis")
    if llm:
        pre += ["--model", llm["model"], "--backend", "openai", "--base_url", llm["base_url"],
                "--api_key", llm.get("api_key", "none")]
    # The pipeline renders its own previews; UniMate's needs Blender rendering and fcntl.
    pre.append("--no_preview")
    _run(pre, env, log, expect=asset_dir / "summary.json", blender=True)
    summary = json.loads((asset_dir / "summary.json").read_text())
    if not summary.get("canonical_glb"):
        raise RuntimeError(f"rig_preprocess built no canonical mesh: {summary.get('canonical_glb_error')}")
    if not summary.get("model_joint_limit", {}).get("fits", True):
        raise RuntimeError(f"rig has {summary.get('n_joints')} joints, over the model's limit")
    for note in summary.get("notes", []):
        log(f"[animate] note: {note}")
    stamp.write_text(key)
    return summary


def trim_loops(samples: Path, loop_prompts: set[str], log=print) -> dict[str, int]:
    """Cut each looping motion to one period plus a closing frame equal to its first (frames 0..L), and
    one-shots back to the model's window (a batch with long loops comes back padded to the longest)."""
    from .motion_sampler import WINDOW, loop_period

    manifest = json.loads((samples / "manifest.json").read_text())
    periods = {}
    for name, entry in manifest.get("samples", {}).items():
        path = samples / manifest.get("motions_dir", "motions") / name
        motion = np.load(path)
        if entry.get("prompt") not in loop_prompts:
            if len(motion) > WINDOW:
                np.save(path, motion[:WINDOW])
            continue
        L = loop_period(motion)
        if L is None:
            log(f"[animate] warning: {name} came back without a loop; kept as sampled")
            continue
        np.save(path, motion[:L + 1])
        periods[name] = L
        log(f"[animate] loop: {entry.get('prompt')!r} -> {L} frames ({L / 30:.2f} s)")
    return periods


def run(rigged: Path, out_dir: Path, prompts: list[str], name: str = "asset", device: str = "cuda",
        repetitions: int = 1, seed: int = 0, cfg_scale: float | None = None, annotate: str = "rule",
        llm: dict | None = None, loop: str = "auto", flow_steps: int | None = None, fbx: bool = False,
        video=None, subject: str | None = None, video_depth: bool = False, log=print) -> list[dict]:
    """Clips for ``prompts``, in their order. ``video`` (a :class:`assetgen.capture.video.VideoSpec`) is how
    ``[video]`` prompts get their videos."""
    rigged, out_dir = rigged.resolve(), out_dir.resolve()
    videos = [p for p in prompts if from_video(p)]
    prompts = [p for p in prompts if not from_video(p)]
    clips = run_unimate(rigged, out_dir, prompts, name, device, repetitions, seed, cfg_scale, annotate, llm, loop,
                        flow_steps, fbx, log) if prompts else []
    if videos:
        # Video clips go onto the asset UniMate's clips drive, so finalize can merge them by node name. Its
        # summary records the path where it was made, which a continued run has copied, so look by file name.
        asset = rigged
        if clips:
            made = Path(json.loads((out_dir / "asset" / "summary.json").read_text())["canonical_glb"])
            asset = out_dir / "asset" / PureWindowsPath(made).name
        clips += run_video(asset, out_dir / "video", videos, video, subject, seed, device, video_depth, log)
    (out_dir / "clips.json").write_text(json.dumps(clips, indent=2))
    return clips


def run_video(asset: Path, out_dir: Path, prompts: list[str], spec=None, subject: str | None = None, seed: int = 0,
              device: str = "cuda", depth: bool = False, log=print) -> list[dict]:
    """A clip from video for each prompt; one already made the same way is reused."""
    from dataclasses import asdict

    from .capture.run import capture
    from .capture.video import VideoSpec

    spec = spec or VideoSpec.from_env()
    clips = []
    for k, prompt in enumerate(prompts):
        text, _ = split_prompt(prompt)
        folder = out_dir / clip_name(text)
        settings = {k_: v for k_, v in asdict(spec).items() if k_ != "keep_loaded"}
        key = _file_key(asset, prompt, subject, seed + k, depth, json.dumps(settings, sort_keys=True))
        stamp = folder / ".assetgen_key"
        record = folder / "capture.json"
        if stamp.exists() and stamp.read_text() == key and record.exists():
            entry = json.loads(record.read_text())["entry"]
            if Path(entry["glb"]).exists():
                log(f"[animate] {text!r} from video: up to date")
                clips.append(entry)
                continue
        clips.append(capture(asset, prompt, folder, spec, subject, seed + k, device=device, depth=depth, log=log))
        stamp.write_text(key)
    return clips


def run_unimate(rigged: Path, out_dir: Path, prompts: list[str], name: str = "asset", device: str = "cuda",
                repetitions: int = 1, seed: int = 0, cfg_scale: float | None = None, annotate: str = "rule",
                llm: dict | None = None, loop: str = "auto", flow_steps: int | None = None, fbx: bool = False,
                log=print) -> list[dict]:
    if not (unimate_exp_dir() / "config.json").exists():
        raise SystemExit(f"UniMate is not set up at {unimate_dir()}: run `uv run assetgen setup`")
    env = _env(device)
    asset_dir = out_dir / "asset"
    preprocess(rigged, asset_dir, name, env, annotate, llm, log)

    parsed = [split_prompt(p) for p in prompts]
    texts = [t for t, _ in parsed]
    loop_prompts = {t for t, loops in parsed if loops} if loop != "off" else set()
    env["ASSETGEN_LOOP"] = json.dumps({"prompts": sorted(loop_prompts),
                                       "frames": int(loop) if loop.isdigit() else "auto"})
    if flow_steps is not None:
        env["ASSETGEN_FLOW_STEPS"] = str(flow_steps)

    samples = out_dir / "samples"
    animated = out_dir / "animated"
    # The sampler adds to an existing manifest; start clean so only this run's motions are driven.
    for d in (samples, animated):
        if d.exists():
            shutil.rmtree(d)
    sample = ["-m", "unimate.inference.sample", "--exp_dir", unimate_exp_dir(), "--asset", asset_dir,
              "--prompt", *texts, "--num_repetitions", str(repetitions), "--seed", str(seed),
              "--output_dir", samples, "--only_save_motion"]
    if cfg_scale is not None:
        sample += ["--cfg_scale", str(cfg_scale)]
    _run(sample, env, log)
    periods = trim_loops(samples, loop_prompts, log)

    listing = subprocess.run([sys.executable, "-m", "data_process.mesh_animation.sample_manifest", str(samples)],
                             cwd=unimate_dir(), env=env, check=True, capture_output=True, text=True).stdout
    animated.mkdir(parents=True, exist_ok=True)
    captions = _captions_by_file(json.loads((samples / "manifest.json").read_text()))
    jobs, clips = [], []
    for line in filter(None, listing.splitlines()):
        anim, char, cond, dtype = line.split("\t")
        stem = Path(anim).stem
        jobs.append({"anim": anim, "char": char, "cond": cond, "dtype": dtype, "out_dir": str(animated)})
        L = periods.get(Path(anim).name)
        clips.append({"glb": str(animated / f"{stem}.glb"), "motion": anim,
                      "prompt": captions.get(Path(anim).name, stem), "loop": L is not None,
                      "loop_frames": L})
    jobs_file = out_dir / "drive_jobs.json"
    jobs_file.write_text(json.dumps(jobs, indent=2))
    formats = "glb,fbx" if fbx else "glb"
    _run(["-m", "assetgen.drive_clips", "--jobs", jobs_file, "--formats", formats], env, log,
         expect=Path(clips[-1]["glb"]) if clips else None, blender=True)
    missing = [c["glb"] for c in clips if not Path(c["glb"]).exists()]
    if missing:
        raise RuntimeError(f"no animated asset for {missing}")
    return clips


def _captions_by_file(manifest) -> dict:
    """Motion file name -> prompt, from the sampler's manifest.json."""
    return {name: entry.get("prompt", Path(name).stem) for name, entry in manifest.get("samples", {}).items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rigged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", nargs="+", default=DEFAULT_PROMPTS,
                    help='motions; each loops seamlessly unless it ends in "[once]"; "[video]" makes it from video')
    ap.add_argument("--name", default="asset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg_scale", type=float, default=None)
    ap.add_argument("--loop", default="auto",
                    help="auto (loop length from the motion's own cycle), off, or a loop length in frames")
    ap.add_argument("--flow_steps", type=int, default=None, help="sampler steps (default 32; 0 = UniMate's dopri5)")
    ap.add_argument("--fbx", action="store_true", help="also export each clip as FBX")
    ap.add_argument("--llm_base_url", default=None, help="OpenAI-compatible endpoint for joint labels")
    ap.add_argument("--llm_model", default="local")
    ap.add_argument("--subject", default=None, help="[video] prompts: what the character is")
    ap.add_argument("--video_backend", default=None, choices=["sdcpp", "command"])
    ap.add_argument("--video_url", default=None)
    ap.add_argument("--video_model", default=None)
    ap.add_argument("--video_command", default=None)
    ap.add_argument("--video_steps", type=int, default=None)
    ap.add_argument("--video_size", default=None, help="WIDTHxHEIGHT of the videos (default 640x640)")
    ap.add_argument("--video_view_yaw", type=float, default=None,
                    help="where the videos' camera stands, degrees round from the character's front")
    ap.add_argument("--video_depth", action="store_true", help="fit [video] clips to estimated depth too")
    a = ap.parse_args(argv)
    llm = {"base_url": a.llm_base_url, "model": a.llm_model} if a.llm_base_url else None
    video = None
    if any(from_video(p) for p in a.prompt):
        from .capture.video import VideoSpec

        width = height = None
        if a.video_size:
            width, height = (int(v) for v in a.video_size.lower().split("x"))
        video = VideoSpec.from_env(backend=a.video_backend, url=a.video_url, model=a.video_model,
                                   command=a.video_command, steps=a.video_steps, width=width, height=height,
                                   view_yaw=a.video_view_yaw)
    run(Path(a.rigged), Path(a.out), a.prompt, a.name, a.device, a.repetitions, a.seed, a.cfg_scale, llm=llm,
        loop=a.loop, flow_steps=a.flow_steps, fbx=a.fbx, video=video, subject=a.subject,
        video_depth=a.video_depth)


if __name__ == "__main__":
    main()
