"""Stage 3: rigged mesh + text prompts -> animated clips with UniMate.

1. ``rig_preprocess`` turns the rig into a UniMate asset (joint labels, facing,
   canonical rest pose) without stopping for review. The facing pair is chosen
   here from geometry: stage 1 made the asset face +Z, so the right side of a
   mirror pair is the joint at -X.
2. The sampler generates one motion per prompt (and repetition).
3. Each motion drives the canonical mesh and is exported as a .glb.
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
from pathlib import Path

import numpy as np

from .thirdparty import unimate_dir, unimate_exp_dir

DEFAULT_PROMPTS = [
    "An object stands idle and breathes.",
    "An object walks forward.",
    "An object runs forward.",
    "An object attacks forward.",
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
            if sep < 0.08 * size or mirror > 0.1 * sep:
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
        env["CUDA_VISIBLE_DEVICES"] = m.group(1)
        env["HIP_VISIBLE_DEVICES"] = m.group(1)
    env["PYTHONPATH"] = str(unimate_dir()) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(args: list[str], env: dict, log=print, expect: Path | None = None, blender: bool = False) -> None:
    """Run a UniMate entry point. bpy, used as a module, can crash while the interpreter shuts
    down after its work is saved; with ``expect`` that file being written during this run counts
    as success."""
    log("[animate] $ " + " ".join(str(a) for a in args))
    started = time.time()
    prefix = ["-m", "assetgen.unimate_compat"] if blender else []
    code = subprocess.run([sys.executable, *prefix, *map(str, args)], cwd=unimate_dir(), env=env).returncode
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


def run(rigged: Path, out_dir: Path, prompts: list[str], name: str = "asset", device: str = "cuda",
        repetitions: int = 1, seed: int = 0, cfg_scale: float | None = None, annotate: str = "rule",
        llm: dict | None = None, log=print) -> list[dict]:
    rigged, out_dir = rigged.resolve(), out_dir.resolve()
    env = _env(device)
    asset_dir = out_dir / "asset"
    rig = json.loads(rigged.with_suffix(".json").read_text())
    right, left, body_axis = facing_pair(np.array(rig["joints"]), np.array(rig["parents"]), rig["names"])
    log(f"[animate] facing pair: right={right} left={left}{' (head/tail)' if body_axis else ''}")

    pre = ["-m", "data_process.rig_preprocess", "run", "--input", rigged, "--output_dir", asset_dir,
           "--name", name, "--no_review", "--overwrite", "--face_r", right, "--face_l", left,
           "--annotate", "llm" if llm else annotate]
    if body_axis:
        pre.append("--body_axis")
    if llm:
        pre += ["--model", llm["model"], "--backend", "openai", "--base_url", llm["base_url"],
                "--api_key", llm.get("api_key", "none")]
    if sys.platform == "win32":
        pre.append("--no_preview")  # the preview renderer locks files with fcntl
    _run(pre, env, log, expect=asset_dir / "summary.json", blender=True)
    summary = json.loads((asset_dir / "summary.json").read_text())
    if not summary.get("canonical_glb"):
        raise RuntimeError(f"rig_preprocess built no canonical mesh: {summary.get('canonical_glb_error')}")
    if not summary.get("model_joint_limit", {}).get("fits", True):
        raise RuntimeError(f"rig has {summary.get('n_joints')} joints, over the model's limit")
    for note in summary.get("notes", []):
        log(f"[animate] note: {note}")

    samples = out_dir / "samples"
    animated = out_dir / "animated"
    # The sampler adds to an existing manifest; start clean so only this run's motions are driven.
    for d in (samples, animated):
        if d.exists():
            shutil.rmtree(d)
    sample = ["-m", "unimate.inference.sample", "--exp_dir", unimate_exp_dir(), "--asset", asset_dir,
              "--prompt", *prompts, "--num_repetitions", str(repetitions), "--seed", str(seed),
              "--output_dir", samples, "--only_save_motion"]
    if cfg_scale is not None:
        sample += ["--cfg_scale", str(cfg_scale)]
    _run(sample, env, log)

    jobs = subprocess.run([sys.executable, "-m", "data_process.mesh_animation.sample_manifest", str(samples)],
                          cwd=unimate_dir(), env=env, check=True, capture_output=True, text=True).stdout
    animated.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((samples / "manifest.json").read_text())
    captions = _captions_by_file(manifest)
    clips = []
    for line in filter(None, jobs.splitlines()):
        anim, char, cond, dtype = line.split("\t")
        stem = Path(anim).stem
        _run(["data_process/mesh_animation/animate_motion.py", "--dataset_type", dtype, "--anim_path", anim,
              "--char_path", char, "--cond_path", cond, "--output_dir", animated, "--anim_mode", "fk",
              "--extra_bones_strategy", "merge"], env, log, expect=animated / f"{stem}.fbx", blender=True)
        clips.append({"glb": str(animated / f"{stem}.glb"), "motion": anim,
                      "prompt": captions.get(Path(anim).name, stem)})
    (out_dir / "clips.json").write_text(json.dumps(clips, indent=2))
    return clips


def _captions_by_file(manifest) -> dict:
    """Motion file name -> prompt, from the sampler's manifest.json."""
    return {name: entry.get("prompt", Path(name).stem) for name, entry in manifest.get("samples", {}).items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rigged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--name", default="asset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg_scale", type=float, default=None)
    ap.add_argument("--llm_base_url", default=None, help="OpenAI-compatible endpoint for joint labels")
    ap.add_argument("--llm_model", default="local")
    a = ap.parse_args(argv)
    llm = {"base_url": a.llm_base_url, "model": a.llm_model} if a.llm_base_url else None
    run(Path(a.rigged), Path(a.out), a.prompt, a.name, a.device, a.repetitions, a.seed, a.cfg_scale, llm=llm)


if __name__ == "__main__":
    main()
