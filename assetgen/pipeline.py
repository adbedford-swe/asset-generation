"""Run the stages for one asset, each in its own process, resumably.

    <out>/
      0_splat/    splat.ply, input.png, camera.json          (generate)
      1_mesh/     mesh.glb, canonical.ply, front.png, compare.png, report.json
      2_rig/      rigged.glb, rigged.json
      3_motion/   asset/, samples/, animated/, clips.json
      <name>.glb  the rigged, textured asset with every clip  (finalize)
      manifest.json, logs/<stage>.log

A stage is skipped when its outputs exist and it last ran with the same
arguments; once a stage runs, every later stage runs too.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ["generate", "mesh", "rig", "animate", "finalize"]


@dataclass
class Options:
    out: Path
    image: Path | None = None
    ply: Path | None = None
    reference: Path | None = None
    name: str = "asset"
    device: str = "cuda"
    mode: str = "synthetic"
    seed: int = 42
    prompts: list[str] = field(default_factory=list)
    repetitions: int = 1
    height: float | None = None
    rig_class: str | None = None
    num_beams: int = 10
    skin_postprocess: bool = False
    mesh_args: list[str] = field(default_factory=list)
    llm_base_url: str | None = None
    llm_model: str = "local"
    cfg_scale: float | None = None


def stage_commands(o: Options) -> dict[str, tuple[list[str], list[Path]]]:
    out = o.out.resolve()
    splat = out / "0_splat"
    ply = (o.ply.resolve() if o.ply else splat / "splat.ply")
    reference = o.reference.resolve() if o.reference else (splat / "input.png" if o.image else None)
    cmds = {}
    if o.image:
        cmds["generate"] = (["assetgen.generate", "--image", o.image.resolve(), "--out", splat, "--device", o.device,
                             "--mode", o.mode, "--seed", o.seed], [splat / "splat.ply", splat / "input.png"])
    mesh = ["assetgen.meshing", "--ply", ply, "--out", out / "1_mesh", "--device", o.device, *o.mesh_args]
    if reference:
        mesh += ["--reference", reference]
    cmds["mesh"] = (mesh, [out / "1_mesh" / "mesh.glb"])
    rig = ["assetgen.rig", "--mesh", out / "1_mesh" / "mesh.glb", "--out", out / "2_rig" / "rigged.glb",
           "--device", o.device, "--seed", o.seed, "--num_beams", o.num_beams]
    if o.rig_class:
        rig += ["--cls", o.rig_class]
    if o.skin_postprocess:
        rig.append("--postprocess")
    cmds["rig"] = (rig, [out / "2_rig" / "rigged.glb", out / "2_rig" / "rigged.json"])
    anim = ["assetgen.animate", "--rigged", out / "2_rig" / "rigged.glb", "--out", out / "3_motion",
            "--name", o.name, "--device", o.device, "--repetitions", o.repetitions, "--seed", o.seed]
    if o.prompts:
        anim += ["--prompt", *o.prompts]
    if o.cfg_scale is not None:
        anim += ["--cfg_scale", o.cfg_scale]
    if o.llm_base_url:
        anim += ["--llm_base_url", o.llm_base_url, "--llm_model", o.llm_model]
    cmds["animate"] = (anim, [out / "3_motion" / "clips.json"])
    fin = ["assetgen.finalize", "--clips", out / "3_motion" / "clips.json", "--out", out / f"{o.name}.glb"]
    if o.height is not None:
        fin += ["--height", o.height]
    cmds["finalize"] = (fin, [out / f"{o.name}.glb"])
    return {k: ([str(a) for a in v[0]], v[1]) for k, v in cmds.items()}


def child_env() -> dict:
    env = dict(os.environ)
    # RDNA3/RDNA4 only get fused SDPA kernels with this set (PyTorch keeps them experimental).
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(o: Options, start: str | None = None, stop: str | None = None, force: bool = False, log=print) -> Path:
    out = o.out.resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": {}}
    cmds = stage_commands(o)
    order = [s for s in STAGES if s in cmds]
    if start:
        order = order[order.index(start):]
    if stop:
        order = order[:order.index(stop) + 1]

    if any(s in order for s in ("rig", "animate")):
        from .thirdparty import require

        require(log)

    rerun = force
    for stage in order:
        args, outputs = cmds[stage]
        prev = manifest["stages"].get(stage, {})
        if not rerun and prev.get("args") == args and prev.get("ok") and all(p.exists() for p in outputs):
            log(f"== {stage}: up to date")
            continue
        rerun = True
        log(f"== {stage}")
        t0 = time.time()
        with open(out / "logs" / f"{stage}.log", "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    env=child_env(), text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                stamped = f"{time.time() - t0:7.1f}s  {line}"
                fh.write(stamped)
                fh.flush()
                if "Warning" not in line and not line.lstrip().startswith("warnings.warn"):
                    log("   " + stamped.rstrip())
            code = proc.wait()
        manifest["stages"][stage] = {"args": args, "ok": code == 0, "seconds": round(time.time() - t0, 1)}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if code != 0:
            raise SystemExit(f"stage {stage} failed (exit {code}); see {out / 'logs' / (stage + '.log')}")
        log(f"   {stage} done in {time.time() - t0:.0f}s")
    return out / f"{o.name}.glb"
