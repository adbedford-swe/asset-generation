"""Run the stages for one asset, each in its own process, resumably.

    <out>/
      0_image/    image.png, image.json                      (image: text to picture)
      0_splat/    splat.ply, input.png, camera.json          (generate)
      1_mesh/     mesh.glb, canonical.ply, front.png, compare.png, report.json
      2_rig/      rigged.glb, rigged.json
      3_motion/   asset/, samples/, animated/, clips.json
      <name>.glb  the rigged, textured asset with every clip  (finalize)
      <name>.assetgen  the character file engines read      (finalize)
      manifest.json, logs/<stage>.log

A stage is skipped when its outputs exist and it last ran with the same
arguments; once a stage runs, every later stage runs too.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .imagegen import ImageSpec

STAGES = ["image", "generate", "mesh", "rig", "animate", "finalize"]
ANSI = re.compile(r"\[[0-9;]*[A-Za-z]")  # colour codes some stages print


@dataclass
class Options:
    out: Path
    text: str | None = None
    image_spec: ImageSpec | None = None
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
    rig_candidates: int = 3
    skin_postprocess: bool = False
    mesh_args: list[str] = field(default_factory=list)
    llm_base_url: str | None = None
    llm_model: str = "local"
    cfg_scale: float | None = None
    loop: str = "auto"
    root_motion: bool = False
    flow_steps: int | None = None
    fbx: bool = False


def stage_commands(o: Options) -> dict[str, tuple[list[str], list[Path]]]:
    out = o.out.resolve()
    splat = out / "0_splat"
    image = o.image.resolve() if o.image else None
    cmds = {}
    if o.text and not image and not o.ply:
        spec = o.image_spec or ImageSpec.from_env()
        image = out / "0_image" / "image.png"
        args = ["assetgen.imagegen", "--prompt", o.text, "--out", out / "0_image", "--seed", o.seed,
                "--backend", spec.backend, "--width", spec.width, "--height", spec.height]
        for flag, value in (("--url", spec.url), ("--model", spec.model), ("--steps", spec.steps),
                            ("--cfg_scale", spec.cfg_scale), ("--sampler", spec.sampler),
                            ("--workflow", spec.workflow), ("--command", spec.command)):
            if value is not None:
                args += [flag, value]
        if not spec.style:
            args.append("--raw")
        cmds["image"] = (args, [image])
    ply = (o.ply.resolve() if o.ply else splat / "splat.ply")
    reference = o.reference.resolve() if o.reference else (splat / "input.png" if image else None)
    if image:
        cmds["generate"] = (["assetgen.generate", "--image", image, "--out", splat, "--device", o.device,
                             "--mode", o.mode, "--seed", o.seed], [splat / "splat.ply", splat / "input.png"])
    mesh = ["assetgen.meshing", "--ply", ply, "--out", out / "1_mesh", "--device", o.device, *o.mesh_args]
    if reference:
        mesh += ["--reference", reference]
    cmds["mesh"] = (mesh, [out / "1_mesh" / "mesh.glb"])
    rig = ["assetgen.rig", "--mesh", out / "1_mesh" / "mesh.glb", "--out", out / "2_rig" / "rigged.glb",
           "--device", o.device, "--seed", o.seed, "--num_beams", o.num_beams, "--candidates", o.rig_candidates]
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
    anim += ["--loop", o.loop]
    if o.flow_steps is not None:
        anim += ["--flow_steps", o.flow_steps]
    if o.fbx:
        anim.append("--fbx")
    if o.llm_base_url:
        anim += ["--llm_base_url", o.llm_base_url, "--llm_model", o.llm_model]
    cmds["animate"] = (anim, [out / "3_motion" / "clips.json"])
    fin = ["assetgen.finalize", "--clips", out / "3_motion" / "clips.json", "--out", out / f"{o.name}.glb"]
    if o.height is not None:
        fin += ["--height", o.height]
    if o.root_motion:
        fin.append("--root_motion")
    cmds["finalize"] = (fin, [out / f"{o.name}.glb", out / f"{o.name}.assetgen"])
    return {k: ([str(a) for a in v[0]], v[1]) for k, v in cmds.items()}


def child_env() -> dict:
    env = dict(os.environ)
    # RDNA3/RDNA4 only get fused SDPA kernels with this set (PyTorch keeps them experimental).
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class Cancelled(SystemExit):
    """The run was stopped from outside (``run(..., cancel=event)``)."""


def _popen(args: list[str]) -> subprocess.Popen:
    # Its own process group, so cancelling stops the stage's children (Blender, UniMate) too.
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    return subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=child_env(), text=True, encoding="utf-8", errors="replace", **group)


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(o: Options, start: str | None = None, stop: str | None = None, force: bool = False, log=print,
        cancel: threading.Event | None = None, on_stage=None) -> Path:
    """Run the stages from ``start`` to ``stop``. Setting ``cancel`` stops the running stage and raises
    :class:`Cancelled`; ``on_stage(stage, state)`` hears "running", "done" or "skipped"."""
    out = o.out.resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": {}}
    # What the run was made from, so it can be continued later (see run_inputs).
    (out / "run.json").write_text(json.dumps({
        "text": o.text, "image_spec": o.image_spec.public() if o.image_spec else None,
        "image": str(o.image) if o.image else None, "ply": str(o.ply) if o.ply else None,
        "reference": str(o.reference) if o.reference else None, "name": o.name, "mode": o.mode}, indent=2))
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
            if on_stage:
                on_stage(stage, "skipped")
            continue
        if cancel is not None and cancel.is_set():
            raise Cancelled(f"cancelled before {stage}")
        rerun = True
        log(f"== {stage}")
        if on_stage:
            on_stage(stage, "running")
        t0 = time.time()
        with open(out / "logs" / f"{stage}.log", "w", encoding="utf-8") as fh:
            proc = _popen(args)
            if cancel is not None:
                def stop_when_cancelled(p=proc):
                    while p.poll() is None:
                        if cancel.wait(0.5):
                            _kill_tree(p)
                            return

                threading.Thread(target=stop_when_cancelled, daemon=True).start()
            for line in proc.stdout:
                stamped = f"{time.time() - t0:7.1f}s  {ANSI.sub('', line)}"
                fh.write(stamped)
                fh.flush()
                if "Warning" not in line and not line.lstrip().startswith("warnings.warn"):
                    log("   " + stamped.rstrip())
            code = proc.wait()
        manifest["stages"][stage] = {"args": args, "ok": code == 0, "seconds": round(time.time() - t0, 1)}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if cancel is not None and cancel.is_set():
            raise Cancelled(f"cancelled during {stage}")
        if code != 0:
            raise SystemExit(f"stage {stage} failed (exit {code}); see {out / 'logs' / (stage + '.log')}")
        log(f"   {stage} done in {time.time() - t0:.0f}s")
        if on_stage:
            on_stage(stage, "done")
    return out / f"{o.name}.glb"


STAGE_DIRS = {"image": "0_image", "generate": "0_splat", "mesh": "1_mesh", "rig": "2_rig", "animate": "3_motion"}


def run_inputs(folder: Path) -> dict:
    """What a run folder was made from: run.json, or for older runs, the arguments in its manifest."""
    folder = Path(folder)
    if (folder / "run.json").exists():
        return json.loads((folder / "run.json").read_text())
    stages = json.loads((folder / "manifest.json").read_text()).get("stages", {})

    def arg(stage, flag):
        args = stages.get(stage, {}).get("args", [])
        return args[args.index(flag) + 1] if flag in args else None

    return {"image": arg("generate", "--image"), "ply": None if "generate" in stages else arg("mesh", "--ply"),
            "reference": None if "generate" in stages else arg("mesh", "--reference"),
            "name": arg("animate", "--name") or folder.name, "mode": arg("generate", "--mode") or "synthetic"}


def completed_stages(folder: Path) -> list[str]:
    """Stages of a run folder that finished, in pipeline order."""
    path = Path(folder) / "manifest.json"
    if not path.exists():
        return []
    stages = json.loads(path.read_text()).get("stages", {})
    return [s for s in STAGES if stages.get(s, {}).get("ok")]


def branch(source: Path, start: str, target: Path) -> Path:
    """A new run folder holding ``source``'s results for every stage before ``start``, so the run
    can continue from ``start`` without touching ``source``."""
    import shutil

    source, target = Path(source), Path(target)
    done = completed_stages(source)
    needed = [s for s in STAGES[:STAGES.index(start)] if s not in ("image", "generate") or s in done]
    missing = [s for s in needed if s not in done]
    if missing:
        raise SystemExit(f"{source.name} has not finished {', '.join(missing)}; start from {missing[0]} instead")
    target.mkdir(parents=True)
    for d in ["input", *[STAGE_DIRS[s] for s in needed]]:
        if (source / d).exists():
            shutil.copytree(source / d, target / d)
    # Starting from the motion stage, the rig is unchanged: keep its UniMate preparation, which the
    # motion stage reuses when the rig and its settings match.
    if start == "animate" and (source / "3_motion" / "asset").exists():
        shutil.copytree(source / "3_motion" / "asset", target / "3_motion" / "asset")
    stages = json.loads((source / "manifest.json").read_text())["stages"]
    (target / "manifest.json").write_text(json.dumps({"stages": {s: stages[s] for s in needed}}, indent=2))
    return target
