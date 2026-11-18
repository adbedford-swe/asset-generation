"""assetgen: text or image -> rigged, animated character for Unity, Godot or any glTF engine.

  assetgen setup                       fetch SkinTokens, UniMate and their weights
  assetgen doctor                      check GPU, packages, checkouts and weights
  assetgen run --text "a goblin shaman with a lantern" --out outputs/goblin
  assetgen run --image fig.png --out outputs/fig
  assetgen run --ply splat.ply --reference input.png --out outputs/x --from rig
  assetgen batch --images figs/ --out outputs --devices cuda:0,cuda:1
  assetgen batch --splats splats/ --out outputs --devices cuda:0,cuda:1
  assetgen image --prompt "a goblin shaman" --out drafts/goblin --count 4   (text to image only)
  assetgen export outputs/goblin --engine unity --project ~/MyGame
  assetgen serve [--host 0.0.0.0]                       HTTP API and web UI (also: python server.py)
  assetgen synth --mesh Fox.glb --out testdata/fox      (Idol-like test input from a mesh)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .imagegen import BACKENDS, ImageSpec
from .pipeline import STAGES, Options, run

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _add_run_options(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--name", default=None, help="asset name (default: the input file stem)")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N (ROCm GPUs are cuda devices) or cpu")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"], help="Idol input type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", nargs="+", default=[],
                    help='motion prompts, each "An object <does one thing>."; each loops unless it ends in "[once]"')
    ap.add_argument("--loop", default="auto",
                    help="auto: loop length from each motion's own cycle (3 s without one); off; or a loop "
                         "length in frames at 30 fps, any length up to 600")
    ap.add_argument("--root-motion", action="store_true",
                    help="keep looping clips' root travel (default: they loop in place, speed in the clip's extras)")
    ap.add_argument("--flow-steps", type=int, default=None, help="motion sampler steps (default 32)")
    ap.add_argument("--fbx", action="store_true", help="also export every clip as FBX")
    ap.add_argument("--repetitions", type=int, default=1, help="motions per prompt")
    ap.add_argument("--height", type=float, default=None, help="final asset height in metres")
    ap.add_argument("--rig-class", default=None, help="force a SkinTokens rig template (e.g. mixamo, vroid)")
    ap.add_argument("--num-beams", type=int, default=10, help="SkinTokens beam search width")
    ap.add_argument("--rig-candidates", type=int, default=3, help="rigs sampled per asset; the most plausible is kept")
    ap.add_argument("--skin-postprocess", action="store_true", help="SkinTokens voxel skin clean-up")
    ap.add_argument("--cfg-scale", type=float, default=None, help="UniMate prompt strength (default 3)")
    ap.add_argument("--llm-base-url", default=None, help="OpenAI-compatible endpoint (e.g. llama.cpp) for joint labels")
    ap.add_argument("--llm-model", default="local")
    ap.add_argument("--faces", type=int, default=None, help="mesh face budget (default 40000)")
    ap.add_argument("--texture", type=int, default=None, help="texture size (default 2048)")
    ap.add_argument("--voxels", type=int, default=None, help="fusion resolution (default 256)")
    ap.add_argument("--views", type=int, default=None, help="splat renders for meshing (default 64)")
    ap.add_argument("--render-size", type=int, default=None, help="size of those renders in pixels (default 768)")
    ap.add_argument("--image-backend", default=None, help=f"with --text: {', '.join(BACKENDS)} (default sdcpp)")
    ap.add_argument("--image-url", default=None, help="with --text: the image server, e.g. http://127.0.0.1:1234")
    ap.add_argument("--image-model", default=None)
    ap.add_argument("--image-size", default=None, help="with --text: WIDTHxHEIGHT (default 1024x1024)")
    ap.add_argument("--image-steps", type=int, default=None)
    ap.add_argument("--comfy-workflow", default=None, help="comfyui: an API-format workflow with {{prompt}}")
    ap.add_argument("--image-command", default=None, help='command backend: e.g. "sd-cli -p {prompt} -o {out}"')
    ap.add_argument("--raw-prompt", action="store_true",
                    help="send --text to the image backend as written, without the full-body character framing")
    ap.add_argument("--image-keep-loaded", action="store_true",
                    help="sdcpp: leave sd-server's model loaded after the picture (default: unload it to free the GPU)")
    ap.add_argument("--from", dest="start", choices=STAGES, default=None, help="first stage to run")
    ap.add_argument("--to", dest="stop", choices=STAGES, default=None, help="last stage to run")
    ap.add_argument("--force", action="store_true", help="rerun stages that are up to date")


def image_spec(a) -> ImageSpec:
    width = height = None
    if a.image_size:
        width, height = (int(v) for v in a.image_size.lower().split("x"))
    return ImageSpec.from_env(backend=a.image_backend, url=a.image_url, model=a.image_model, width=width,
                              height=height, steps=a.image_steps, workflow=a.comfy_workflow,
                              command=a.image_command, style=False if a.raw_prompt else None,
                              keep_loaded=True if a.image_keep_loaded else None)


def _options(a, out: Path, image: Path | None, ply: Path | None, reference: Path | None, device: str,
             text: str | None = None) -> Options:
    mesh_args = []
    for flag, value in (("--target_faces", a.faces), ("--texture_size", a.texture), ("--resolution", a.voxels),
                        ("--n_views", a.views), ("--render_size", a.render_size)):
        if value is not None:
            mesh_args += [flag, str(value)]
    stem = (image or ply).stem if (image or ply) else re.sub(r"[^A-Za-z0-9]+", "_", text or "asset").strip("_")[:32]
    return Options(out=out, text=text, image_spec=image_spec(a) if text else None, image=image, ply=ply,
                   reference=reference, name=a.name or stem.replace(" ", "_"),
                   device=device, mode=a.mode, seed=a.seed, prompts=a.prompt, repetitions=a.repetitions,
                   height=a.height, rig_class=a.rig_class, num_beams=a.num_beams, rig_candidates=a.rig_candidates,
                   skin_postprocess=a.skin_postprocess, mesh_args=mesh_args, llm_base_url=a.llm_base_url,
                   llm_model=a.llm_model, cfg_scale=a.cfg_scale, loop=a.loop, root_motion=a.root_motion,
                   flow_steps=a.flow_steps, fbx=a.fbx)


def cmd_run(a) -> None:
    if sum(map(bool, (a.text, a.image, a.ply))) != 1:
        raise SystemExit("give exactly one of --text, --image or --ply")
    o = _options(a, Path(a.out), Path(a.image) if a.image else None, Path(a.ply) if a.ply else None,
                 Path(a.reference) if a.reference else None, a.device, text=a.text)
    final = run(o, a.start, a.stop, a.force)
    if final.exists():
        print(f"done: {final}")


def _splat_folder(item: Path) -> Path | None:
    """<item>/splat.ply, or <item>/0_splat/splat.ply from an earlier run."""
    for folder in (item, item / "0_splat"):
        if (folder / "splat.ply").exists():
            return folder
    return None


def _batch_worker(queue, device, args, out_root, results):
    while True:
        try:
            item = queue.get_nowait()
        except Exception:
            return
        if isinstance(item, str):  # a description, from --texts
            stem = re.sub(r"[^A-Za-z0-9]+", "_", item).strip("_")[:32] or "asset"
            o = _options(args, out_root / stem, None, None, None, device, text=item)
        elif item.is_dir():  # a splat folder, or an earlier run's output folder
            folder = _splat_folder(item)
            ref = folder / "input.png"
            o = _options(args, out_root / item.name, None, folder / "splat.ply", ref if ref.exists() else None, device)
            o.name = args.name or item.name
        else:
            o = _options(args, out_root / item.stem, item, None, None, device)
        try:
            run(o, args.start, args.stop, args.force, log=lambda m, d=device, n=o.name: print(f"[{d} {n}] {m}"))
            results.append((o.name, "ok"))
        except SystemExit as e:
            results.append((o.name, str(e)))


def cmd_batch(a) -> None:
    import queue
    import threading

    if sum(map(bool, (a.images, a.splats, a.texts))) != 1:
        raise SystemExit("give exactly one of --images, --splats or --texts")
    if a.texts:
        items = [line.strip() for line in Path(a.texts).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    elif a.images:
        items = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        items = sorted(p for p in Path(a.splats).iterdir() if p.is_dir() and _splat_folder(p))
    if not items:
        raise SystemExit(f"nothing to do in {a.images or a.splats or a.texts}")
    q = queue.Queue()
    for p in items:
        q.put(p)
    results = []
    # Each stage is its own process, so one thread per GPU is enough to keep every GPU busy.
    threads = [threading.Thread(target=_batch_worker, args=(q, d.strip(), a, Path(a.out), results))
               for d in a.devices.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for name, status in results:
        print(f"{name}: {status}")
    if any(s != "ok" for _, s in results):
        raise SystemExit(1)


def cmd_setup(a) -> None:
    from .thirdparty import setup

    setup(weights=not a.no_weights)


def cmd_doctor(a) -> None:
    from .doctor import doctor

    sys.exit(0 if doctor() else 1)


def cmd_image(a) -> None:
    from .imagegen import write

    write(a.prompt, Path(a.out), image_spec(a), a.seed if a.seed >= 0 else None, a.count)


def cmd_export(a) -> None:
    from .export import export, export_zip, install

    if a.project:
        install(Path(a.source), a.engine, Path(a.project), a.folder, a.package_source)
    elif a.zip:
        print(export_zip(Path(a.source), a.engine, Path(a.out) if a.out else None))
    else:
        print(export(Path(a.source), a.engine, Path(a.out or "exports")))


def cmd_synth(a) -> None:
    import torch
    from PIL import Image

    from .splats import save_ply
    from .synthetic import idol_like, input_image, load_textured_mesh, mesh_to_gaussians

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    canon = mesh_to_gaussians(load_textured_mesh(a.mesh), a.gaussians).to(device)
    save_ply(idol_like(canon, a.yaw), out / "splat.ply")
    Image.fromarray(input_image(canon, a.view)).save(out / "input.png")  # relative to the facing
    print(f"wrote {out / 'splat.ply'} and {out / 'input.png'} (facing {a.yaw} deg, viewed from {a.yaw + a.view})")


def main(argv=None) -> None:
    # Stage output is relayed verbatim; never let a console code page turn a character into a crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(prog="assetgen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one character: text, image or splat to animated glTF")
    r.add_argument("--text", default=None, help="describe the character; an image backend draws it first")
    r.add_argument("--image", default=None)
    r.add_argument("--ply", default=None, help="start from an existing splat instead of an image")
    r.add_argument("--reference", default=None, help="with --ply: the Idol input image, for orientation")
    r.add_argument("--out", required=True)
    _add_run_options(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="a folder of images (or of splat folders), shared among GPUs")
    b.add_argument("--images", default=None)
    b.add_argument("--texts", default=None, help="a text file: one character description per line")
    b.add_argument("--splats", default=None,
                   help="folder of <name>/splat.ply (+ input.png), or of earlier runs' output folders")
    b.add_argument("--out", required=True)
    b.add_argument("--devices", default="cuda:0", help="comma-separated, e.g. cuda:0,cuda:1")
    _add_run_options(b)
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("setup", help="fetch pinned SkinTokens/UniMate checkouts and model weights")
    s.add_argument("--no-weights", action="store_true")
    s.set_defaults(func=cmd_setup)

    for command, text in (("serve", "HTTP API and web UI on one port (see assetgen serve --help)"),
                          ("webui", "same as serve")):
        w = sub.add_parser(command, help=text, add_help=False)
        w.set_defaults(func=lambda a: __import__("assetgen.server", fromlist=["main"]).main(a.rest))

    i = sub.add_parser("image", help="text to character image only (to try a prompt or a backend)")
    i.add_argument("--prompt", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--count", type=int, default=1)
    i.add_argument("--seed", type=int, default=-1)
    for flag in ("--image-backend", "--image-url", "--image-model", "--image-size", "--comfy-workflow",
                 "--image-command"):
        i.add_argument(flag, default=None)
    i.add_argument("--image-steps", type=int, default=None)
    i.add_argument("--raw-prompt", action="store_true")
    i.add_argument("--image-keep-loaded", action="store_true")
    i.set_defaults(func=cmd_image)

    e = sub.add_parser("export", help="package a finished character for unity, godot or plain gltf")
    e.add_argument("source", help="a run folder or its <name>.glb")
    e.add_argument("--engine", choices=["unity", "godot", "gltf"], default="unity")
    e.add_argument("--project", default=None, help="copy straight into this Unity or Godot project")
    e.add_argument("--folder", default=None, help="folder inside the project (Assets/Characters, characters)")
    e.add_argument("--package-source", choices=["auto", "local", "git"], default="auto",
                   help="unity: reference com.assetgen.unity from this checkout (local) or from GitHub (git)")
    e.add_argument("--zip", action="store_true", help="write <name>-<engine>.zip instead of a folder")
    e.add_argument("--out", default=None, help="folder (default exports/) or zip path")
    e.set_defaults(func=cmd_export)

    d = sub.add_parser("doctor", help="check the environment")
    d.set_defaults(func=cmd_doctor)

    y = sub.add_parser("synth", help="Idol-like splat + input image from a textured mesh (for testing)")
    y.add_argument("--mesh", required=True)
    y.add_argument("--out", required=True)
    y.add_argument("--yaw", type=float, default=137.0, help="direction the object faces in the splat")
    y.add_argument("--view", type=float, default=30.0, help="input camera offset from the facing direction")
    y.add_argument("--gaussians", type=int, default=250_000)
    y.set_defaults(func=cmd_synth)

    a, rest = ap.parse_known_args(argv)
    if a.cmd in ("serve", "webui"):
        a.rest = rest
    elif rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    a.func(a)


if __name__ == "__main__":
    main()
