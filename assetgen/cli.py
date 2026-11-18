"""assetgen: image -> rigged, animated glTF.

  assetgen setup                       fetch SkinTokens, UniMate and their weights
  assetgen doctor                      check GPU, packages, checkouts and weights
  assetgen run --image fig.png --out outputs/fig
  assetgen run --ply splat.ply --reference input.png --out outputs/x --from rig
  assetgen batch --images figs/ --out outputs --devices cuda:0,cuda:1
  assetgen batch --splats splats/ --out outputs --devices cuda:0,cuda:1
  assetgen webui [--host 0.0.0.0]                       browser front end
  assetgen synth --mesh Fox.glb --out testdata/fox      (Idol-like test input from a mesh)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import STAGES, Options, run

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _add_run_options(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--name", default=None, help="asset name (default: the input file stem)")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N (ROCm GPUs are cuda devices) or cpu")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"], help="Idol input type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", nargs="+", default=[], help='motion prompts, each "An object <does one thing>."')
    ap.add_argument("--repetitions", type=int, default=1, help="motions per prompt")
    ap.add_argument("--height", type=float, default=None, help="final asset height in metres")
    ap.add_argument("--rig-class", default=None, help="force a SkinTokens rig template (e.g. mixamo, vroid)")
    ap.add_argument("--num-beams", type=int, default=10, help="SkinTokens beam search width")
    ap.add_argument("--skin-postprocess", action="store_true", help="SkinTokens voxel skin clean-up")
    ap.add_argument("--cfg-scale", type=float, default=None, help="UniMate prompt strength (default 3)")
    ap.add_argument("--llm-base-url", default=None, help="OpenAI-compatible endpoint (e.g. llama.cpp) for joint labels")
    ap.add_argument("--llm-model", default="local")
    ap.add_argument("--faces", type=int, default=None, help="mesh face budget (default 40000)")
    ap.add_argument("--texture", type=int, default=None, help="texture size (default 2048)")
    ap.add_argument("--voxels", type=int, default=None, help="fusion resolution (default 256)")
    ap.add_argument("--from", dest="start", choices=STAGES, default=None, help="first stage to run")
    ap.add_argument("--to", dest="stop", choices=STAGES, default=None, help="last stage to run")
    ap.add_argument("--force", action="store_true", help="rerun stages that are up to date")


def _options(a, out: Path, image: Path | None, ply: Path | None, reference: Path | None, device: str) -> Options:
    mesh_args = []
    for flag, value in (("--target_faces", a.faces), ("--texture_size", a.texture), ("--resolution", a.voxels)):
        if value is not None:
            mesh_args += [flag, str(value)]
    stem = (image or ply).stem
    return Options(out=out, image=image, ply=ply, reference=reference, name=a.name or stem.replace(" ", "_"),
                   device=device, mode=a.mode, seed=a.seed, prompts=a.prompt, repetitions=a.repetitions,
                   height=a.height, rig_class=a.rig_class, num_beams=a.num_beams,
                   skin_postprocess=a.skin_postprocess, mesh_args=mesh_args, llm_base_url=a.llm_base_url,
                   llm_model=a.llm_model, cfg_scale=a.cfg_scale)


def cmd_run(a) -> None:
    if bool(a.image) == bool(a.ply):
        raise SystemExit("give exactly one of --image or --ply")
    o = _options(a, Path(a.out), Path(a.image) if a.image else None, Path(a.ply) if a.ply else None,
                 Path(a.reference) if a.reference else None, a.device)
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
        if item.is_dir():  # a splat folder, or an earlier run's output folder
            folder = _splat_folder(item)
            ref = folder / "input.png"
            o = _options(args, out_root / item.name, None, folder / "splat.ply", ref if ref.exists() else None, device)
            o.name = args.name or item.name
        else:
            o = _options(args, out_root / item.stem, item, None, None, device)
        try:
            run(o, args.start, args.stop, args.force, log=lambda m, d=device, n=o.name: print(f"[{d} {n}] {m}"))
            results.append((item.name, "ok"))
        except SystemExit as e:
            results.append((item.name, str(e)))


def cmd_batch(a) -> None:
    import queue
    import threading

    if bool(a.images) == bool(a.splats):
        raise SystemExit("give exactly one of --images or --splats")
    if a.images:
        items = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        items = sorted(p for p in Path(a.splats).iterdir() if p.is_dir() and _splat_folder(p))
    if not items:
        raise SystemExit(f"nothing to do in {a.images or a.splats}")
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
    ap = argparse.ArgumentParser(prog="assetgen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one asset, image or splat to animated glTF")
    r.add_argument("--image", default=None)
    r.add_argument("--ply", default=None, help="start from an existing splat instead of an image")
    r.add_argument("--reference", default=None, help="with --ply: the Idol input image, for orientation")
    r.add_argument("--out", required=True)
    _add_run_options(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="a folder of images (or of splat folders), shared among GPUs")
    b.add_argument("--images", default=None)
    b.add_argument("--splats", default=None,
                   help="folder of <name>/splat.ply (+ input.png), or of earlier runs' output folders")
    b.add_argument("--out", required=True)
    b.add_argument("--devices", default="cuda:0", help="comma-separated, e.g. cuda:0,cuda:1")
    _add_run_options(b)
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("setup", help="fetch pinned SkinTokens/UniMate checkouts and model weights")
    s.add_argument("--no-weights", action="store_true")
    s.set_defaults(func=cmd_setup)

    w = sub.add_parser("webui", help="browser front end")
    w.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines")
    w.add_argument("--port", type=int, default=7860)
    w.add_argument("--device", default=None)
    w.add_argument("--concurrency", type=int, default=1, help="runs at once (one per GPU at most)")
    w.set_defaults(func=lambda a: __import__("assetgen.webui", fromlist=["main"]).main(
        ["--host", a.host, "--port", str(a.port), "--concurrency", str(a.concurrency),
         *(["--device", a.device] if a.device else [])]))

    d = sub.add_parser("doctor", help="check the environment")
    d.set_defaults(func=cmd_doctor)

    y = sub.add_parser("synth", help="Idol-like splat + input image from a textured mesh (for testing)")
    y.add_argument("--mesh", required=True)
    y.add_argument("--out", required=True)
    y.add_argument("--yaw", type=float, default=137.0, help="direction the object faces in the splat")
    y.add_argument("--view", type=float, default=30.0, help="input camera offset from the facing direction")
    y.add_argument("--gaussians", type=int, default=250_000)
    y.set_defaults(func=cmd_synth)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
