"""Browser front end: run the pipeline on an image (or a splat) and inspect every stage.

    assetgen webui [--host 0.0.0.0] [--port 7860]

Each run writes to its own folder under ``outputs/webui`` (or ``$ASSETGEN_WEBUI_OUT``),
exactly like ``assetgen run``, so a run started here can be resumed from the CLI.
"""
from __future__ import annotations

import html
import os
import queue
import re
import threading
import time
from pathlib import Path

from .animate import DEFAULT_PROMPTS
from .pipeline import Options
from .pipeline import run as run_pipeline
from .thirdparty import ROOT

OUT_ROOT = Path(os.environ.get("ASSETGEN_WEBUI_OUT", ROOT / "outputs" / "webui")).resolve()
MODEL_VIEWER = ('<script type="module" '
                'src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>')
EMPTY_VIEWER = ('<div style="height:520px;display:flex;align-items:center;justify-content:center;'
                'border-radius:8px;background:#1d1d22;color:#888">The asset appears here as stages finish</div>')


def device_choices() -> list[str]:
    import torch

    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]


def viewer_html(glb: Path | None, clip: str | None = None) -> str:
    if glb is None or not glb.exists():
        return EMPTY_VIEWER
    src = f"/gradio_api/file={glb.as_posix()}?v={int(glb.stat().st_mtime)}"
    anim = f' animation-name="{html.escape(clip)}" autoplay' if clip else ""
    return (f'<model-viewer src="{src}"{anim} camera-controls touch-action="pan-y" shadow-intensity="1" '
            f'camera-orbit="30deg 75deg auto" style="width:100%;height:520px;background:#1d1d22;'
            f'border-radius:8px"></model-viewer>')


def clip_names(glb: Path) -> list[str]:
    import pygltflib

    try:
        return [a.name for a in pygltflib.GLTF2.load_binary(str(glb)).animations if a.name]
    except Exception:
        return []


def stage_images(out: Path, name: str) -> list[tuple[str, str]]:
    candidates = [
        (out / "1_mesh" / "front.png", "Mesh, front view (should face you)"),
        (out / "1_mesh" / "compare.png", "Splat (top) vs mesh (bottom), held-out views"),
        (out / "2_rig" / "preview.png", "Skeleton"),
        (out / f"{name}.png", "Every clip"),
    ]
    return [(str(p), caption) for p, caption in candidates if p.exists()]


def best_glb(out: Path, name: str) -> Path | None:
    for p in (out / f"{name}.glb", out / "2_rig" / "rigged.glb", out / "1_mesh" / "mesh.glb"):
        if p.exists():
            return p
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_") or "asset"


def build(default_device: str | None = None):
    import gradio as gr

    devices = device_choices()

    def start(image, ply, reference, name, prompts, height, device, mode, seed, repetitions, faces, texture, voxels,
              beams, rig_class, cfg_scale, llm_url):
        if not image and not ply:
            raise gr.Error("Upload an image, or a splat (.ply).")
        stem = Path(ply or image).stem
        name = _slug(name or stem)
        out = OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        mesh_args = ["--target_faces", str(int(faces)), "--texture_size", str(int(texture)),
                     "--resolution", str(int(voxels))]
        o = Options(out=out, image=None if ply else Path(image), ply=Path(ply) if ply else None,
                    reference=Path(reference) if (ply and reference) else None, name=name, device=device, mode=mode,
                    seed=int(seed), prompts=[p.strip() for p in prompts.splitlines() if p.strip()],
                    repetitions=int(repetitions), height=float(height) if height else None,
                    rig_class=rig_class or None, num_beams=int(beams), mesh_args=mesh_args,
                    llm_base_url=llm_url or None, cfg_scale=float(cfg_scale) if cfg_scale else None)

        events: queue.Queue = queue.Queue()
        outcome = {}

        def work():
            try:
                outcome["glb"] = run_pipeline(o, log=events.put)
            except SystemExit as e:
                outcome["error"] = str(e)
            except Exception as e:  # surfaced in the log, not swallowed
                outcome["error"] = f"{type(e).__name__}: {e}"
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        lines = [f"output folder: {out}"]
        shown = None
        while True:
            try:
                item = events.get(timeout=3)
            except queue.Empty:
                item = ""
            if item is None:
                break
            if item:
                lines.append(item)
            glb = best_glb(out, name)
            update_viewer = glb != shown
            shown = glb
            yield ("\n".join(lines[-400:]), stage_images(out, name),
                   gr.update(value=viewer_html(glb)) if update_viewer else gr.update(),
                   gr.update(), None, str(out))
        final = out / f"{name}.glb"
        if "error" in outcome:
            lines.append(f"FAILED: {outcome['error']}")
        clips = clip_names(final) if final.exists() else []
        yield ("\n".join(lines[-400:]), stage_images(out, name),
               viewer_html(best_glb(out, name), clips[0] if clips else None),
               gr.update(choices=clips, value=clips[0] if clips else None),
               str(final) if final.exists() else None, str(out))

    def pick_clip(clip, folder):
        if not folder:
            return gr.update()
        out = Path(folder)
        glbs = sorted(out.glob("*.glb"))
        return viewer_html(glbs[0] if glbs else None, clip)

    with gr.Blocks(title="assetgen") as demo:
        gr.Markdown("## assetgen\nOne image to a textured, rigged, animated glTF: "
                    "Idol → mesh → SkinTokens → UniMate. Each stage's result appears as it finishes.")
        folder = gr.State("")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Tab("From an image"):
                    image = gr.Image(type="filepath", label="Character image", height=300)
                    mode = gr.Radio(["synthetic", "real"], value="synthetic", label="Image type",
                                    info="synthetic: renders and generated art; real: photographs")
                with gr.Tab("From a splat"):
                    ply = gr.File(label="3D Gaussian splat (.ply)", file_types=[".ply"], type="filepath")
                    reference = gr.Image(type="filepath", label="Idol's input image (optional, for facing)",
                                         height=200)
                name = gr.Textbox(label="Name", placeholder="from the file name")
                prompts = gr.Textbox(value="\n".join(DEFAULT_PROMPTS), lines=4, label="Motions, one per line",
                                     info='Start with "An object" and describe one motion.')
                with gr.Row():
                    height = gr.Number(value=1.8, label="Height (m)", minimum=0.01)
                    device = gr.Dropdown(devices, value=default_device or devices[0], label="Device")
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        repetitions = gr.Number(value=1, precision=0, minimum=1, label="Samples per motion")
                    with gr.Row():
                        faces = gr.Number(value=40000, precision=0, label="Face budget")
                        texture = gr.Dropdown([1024, 2048, 4096], value=2048, label="Texture size")
                        voxels = gr.Number(value=256, precision=0, label="Fusion voxels")
                    with gr.Row():
                        beams = gr.Number(value=10, precision=0, minimum=1, label="SkinTokens beams")
                        rig_class = gr.Textbox(label="Rig template", placeholder="articulation (any shape)")
                        cfg_scale = gr.Number(value=3.0, label="Prompt strength")
                    llm_url = gr.Textbox(label="Joint-label LLM endpoint (optional)",
                                         placeholder="http://localhost:8080/v1 (llama.cpp)")
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=6):
                viewer = gr.HTML(EMPTY_VIEWER)
                with gr.Row():
                    clip = gr.Dropdown([], label="Clip", interactive=True)
                    download = gr.File(label="Asset (.glb)", interactive=False)
                gallery = gr.Gallery(label="Stages", columns=2, height=360, object_fit="contain")
                log = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True)

        go.click(start, [image, ply, reference, name, prompts, height, device, mode, seed, repetitions, faces,
                         texture, voxels, beams, rig_class, cfg_scale, llm_url],
                 [log, gallery, viewer, clip, download, folder])
        clip.change(pick_clip, [clip, folder], viewer)
    return demo


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None)
    ap.add_argument("--concurrency", type=int, default=1, help="runs at once (one per GPU at most)")
    a = ap.parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build(a.device)
    demo.queue(default_concurrency_limit=a.concurrency)
    demo.launch(server_name=a.host, server_port=a.port, allowed_paths=[str(OUT_ROOT)], head=MODEL_VIEWER)


if __name__ == "__main__":
    main()
