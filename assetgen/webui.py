"""Browser front end: describe or upload a character, watch every stage, take it to an engine.

Served by ``assetgen serve`` (or ``python server.py``) next to the HTTP API, on the
same job queue: a job started here shows up in the API and the Unity window, and
the other way round. Runs land in ``outputs/webui/<time>-<name>/``, laid out like
``assetgen run``, so the CLI can resume them. An earlier run can be continued from
any stage: its results up to that stage are copied into a new run, so the run you
started from is never changed.
"""
from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from .animate import DEFAULT_PROMPTS
from .imagegen import BACKENDS, ImageSpec, probe
from .pipeline import completed_stages
from .server import JobManager, Settings, best_glb, slug, version

MODEL_VIEWER = ('<script type="module" '
                'src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>')
EMPTY_VIEWER = ('<div style="height:520px;display:flex;align-items:center;justify-content:center;'
                'border-radius:8px;background:#1d1d22;color:#888">The character appears here as stages finish</div>')
STAGE_LABELS = {"mesh": "Mesh (re-mesh the splat)", "rig": "Rig (keep the mesh)",
                "animate": "Motion (keep mesh and rig)", "finalize": "Finalize (re-assemble the clips)"}
STAGE_NAMES = {"image": "Image", "generate": "Splat", "mesh": "Mesh", "rig": "Rig", "animate": "Motion",
               "finalize": "Finalize"}
CSS = """
.stage-strip {display:flex; gap:6px; flex-wrap:wrap; margin:2px 0 6px}
.stage-strip span {padding:3px 10px; border-radius:999px; font-size:12px; background:#2a2a31; color:#9a9aa5}
.stage-strip span.running {background:#3b4cca; color:white}
.stage-strip span.done, .stage-strip span.skipped {background:#1f6f43; color:white}
.stage-strip span.failed {background:#a33; color:white}
"""


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
        (out / "0_image" / "image.png", "Generated image"),
        (out / "1_mesh" / "front.png", "Mesh, front view (should face you)"),
        (out / "1_mesh" / "compare.png", "Splat (top) vs mesh (bottom), held-out views"),
        (out / "2_rig" / "preview.png", "Skeleton"),
        (out / f"{name}.png", "Every clip"),
    ]
    return [(str(p), caption) for p, caption in candidates if p.exists()]


def stage_strip(job) -> str:
    if job is None:
        return ""
    chips = "".join(f'<span class="{html.escape(state)}">{STAGE_NAMES.get(s, s)}</span>'
                    for s, state in job.to_dict()["stages"].items())
    status = {"running": f"running {STAGE_NAMES.get(job.stage, job.stage or '')}".strip(),
              "failed": f"failed: {job.error or ''}", "cancelled": "cancelled"}.get(job.status, job.status)
    return f'<div class="stage-strip">{chips}</div><div style="font-size:13px;color:#aaa">{html.escape(status)}</div>'


def character_markdown(data: dict | None) -> str:
    if not data:
        return ""
    model = data.get("model", {})
    rows = ["| Clip | Role | Loop | Length | Speed |", "|---|---|---|---|---|"]
    for c in data.get("clips", []):
        speed = f"{c['speed']:.2f} m/s" if c.get("speed") else ""
        rows.append(f"| `{c['name']}` | {c['role']} | {'yes' if c['loop'] else 'once'} | "
                    f"{c['duration']:.2f} s | {speed} |")
    blend = " → ".join(f"`{b['clip']}` ({b['speed']:.2f})" for b in data.get("locomotion", {}).get("blend", []))
    lines = [f"**{data['name']}**: {model.get('height', 0):.2f} m tall, {len(data['skeleton']['joints'])} joints, "
             f"{model.get('triangles', 0):,} triangles"]
    prompt = data.get("source", {}).get("prompt")
    if prompt:
        lines.append(f"*{html.escape(prompt)}*")
    lines += ["", *rows]
    if blend:
        lines += ["", f"Locomotion blend: {blend}"]
    return "\n".join(lines)


def list_runs(manager: JobManager) -> list[tuple[str, str]]:
    runs = []
    for job in manager.list(200):
        done = completed_stages(job.folder)
        state = job.status if job.active else (", ".join(done) if done else "nothing finished")
        runs.append((f"{job.id}  ·  {state}", job.id))
    return runs


def build(manager: JobManager | None = None):

    manager = manager or JobManager()
    devices = ["auto", *manager.devices] if len(manager.devices) > 1 else list(manager.devices)
    if "cpu" not in devices:
        devices.append("cpu")
    spec0 = manager.image_spec

    def settings(prompts, height, device, loop, loop_frames, root_motion, seed, repetitions, faces, texture,
                 voxels, beams, candidates, rig_class, cfg_scale, llm_url) -> Settings:
        return Settings(motions=[p.strip() for p in prompts.splitlines() if p.strip()],
                        height=float(height) if height else None, device=device or "auto", seed=int(seed),
                        loop=(str(int(loop_frames)) if loop_frames else "auto") if loop else "off",
                        root_motion=bool(root_motion), repetitions=int(repetitions), faces=int(faces),
                        texture=int(texture), voxels=int(voxels), num_beams=int(beams),
                        rig_candidates=int(candidates), rig_class=rig_class or None,
                        cfg_scale=float(cfg_scale) if cfg_scale else None, llm_base_url=llm_url or None)

    def image_spec(backend, url, model, size, steps) -> ImageSpec:
        try:
            width, height = (int(v) for v in (size or "1024x1024").lower().split("x"))
        except ValueError:
            raise gr.Error(f"Image size {size!r} is not WIDTHxHEIGHT")
        return ImageSpec(backend=backend or spec0.backend, url=url or None, model=model or None, width=width,
                         height=height, steps=int(steps) if steps else None, workflow=spec0.workflow,
                         command=spec0.command)

    def exports_for(job) -> list[str]:
        from .export import export_zip

        glb = job.folder / f"{job.name}.glb"
        (job.folder / "exports").mkdir(exist_ok=True)
        return [str(export_zip(glb, e, job.folder / "exports" / f"{job.name}-{e}.zip")) for e in ("unity", "godot")]

    def results(job, clip=None):
        """Everything the right-hand side shows for a job."""
        glb = best_glb(job.folder, job.name)
        final = job.folder / f"{job.name}.glb"
        clips = clip_names(final) if final.exists() else []
        clip = clip if clip in clips else (clips[0] if clips else None)
        return (stage_strip(job), stage_images(job.folder, job.name), viewer_html(glb, clip),
                gr.update(choices=clips, value=clip), str(final) if final.exists() else None,
                exports_for(job) if final.exists() else None, character_markdown(job.character()))

    def follow(job):
        """Stream a job's progress into the page until it ends."""
        seq, shown = -1, None
        while job.active:
            seq = job.wait(seq, 2.0)
            glb = best_glb(job.folder, job.name)
            viewer = gr.update(value=viewer_html(glb)) if glb != shown else gr.update()
            shown = glb
            yield (stage_strip(job), "\n".join(job.lines[-400:]), stage_images(job.folder, job.name), viewer,
                   gr.update(), None, None, "", job.id, gr.update())
        strip, gallery, viewer, clip, glb_file, exports, summary = results(job)
        yield (strip, "\n".join(job.lines[-400:]), gallery, viewer, clip, glb_file, exports, summary, job.id,
               gr.update(choices=list_runs(manager), value=job.id))

    def submit(**kwargs):
        try:
            return manager.submit(**kwargs)
        except (ValueError, KeyError) as e:
            raise gr.Error(str(e))

    def preview_images(text, backend, url, model, size, steps, count, seed):
        if not (text or "").strip():
            raise gr.Error("Describe the character first.")
        try:
            draft = manager.draft_images(text.strip(), int(count), int(seed) if seed is not None else None,
                                         image_spec(backend, url, model, size, steps))
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"Image backend failed: {e}")
        paths = [str(manager.draft_path(draft["id"], k)) for k in range(draft["count"])]
        return paths, {"id": draft["id"], "index": 0, "prompt": text.strip()}

    def pick_candidate(state, evt: gr.SelectData):
        return {**state, "index": evt.index} if state else None

    def start_text(text, draft, backend, url, model, size, steps, name, *rest):
        text = (text or "").strip()
        if not text:
            raise gr.Error("Describe the character first.")
        use_draft = bool(draft) and draft.get("prompt") == text
        job = submit(prompt=text, draft=(draft["id"], draft["index"]) if use_draft else None,
                     name=name or slug(text)[:32], settings=settings(*rest),
                     image_spec=image_spec(backend, url, model, size, steps))
        yield from follow(job)

    def start_image(image, mode, name, *rest):
        if not image:
            raise gr.Error("Upload an image of the character.")
        s = settings(*rest)
        s.mode = mode
        yield from follow(submit(image=Path(image), name=name or None, settings=s))

    def start_splat(ply, reference, name, *rest):
        if not ply:
            raise gr.Error("Upload a 3D Gaussian splat (.ply).")
        yield from follow(submit(ply=Path(ply), reference=Path(reference) if reference else None,
                                 name=name or None, settings=settings(*rest)))

    def continue_run(source, stage, *rest):
        if not source:
            raise gr.Error("Pick an earlier run to continue.")
        try:
            job = manager.continue_job(source, stage, settings(*rest))
        except (ValueError, KeyError, SystemExit) as e:
            raise gr.Error(str(e))
        yield from follow(job)

    def show_run(job_id):
        empty = ("", [], EMPTY_VIEWER, gr.update(choices=[], value=None), None, None, "", "", "")
        if not job_id:
            return empty
        try:
            job = manager.get(job_id)
        except KeyError:
            return empty
        strip, gallery, viewer, clip, glb_file, exports, summary = results(job)
        log = "\n".join(job.lines[-400:]) or f"{job.id}: {job.status}"
        return strip, gallery, viewer, clip, glb_file, exports, summary, log, job.id

    def pick_clip(clip, job_id):
        if not job_id:
            return gr.update()
        job = manager.get(job_id)
        return viewer_html(best_glb(job.folder, job.name), clip)

    def cancel(job_id):
        if job_id:
            try:
                manager.cancel(job_id)
            except KeyError:
                pass

    def backend_status(backend, url):
        spec = ImageSpec(backend=backend, url=url or None, command=spec0.command, workflow=spec0.workflow)
        p = probe(spec)
        where = spec.base_url or spec.command or "not configured"
        if p.get("ok"):
            models = ", ".join(m for m in p.get("models", [])[:3] if m)
            return f"🟢 {backend} at {where}" + (f" ({models})" if models else "")
        return f"⚪ {backend} at {where} is not answering; generating from text needs it running"

    with gr.Blocks(title="assetgen") as demo:
        gr.Markdown(f"## assetgen <small style='color:#888'>{version()}</small>\n"
                    "Text or a picture in; a textured, rigged character with looping animation clips out, "
                    "ready for Unity and Godot. [API reference](/docs)")
        job_state = gr.State("")
        draft_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Tab("From text"):
                    text = gr.Textbox(label="Describe the character", lines=2, max_lines=6,
                                      placeholder="a goblin shaman with a lantern and a hooded cloak")
                    with gr.Row():
                        count = gr.Slider(1, 8, value=4, step=1, label="Candidates")
                        img_seed = gr.Number(value=-1, precision=0, label="Image seed (-1: random)")
                    preview = gr.Button("Preview images")
                    candidates = gr.Gallery(label="Pick one, or generate straight away", columns=4, height=220,
                                            allow_preview=False, object_fit="contain")
                    go_text = gr.Button("Generate character", variant="primary")
                    with gr.Accordion("Image generator", open=False):
                        backend = gr.Dropdown(sorted(BACKENDS), value=spec0.backend, label="Backend",
                                              info="sdcpp: stable-diffusion.cpp's sd-server (also A1111, Forge)")
                        backend_url = gr.Textbox(value=spec0.url or "", label="URL", placeholder=spec0.base_url,
                                                 max_lines=1)
                        with gr.Row():
                            image_model = gr.Textbox(value=spec0.model or "", label="Model", max_lines=1,
                                                     placeholder="server default")
                            image_size = gr.Dropdown(["512x512", "768x768", "1024x1024", "832x1216"],
                                                     value=f"{spec0.width}x{spec0.height}", label="Size",
                                                     allow_custom_value=True)
                            image_steps = gr.Number(value=spec0.steps, precision=0, label="Steps")
                        backend_info = gr.Markdown(backend_status(spec0.backend, spec0.url))
                with gr.Tab("From an image"):
                    image = gr.Image(type="filepath", label="Character image", height=300)
                    mode = gr.Radio(["synthetic", "real"], value="synthetic", label="Image type",
                                    info="synthetic: renders and generated art; real: photographs")
                    go_image = gr.Button("Generate character", variant="primary")
                with gr.Tab("From a splat"):
                    ply = gr.File(label="3D Gaussian splat (.ply)", file_types=[".ply"], type="filepath")
                    reference = gr.Image(type="filepath", label="Idol's input image (optional, for facing)",
                                         height=200)
                    go_splat = gr.Button("Generate character", variant="primary")
                with gr.Tab("Continue a run"):
                    with gr.Row():
                        runs = gr.Dropdown(list_runs(manager), label="Earlier run", scale=5,
                                           info="Selecting one shows its results")
                        refresh = gr.Button("↻", scale=1, min_width=40)
                    stage = gr.Radio([(label, s) for s, label in STAGE_LABELS.items()], value="animate",
                                     label="Start from",
                                     info="Earlier stages are copied into a new run; the one you pick stays as it is. "
                                          "The settings below apply to the stages that run.")
                    go_continue = gr.Button("Run from this stage", variant="primary")
                name = gr.Textbox(label="Name (new runs)", placeholder="from the description or file name",
                                  max_lines=1)
                prompts = gr.Textbox(value="\n".join(DEFAULT_PROMPTS), lines=4, max_lines=10,
                                     label="Motions, one per line",
                                     info='Start with "An object" and describe one motion. Clips loop seamlessly; '
                                          'end a line with [once] for a one-shot (attack, jump).')
                with gr.Row():
                    loop = gr.Checkbox(value=True, label="Seamless loops",
                                       info="Loop length follows each motion's own cycle")
                    loop_frames = gr.Number(value=0, precision=0, minimum=0, maximum=600,
                                            label="Loop length (frames, 30 fps)",
                                            info="0: from each motion's own cycle (3 s if it has none)")
                    root_motion = gr.Checkbox(value=False, label="Keep root motion",
                                              info="Off: loops play in place; their speed is in the character file")
                with gr.Row():
                    height = gr.Number(value=1.8, label="Height (m)", minimum=0.01)
                    device = gr.Dropdown(devices, value=devices[0], label="Device")
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
                        rig_candidates = gr.Number(value=3, precision=0, minimum=1, label="Rig candidates")
                        rig_class = gr.Textbox(label="Rig template", placeholder="articulation (any shape)",
                                               max_lines=1)
                        cfg_scale = gr.Number(value=3.0, label="Prompt strength")
                    llm_url = gr.Textbox(label="Joint-label LLM endpoint (optional)", max_lines=1,
                                         placeholder="http://localhost:8080/v1 (llama.cpp)")
            with gr.Column(scale=6):
                with gr.Row():
                    strip = gr.HTML("")
                    stop = gr.Button("Cancel", size="sm", scale=0, min_width=80)
                viewer = gr.HTML(EMPTY_VIEWER)
                with gr.Row():
                    clip = gr.Dropdown([], label="Clip", interactive=True)
                    download = gr.File(label="Character (.glb)", interactive=False)
                    exports = gr.File(label="For Unity / Godot (.zip)", file_count="multiple", interactive=False)
                summary = gr.Markdown("")
                gallery = gr.Gallery(label="Stages", columns=3, height=300, object_fit="contain")
                log = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)

        common = [prompts, height, device, loop, loop_frames, root_motion, seed, repetitions, faces, texture, voxels,
                  beams, rig_candidates, rig_class, cfg_scale, llm_url]
        image_controls = [backend, backend_url, image_model, image_size, image_steps]
        outputs = [strip, log, gallery, viewer, clip, download, exports, summary, job_state, runs]
        preview.click(preview_images, [text, *image_controls, count, img_seed], [candidates, draft_state])
        candidates.select(pick_candidate, draft_state, draft_state)
        go_text.click(start_text, [text, draft_state, *image_controls, name, *common], outputs)
        go_image.click(start_image, [image, mode, name, *common], outputs)
        go_splat.click(start_splat, [ply, reference, name, *common], outputs)
        go_continue.click(continue_run, [runs, stage, *common], outputs)
        stop.click(cancel, job_state, None)
        backend.change(backend_status, [backend, backend_url], backend_info)
        backend_url.blur(backend_status, [backend, backend_url], backend_info)
        shown = [strip, gallery, viewer, clip, download, exports, summary, log, job_state]
        runs.input(show_run, runs, shown)
        refresh.click(lambda: gr.update(choices=list_runs(manager)), None, runs)
        clip.change(pick_clip, [clip, job_state], viewer)
    return demo


def main(argv=None) -> None:
    """``assetgen webui`` is the server, web UI included; kept so existing scripts keep working."""
    from .server import main as serve

    serve(argv)


if __name__ == "__main__":
    main()
