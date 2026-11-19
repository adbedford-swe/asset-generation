"""Build a scene from its spec: every object as a prop, the ground, the layout, one glTF.

    <out>/
      scene.json            the spec, as built (paths resolved)
      props/<id>/           each object's own run (image, splat, mesh, <id>.glb, <id>.assetgen), resumable
      ground/               texture.png (tileable), image.json
      assets/<id>.glb       each object as the scene uses it (harmonized), with its .assetgen
      <name>.glb            the whole scene: ground and every placed object (instanced)
      <name>.assetgen       the scene file (format assetgen.scene)
      <name>.png            a preview: three-quarter view and plan view
      logs/

Objects are made like ``assetgen run --kind prop``, each in its own resumable run, so
building again after editing the spec only redoes what changed: move an object or
change a scatter and only the layout and assembly run again; change a prompt and only
that object is remade. With several devices, objects are made side by side.
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..imagegen import ImageSpec
from ..pipeline import Cancelled, Options
from ..pipeline import run as run_pipeline
from . import harmonize as harmonize_
from .assemble import describe, ground_textured, preview, write_assets, write_glb
from .layout import Footprint, plan
from .spec import ObjectSpec, SceneSpec
from .terrain import heightfield, tileable

STEPS = ["props", "ground", "layout", "assemble"]


def prompt_for(o: ObjectSpec, spec: SceneSpec) -> str | None:
    text = o.prompt or (o.id.replace("_", " ") if o.redraw > 0 else None)
    if text and o.style and spec.style:
        text = f"{text.strip().rstrip('.')}, {spec.style}"
    return text


def object_options(o: ObjectSpec, k: int, spec: SceneSpec, out: Path, device: str, image_spec: ImageSpec,
                   mesh_args: list[str] | None = None, idol_steps: int | None = None) -> Options:
    folder = out / "props" / o.id
    image = Path(o.image) if o.image else None
    if image is not None and o.redraw > 0:
        image = folder / "0_image" / "image.png"  # made by redraw() before the run
    return Options(out=folder, text=prompt_for(o, spec) if not (o.image or o.ply) else None,
                   image_spec=replace(image_spec, kind=o.kind), image=image, ply=Path(o.ply) if o.ply else None,
                   reference=Path(o.reference) if o.reference else None, name=o.id, device=device,
                   seed=o.seed if o.seed is not None else spec.seed + k, kind=o.kind, height=o.height,
                   collider=o.collider, mesh_args=list(mesh_args or []), idol_steps=idol_steps)


def redraw(o: ObjectSpec, spec: SceneSpec, out: Path, image_spec: ImageSpec, log=print) -> None:
    """Redraw an object's picture in the scene's style (image to image), unless that is already done."""
    from ..imagegen import write

    folder = out / "props" / o.id / "0_image"
    meta = folder / "image.json"
    prompt = prompt_for(o, spec)
    if meta.exists():
        made = json.loads(meta.read_text())
        if made.get("prompt") == prompt and made.get("init") == Path(o.image).name and \
                made.get("strength") == o.redraw and (folder / "image.png").exists():
            return
    write(prompt, folder, replace(image_spec, kind=o.kind), spec.seed, init=Path(o.image), strength=o.redraw,
          log=log)


def adopt(o: ObjectSpec, out: Path, log=print) -> None:
    """Take an existing asset (a run folder, a .glb or its .assetgen) as the object's prop, unless it
    already was."""
    source = Path(o.asset)
    if source.is_dir():
        from ..export import find_asset

        source = find_asset(source)
    glb = source.with_suffix(".glb")
    folder = out / "props" / o.id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{o.id}.glb"
    record = folder / "adopted.json"
    made_from = {"source": str(glb.resolve()), "modified": glb.stat().st_mtime_ns, "height": o.height,
                 "kind": o.kind, "collider": o.collider}
    if record.exists() and json.loads(record.read_text()) == made_from and target.with_suffix(".assetgen").exists():
        return
    adopt_into(o, glb, target, log)
    record.write_text(json.dumps(made_from, indent=2))


def adopt_into(o: ObjectSpec, glb: Path, target: Path, log=print) -> None:
    from ..prop import finalize, load

    asset_file = glb.with_suffix(".assetgen")
    if asset_file.exists():
        data = json.loads(asset_file.read_text())
        if data.get("format") == "assetgen.prop" and abs(data["model"]["height"] - o.height) < 1e-3:
            load(asset_file)
            shutil.copy2(glb, target)
            data.update(name=o.id, model=dict(data["model"], file=target.name))
            target.with_suffix(".assetgen").write_text(json.dumps(data, indent=2))
            if glb.with_suffix(".png").exists():
                shutil.copy2(glb.with_suffix(".png"), target.with_suffix(".png"))
            return
    # Anything else (a character, a prop at another height, a bare mesh) is finalized as a prop at this height.
    finalize(glb, target, o.kind, o.height, o.collider, source={"asset": glb.name}, log=log)


def make_objects(spec: SceneSpec, out: Path, devices: list[str], image_spec: ImageSpec, log=print,
                 cancel: threading.Event | None = None, force: bool = False, mesh_args: list[str] | None = None,
                 idol_steps: int | None = None) -> None:
    """Every object as a prop, spread over ``devices``. Raises SystemExit naming the objects that failed."""
    work: queue.Queue = queue.Queue()
    for k, o in enumerate(spec.objects):
        work.put((k, o))
    failed: list[str] = []
    lock = threading.Lock()

    def worker(device: str):
        while True:
            try:
                k, o = work.get_nowait()
            except queue.Empty:
                return

            def say(message, name=o.id):
                log(f"[{name}] {message}")

            try:
                if cancel is not None and cancel.is_set():
                    raise Cancelled("cancelled")
                if o.asset:
                    adopt(o, out, say)
                    continue
                if o.image and o.redraw > 0:
                    redraw(o, spec, out, image_spec, say)
                run_pipeline(object_options(o, k, spec, out, device, image_spec, mesh_args, idol_steps),
                             force=force, log=say, cancel=cancel)
            except Cancelled:
                with lock:
                    failed.append(f"{o.id} (cancelled)")
            except (SystemExit, Exception) as e:  # one object failing does not stop the others
                with lock:
                    failed.append(f"{o.id}: {e}")

    threads = [threading.Thread(target=worker, args=(d,), daemon=True) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if cancel is not None and cancel.is_set():
        raise Cancelled("cancelled while making the objects")
    if failed:
        raise SystemExit("some objects could not be made:\n  " + "\n  ".join(failed))


def view_of(folder: Path) -> float:
    """Where the camera of an object's picture stood, as a yaw from the object's front: what the mesh stage found
    when it turned the object to face +Z. 0 (its front) when the object was not meshed here."""
    path = Path(folder) / "1_mesh" / "orientation.json"
    if not path.exists():
        return 0.0
    o = json.loads(path.read_text())
    return float(o["camera_yaw_deg"] - o["facing_yaw_deg"]) % 360


def make_ground_texture(spec: SceneSpec, out: Path, image_spec: ImageSpec, log=print) -> Path | None:
    """ground/texture.png: the given picture, or one drawn from the ground's prompt; tileable either way."""
    from PIL import Image

    from ..imagegen import write

    folder = out / "ground"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "texture.png"
    if spec.ground.texture:
        source = np.asarray(Image.open(spec.ground.texture).convert("RGB"))
        covers_all = spec.ground.tile >= max(spec.ground.size) - 1e-6  # one picture of the whole ground: no repeats
        Image.fromarray(source if covers_all else tileable(source)).save(target)
        return target
    if not spec.ground.prompt:
        return None
    prompt = spec.ground.prompt if not spec.style else f"{spec.ground.prompt.strip().rstrip('.')}, {spec.style}"
    meta = folder / "image.json"
    if not (meta.exists() and json.loads(meta.read_text()).get("prompt") == prompt and target.exists()):
        size = min(image_spec.width, image_spec.height, 768)
        write(prompt, folder, replace(image_spec, kind="texture", width=size, height=size), spec.seed, log=log)
        Image.fromarray(tileable(np.asarray(Image.open(folder / "image.png").convert("RGB")))).save(target)
    return target


def build(spec: SceneSpec, out: Path, devices: list[str] | None = None, image_spec: ImageSpec | None = None,
          log=print, cancel: threading.Event | None = None, on_stage=None, force: bool = False,
          mesh_args: list[str] | None = None, idol_steps: int | None = None) -> Path:
    """Build ``spec`` into ``out``; returns the scene file. ``on_stage(step, state)`` hears each of STEPS."""
    from PIL import Image

    from ..gltf import load_textured_glb
    from ..prop import load as load_prop

    out = Path(out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "scene.json").write_text(json.dumps(spec.to_dict(), indent=2))
    image_spec = image_spec or ImageSpec.from_env()
    devices = devices or ["cuda"]

    def step(name, fn):
        if cancel is not None and cancel.is_set():
            raise Cancelled(f"cancelled before {name}")
        log(f"== {name}")
        if on_stage:
            on_stage(name, "running")
        t0 = time.time()
        try:
            result = fn()
        except BaseException:
            if on_stage:
                on_stage(name, "failed")
            raise
        log(f"   {name} done in {time.time() - t0:.0f}s")
        if on_stage:
            on_stage(name, "done")
        return result

    step("props", lambda: make_objects(spec, out, devices, image_spec, log, cancel, force, mesh_args, idol_steps))
    objects = {o.id: load_prop(out / "props" / o.id / f"{o.id}.assetgen") for o in spec.objects}
    sources = {o.id: out / "props" / o.id / f"{o.id}.glb" for o in spec.objects}
    meshes = {k: load_textured_glb(p) for k, p in sources.items()}

    def ground():
        texture = make_ground_texture(spec, out, image_spec, log)
        field = heightfield(spec.ground.size, spec.ground.relief, spec.ground.feature, spec.ground.resolution,
                            spec.seed)
        return texture, field

    texture_path, field = step("ground", ground)
    layout = step("layout", lambda: plan(spec, {k: Footprint.of(v, view_of(out / "props" / k))
                                                 for k, v in objects.items()}, field))
    for w in layout.warnings:
        log(f"   warning: {w}")
    counts = {o.id: sum(1 for i in layout.instances if i.object == o.id) for o in spec.objects}
    log("   " + ", ".join(f"{k} x{n}" for k, n in counts.items()))

    def assemble():
        looks = {}
        if spec.harmonize > 0:
            own = {k: harmonize_.look(harmonize_.surface_colors(m)) for k, m in meshes.items()}
            if spec.concept:
                target = harmonize_.image_look(np.asarray(Image.open(spec.concept).convert("RGB")))
            else:
                target = harmonize_.pooled(list(own.values()))
            for k, m in meshes.items():
                meshes[k] = replace(m, texture=harmonize_.apply(m.texture, own[k], target, spec.harmonize))
                looks[k] = own[k].to_dict()
            log(f"   harmonized {len(meshes)} objects towards {'the concept' if spec.concept else 'their pooled'} "
                f"look at strength {spec.harmonize:g}")
        ground_tex = np.asarray(Image.open(texture_path).convert("RGB")) if texture_path else None
        ground_mesh = ground_textured(layout, spec, ground_tex)
        write_assets(out, objects, meshes, sources)
        glb = out / f"{spec.name}.glb"
        write_glb(glb, spec, layout, meshes, ground_mesh)
        Image.fromarray(preview(spec, layout, meshes, ground_mesh)).save(glb.with_suffix(".png"))
        data = describe(glb, spec, layout, objects,
                        texture_path.relative_to(out).as_posix() if texture_path else None, looks)
        path = glb.with_suffix(".assetgen")
        path.write_text(json.dumps(data, indent=2))
        return path

    path = step("assemble", assemble)
    log(f"   {len(layout.instances)} objects placed on {spec.ground.size[0]:g} x {spec.ground.size[1]:g} m: {path}")
    return path
