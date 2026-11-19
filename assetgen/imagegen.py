"""Text to picture, through whichever image generator you run.

The 3D stages need one picture of one thing, whole and on a plain background. This
module frames a short description into such a prompt for the kind of asset being
made (a character, a prop, a diorama) and asks an image backend for it. Given a
starting picture it redraws that instead (image to image), which is how a rough
crop or a sketch becomes a clean picture of the same thing in the same style.

  sdcpp     stable-diffusion.cpp's sd-server: loads its model for the picture and unloads it after,
            so the GPU is free for the 3D stages (--image-keep-loaded to keep it)
  a1111     A1111, Forge, SD.Next: /sdapi/v1/txt2img and img2img (sd-server serves them too)
  openai    any OpenAI-compatible /v1/images/generations and /edits endpoint (OpenAI, LocalAI, sd-server)
  comfyui   a ComfyUI server running your workflow (API format, with {{prompt}} placeholders)
  command   any program that writes an image: "sd-cli -p {prompt} -o {out} --seed {seed}"

More backends register with :func:`register` (and :func:`register_editor` for image to
image). API keys are read from the environment (``ASSETGEN_IMAGE_API_KEY``, or
``OPENAI_API_KEY`` for openai), never from arguments, so they stay out of run manifests.

    python -m assetgen.imagegen --prompt "a goblin with a lantern" --out run/0_image --backend sdcpp
    python -m assetgen.imagegen --prompt "a wooden barrel" --kind prop --init crop.png --strength 0.5 --out run/0_image
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

STYLE = ("{prompt}, full body character, standing in a relaxed A-pose, seen from the front at a slight "
         "three-quarter angle, the whole figure in frame with space around it, centered, plain white "
         "background, soft even studio lighting, 3D game character render")
NEGATIVE = ("cropped, cut off, out of frame, close-up, portrait, multiple characters, crowd, scenery, "
            "detailed background, text, watermark, logo, border, cast shadow, blurry, lowres")
# How a description is framed for each kind of picture, and what to steer away from. The first three
# feed the 3D stages; "concept" draws a whole environment to take apart, "texture" a tileable surface.
STYLES = {
    "character": STYLE,
    "prop": ("{prompt}, a single game prop, the whole object in frame with space around it, centered, seen from "
             "the front at a slight three-quarter angle from slightly above, plain white background, soft even "
             "studio lighting, 3D game asset render"),
    "diorama": ("{prompt}, an isometric diorama on a small square base, the whole scene in frame with space around "
                "it, centered, seen at a three-quarter angle from above, plain white background, soft even studio "
                "lighting, 3D game environment render"),
    "concept": ("{prompt}, game environment concept art, seen from an elevated three-quarter view, the whole area "
                "in frame, every object whole and clearly separated from the others, open ground between them, "
                "soft even daylight, 3D game environment render"),
    "texture": ("{prompt}, seamless tileable texture, flat top-down view of the surface, even diffuse lighting, "
                "no shadows, no perspective, game texture"),
}
NEGATIVES = {
    "character": NEGATIVE,
    "prop": ("cropped, cut off, out of frame, close-up, multiple objects, person, character, hands, scenery, "
             "detailed background, text, watermark, logo, border, cast shadow, blurry, lowres"),
    "diorama": "cropped, cut off, out of frame, close-up, text, watermark, logo, border, picture frame, blurry, lowres",
    "concept": ("people, characters, crowd, fog, haze, close-up, cropped objects, overlapping objects, text, "
                "watermark, logo, border, blurry, lowres"),
    "texture": "perspective, horizon, objects, shadows, vignette, seams, border, text, watermark, logo, blurry, lowres",
}
DEFAULT_URLS = {"sdcpp": "http://127.0.0.1:1234", "a1111": "http://127.0.0.1:7860",
                "openai": "https://api.openai.com/v1", "comfyui": "http://127.0.0.1:8188"}
TIMEOUT = float(os.environ.get("ASSETGEN_IMAGE_TIMEOUT", 900))
STATE_DIR = Path(os.environ.get("ASSETGEN_STATE_DIR") or Path.home() / ".cache" / "assetgen")


@dataclass
class ImageSpec:
    """Where and how to generate. Everything but ``backend`` is optional."""
    backend: str = "sdcpp"
    url: str | None = None
    model: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str | None = None
    negative: str | None = None       # default: the kind's (NEGATIVES)
    workflow: str | None = None       # comfyui: path to an API-format workflow
    command: str | None = None        # command: program template
    style: bool = True                # frame the prompt for its kind (STYLES)
    keep_loaded: bool = False         # sdcpp: leave the model loaded after generating
    kind: str = "character"           # a key of STYLES: how the prompt is framed
    edit_workflow: str | None = None  # comfyui, image to image: a workflow with {{image}} and {{denoise}}
    edit_command: str | None = None   # command, image to image: a template with {init} and {strength}

    @property
    def base_url(self) -> str:
        return (self.url or DEFAULT_URLS.get(self.backend, "")).rstrip("/")

    @classmethod
    def from_env(cls, **overrides) -> "ImageSpec":
        env = {"backend": os.environ.get("ASSETGEN_IMAGE_BACKEND"), "url": os.environ.get("ASSETGEN_IMAGE_URL"),
               "model": os.environ.get("ASSETGEN_IMAGE_MODEL"), "workflow": os.environ.get("ASSETGEN_COMFY_WORKFLOW"),
               "command": os.environ.get("ASSETGEN_IMAGE_COMMAND"),
               "edit_workflow": os.environ.get("ASSETGEN_COMFY_EDIT_WORKFLOW"),
               "edit_command": os.environ.get("ASSETGEN_IMAGE_EDIT_COMMAND"),
               "keep_loaded": os.environ.get("ASSETGEN_IMAGE_KEEP_LOADED", "").lower() in ("1", "true", "yes")}
        values = {k: v for k, v in env.items() if v}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and k != "negative"}


def frame(prompt: str, spec: ImageSpec) -> str:
    if not spec.style:
        return prompt
    if spec.kind not in STYLES:
        raise ValueError(f"unknown kind {spec.kind!r}; one of {', '.join(STYLES)}")
    return STYLES[spec.kind].format(prompt=prompt.strip().rstrip("."))


def negative_prompt(spec: ImageSpec) -> str:
    return spec.negative if spec.negative is not None else NEGATIVES.get(spec.kind, NEGATIVE)


def _api_key(backend: str) -> str | None:
    fallback = os.environ.get("OPENAI_API_KEY") if backend == "openai" else None
    return os.environ.get("ASSETGEN_IMAGE_API_KEY") or fallback


def _headers(backend: str) -> dict:
    key = _api_key(backend)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _decode(b64: str) -> bytes:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _check(response):
    if response.status_code >= 400:
        raise RuntimeError(f"{response.request.method} {response.url} -> {response.status_code}: {response.text[:400]}")
    return response


class ModelNotLoaded(RuntimeError):
    """sd-server has no model loaded (another job unloaded it, or nothing was ever loaded)."""


def _sdapi_body(prompt: str, spec: ImageSpec, seed: int, count: int) -> dict:
    body = {"prompt": prompt, "negative_prompt": negative_prompt(spec), "width": spec.width,
            "height": spec.height, "seed": seed, "batch_size": count}
    if spec.steps:
        body["steps"] = spec.steps
    if spec.cfg_scale:
        body["cfg_scale"] = spec.cfg_scale
    if spec.sampler:
        body["sampler_name"] = spec.sampler
    if spec.model and spec.backend != "sdcpp":
        body["override_settings"] = {"sd_model_checkpoint": spec.model}
    return body


def _sdapi_call(endpoint: str, body: dict, spec: ImageSpec, count: int) -> list[bytes]:
    import requests

    r = requests.post(f"{spec.base_url}/sdapi/v1/{endpoint}", json=body, headers=_headers(spec.backend),
                      timeout=TIMEOUT)
    if r.status_code == 503 and "model_not_loaded" in r.text:
        raise ModelNotLoaded(f"sd-server at {spec.base_url} has no model loaded")
    return [_decode(b) for b in _check(r).json()["images"]][:count]


def sdapi(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """A1111's /sdapi/v1/txt2img, as served by sd-server, A1111, Forge and SD.Next."""
    return _sdapi_call("txt2img", _sdapi_body(prompt, spec, seed, count), spec, count)


def sdapi_edit(prompt: str, spec: ImageSpec, seed: int, count: int, init: bytes, strength: float,
               mask: bytes | None = None) -> list[bytes]:
    """A1111's /sdapi/v1/img2img: redraw ``init`` (with ``mask``, only where it is white)."""
    body = _sdapi_body(prompt, spec, seed, count)
    body.update(init_images=[base64.b64encode(init).decode()], denoising_strength=float(strength))
    if mask is not None:
        body["mask"] = base64.b64encode(mask).decode()
    return _sdapi_call("img2img", body, spec, count)


def sdcpp(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """stable-diffusion.cpp's sd-server, its model loaded for the picture and unloaded after (assetgen.sdcpp)."""
    from . import sdcpp

    return sdcpp.with_model(spec, lambda: sdapi(prompt, spec, seed, count))


def sdcpp_edit(prompt: str, spec: ImageSpec, seed: int, count: int, init: bytes, strength: float,
               mask: bytes | None = None) -> list[bytes]:
    from . import sdcpp

    return sdcpp.with_model(spec, lambda: sdapi_edit(prompt, spec, seed, count, init, strength, mask))


def _openai_base(spec: ImageSpec) -> str:
    return spec.base_url if spec.base_url.endswith("/v1") else f"{spec.base_url}/v1"


def _openai_images(r) -> list[bytes]:
    import requests

    images = []
    for item in r.json()["data"]:
        if item.get("b64_json"):
            images.append(_decode(item["b64_json"]))
        elif item.get("url"):
            images.append(_check(requests.get(item["url"], timeout=TIMEOUT)).content)
    return images


def openai_images(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """OpenAI-compatible /v1/images/generations (OpenAI, LocalAI, sd-server's OpenAI layer)."""
    import requests

    body = {"prompt": prompt, "n": count, "size": f"{spec.width}x{spec.height}"}
    if spec.model:
        body["model"] = spec.model
    if (spec.model or "").startswith("dall-e"):
        body["response_format"] = "b64_json"
    return _openai_images(_check(requests.post(f"{_openai_base(spec)}/images/generations", json=body,
                                               headers=_headers(spec.backend), timeout=TIMEOUT)))


def openai_edit(prompt: str, spec: ImageSpec, seed: int, count: int, init: bytes, strength: float,
                mask: bytes | None = None) -> list[bytes]:
    """OpenAI-compatible /v1/images/edits. The API takes no strength: the model decides how much to change."""
    import requests

    data = {"prompt": prompt, "n": str(count), "size": f"{spec.width}x{spec.height}"}
    if spec.model:
        data["model"] = spec.model
    files = {"image": ("image.png", init, "image/png")}
    if mask is not None:
        # The edits API repaints where its mask is transparent; assetgen's masks are white there.
        files["mask"] = ("mask.png", _transparent_where_white(mask), "image/png")
    return _openai_images(_check(requests.post(f"{_openai_base(spec)}/images/edits", data=data, files=files,
                                               headers=_headers(spec.backend), timeout=TIMEOUT)))


def _transparent_where_white(mask: bytes) -> bytes:
    from PIL import Image, ImageOps

    m = Image.open(io.BytesIO(mask)).convert("L")
    rgba = Image.new("RGBA", m.size, (0, 0, 0, 255))
    rgba.putalpha(ImageOps.invert(m))
    return _png(rgba)


def _fill(node, values: dict):
    """Replace {{name}} placeholders in a ComfyUI workflow; a value that is the whole string keeps its type."""
    if isinstance(node, dict):
        return {k: _fill(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, values) for v in node]
    if isinstance(node, str):
        for k, v in values.items():
            token = "{{" + k + "}}"
            if node == token:
                return v
            node = node.replace(token, str(v))
    return node


def comfyui(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """Queue the workflow once per image and collect its saved images."""
    if not spec.workflow:
        raise ValueError("comfyui needs a workflow: an API-format JSON with {{prompt}} (and optionally "
                         "{{negative}}, {{seed}}, {{width}}, {{height}}, {{steps}}) placeholders")
    return _comfy_run(spec.workflow, prompt, spec, seed, count, {})


def comfyui_edit(prompt: str, spec: ImageSpec, seed: int, count: int, init: bytes, strength: float,
                 mask: bytes | None = None) -> list[bytes]:
    """Upload ``init`` and run the image-to-image workflow: its LoadImage takes {{image}} and its sampler
    {{denoise}}; {{mask}} names the uploaded mask when there is one."""
    import requests

    workflow = spec.edit_workflow or spec.workflow
    if not workflow or "{{image}}" not in Path(workflow).read_text():
        raise ValueError("comfyui image to image needs a workflow with an {{image}} placeholder (a LoadImage node) "
                         "and {{denoise}}: --comfy-edit-workflow or ASSETGEN_COMFY_EDIT_WORKFLOW")

    def upload(data: bytes, name: str) -> str:
        r = _check(requests.post(f"{spec.base_url}/upload/image", files={"image": (name, data, "image/png")},
                                 data={"overwrite": "true"}, timeout=60))
        info = r.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    extra = {"image": upload(init, f"assetgen-{uuid.uuid4().hex}.png"), "denoise": float(strength)}
    if mask is not None:
        extra["mask"] = upload(mask, f"assetgen-{uuid.uuid4().hex}-mask.png")
    return _comfy_run(workflow, prompt, spec, seed, count, extra)


def _comfy_run(workflow: str, prompt: str, spec: ImageSpec, seed: int, count: int, extra: dict) -> list[bytes]:
    import requests

    template = json.loads(Path(workflow).read_text())
    images = []
    for k in range(count):
        values = {"prompt": prompt, "negative": negative_prompt(spec), "seed": seed + k, "width": spec.width,
                  "height": spec.height, "steps": spec.steps or 30, "cfg": spec.cfg_scale or 5.0, **extra}
        r = _check(requests.post(f"{spec.base_url}/prompt", json={"prompt": _fill(template, values),
                                                                   "client_id": uuid.uuid4().hex}, timeout=60))
        prompt_id = r.json()["prompt_id"]
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            history = _check(requests.get(f"{spec.base_url}/history/{prompt_id}", timeout=60)).json()
            if prompt_id in history:
                break
            time.sleep(1.0)
        else:
            raise TimeoutError(f"ComfyUI did not finish {prompt_id} within {TIMEOUT:.0f}s")
        saved = [img for output in history[prompt_id].get("outputs", {}).values()
                 for img in output.get("images", []) if img.get("type") == "output"]
        if not saved:
            raise RuntimeError(f"ComfyUI workflow {workflow} saved no image (it needs a SaveImage node)")
        images.append(_check(requests.get(f"{spec.base_url}/view", params=saved[0], timeout=60)).content)
    return images


def split_command(template: str) -> list[str]:
    """Split a command line into arguments. On Windows, backslashes in paths stay as they are and
    quotes only group (POSIX shlex would read the backslashes as escapes)."""
    if os.name != "nt":
        return shlex.split(template)
    parts = shlex.split(template, posix=False)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


def _run_command(template: str, prompt: str, spec: ImageSpec, seed: int, count: int, extra: dict) -> list[bytes]:
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(count):
            out = Path(tmp) / f"image_{k}.png"
            values = {"prompt": prompt, "negative": negative_prompt(spec), "out": str(out), "seed": seed + k,
                      "width": spec.width, "height": spec.height, "steps": spec.steps or 20, **extra}
            args = [part.format(**values) for part in split_command(template)]
            subprocess.run(args, check=True, timeout=TIMEOUT)
            images.append(out.read_bytes())
    return images


def command(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """Run a program once per image. The template's {prompt}, {negative}, {out}, {seed}, {width},
    {height} and {steps} are filled in; the program writes a PNG to {out}."""
    if not spec.command:
        raise ValueError('command needs a template, e.g. "sd-cli -m model.gguf -p {prompt} -o {out} -s {seed}"')
    return _run_command(spec.command, prompt, spec, seed, count, {})


def command_edit(prompt: str, spec: ImageSpec, seed: int, count: int, init: bytes, strength: float,
                 mask: bytes | None = None) -> list[bytes]:
    """The image-to-image template also gets {init} (a PNG path), {strength} and {mask} (a path, or empty)."""
    template = spec.edit_command or spec.command
    if not template or "{init}" not in template:
        raise ValueError('command image to image needs a template with {init}, e.g. "sd-cli -m model.gguf -i {init} '
                         '--strength {strength} -p {prompt} -o {out}": --image-edit-command or '
                         'ASSETGEN_IMAGE_EDIT_COMMAND')
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "init.png").write_bytes(init)
        extra = {"init": str(Path(tmp) / "init.png"), "strength": strength, "mask": ""}
        if mask is not None:
            (Path(tmp) / "mask.png").write_bytes(mask)
            extra["mask"] = str(Path(tmp) / "mask.png")
        return _run_command(template, prompt, spec, seed, count, extra)


BACKENDS: dict[str, Callable[[str, ImageSpec, int, int], list[bytes]]] = {
    "sdcpp": sdcpp, "a1111": sdapi, "openai": openai_images, "comfyui": comfyui, "command": command,
}
# Image to image: fn(prompt, spec, seed, count, init, strength, mask) -> encoded images. ``init`` and
# ``mask`` are PNG bytes at the spec's size; the mask is white where the picture may change.
EDITORS: dict[str, Callable[..., list[bytes]]] = {
    "sdcpp": sdcpp_edit, "a1111": sdapi_edit, "openai": openai_edit, "comfyui": comfyui_edit,
    "command": command_edit,
}


def register(name: str, fn: Callable[[str, ImageSpec, int, int], list[bytes]], default_url: str | None = None):
    """Add a backend: ``fn(prompt, spec, seed, count)`` returns encoded images (PNG or JPEG bytes)."""
    BACKENDS[name] = fn
    if default_url:
        DEFAULT_URLS[name] = default_url


def register_editor(name: str, fn: Callable[..., list[bytes]]) -> None:
    """Add image to image to a backend: ``fn(prompt, spec, seed, count, init, strength, mask)``."""
    EDITORS[name] = fn


def probe(spec: ImageSpec) -> dict:
    """Whether a backend answers, and the models it reports. Never raises."""
    import requests

    try:
        if spec.backend == "sdcpp":
            r = requests.get(f"{spec.base_url}/sdcpp/v1/models/status", headers=_headers(spec.backend), timeout=3)
            if r.ok:
                status = r.json()
                recipe = status.get("recipe")
                models = [recipe] if isinstance(recipe, str) else []
                return {"ok": True, "status": r.status_code, "models": models, "model_state": status.get("state")}
            r = requests.get(f"{spec.base_url}/sdapi/v1/sd-models", headers=_headers(spec.backend), timeout=3)
            models = [m.get("model_name") or m.get("title") for m in r.json()] if r.ok else []
        elif spec.backend == "a1111":
            r = requests.get(f"{spec.base_url}/sdapi/v1/sd-models", headers=_headers(spec.backend), timeout=3)
            models = [m.get("model_name") or m.get("title") for m in r.json()] if r.ok else []
        elif spec.backend == "openai":
            r = requests.get(f"{_openai_base(spec)}/models", headers=_headers(spec.backend), timeout=5)
            models = [m["id"] for m in r.json().get("data", [])][:50] if r.ok else []
        elif spec.backend == "comfyui":
            r = requests.get(f"{spec.base_url}/system_stats", timeout=3)
            models = []
        else:
            return {"ok": bool(spec.command), "models": []}
        return {"ok": r.ok, "status": r.status_code, "models": models}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def fit_canvas(image, width: int, height: int, background=(255, 255, 255)):
    """``image`` scaled to fit ``width`` x ``height``, centred on a plain canvas; transparent parts take
    the background colour."""
    from PIL import Image

    image = image.convert("RGBA")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2), resized)
    return canvas


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def generate(prompt: str, spec: ImageSpec, seed: int | None = None, count: int = 1, init=None,
             strength: float = 0.6, mask=None) -> list:
    """``count`` PIL images of what ``prompt`` describes. With ``init`` (a PIL image or a path) the backend
    redraws that picture instead: ``strength`` 0 keeps it, 1 ignores it; ``mask`` (white where the picture may
    change) limits where."""
    from PIL import Image

    if spec.backend not in BACKENDS:
        raise ValueError(f"unknown image backend {spec.backend!r}; known: {', '.join(sorted(BACKENDS))}")
    seed = random.randint(0, 2**31 - 1) if seed is None or seed < 0 else seed
    if init is None:
        raw = BACKENDS[spec.backend](frame(prompt, spec), spec, seed, count)
    else:
        if spec.backend not in EDITORS:
            raise ValueError(f"image backend {spec.backend!r} cannot redraw a picture (no image to image)")
        init = Image.open(init) if isinstance(init, (str, Path)) else init
        init_png = _png(fit_canvas(init, spec.width, spec.height))
        mask_png = None
        if mask is not None:
            mask = Image.open(mask) if isinstance(mask, (str, Path)) else mask
            mask_png = _png(fit_canvas(mask.convert("L").convert("RGB"), spec.width, spec.height, (0, 0, 0)))
        raw = EDITORS[spec.backend](frame(prompt, spec), spec, seed, count, init_png, float(strength), mask_png)
    if not raw:
        raise RuntimeError(f"{spec.backend} returned no images")
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in raw]


def write(prompt: str, out_dir: Path, spec: ImageSpec, seed: int | None = None, count: int = 1,
          log=print, init: Path | None = None, strength: float = 0.6, mask: Path | None = None) -> Path:
    """Generate into ``out_dir``: image.png (the first), candidate_<k>.png when count > 1, image.json.
    With ``init``, redraw that picture (see :func:`generate`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = random.randint(0, 2**31 - 1) if seed is None or seed < 0 else seed
    how = f"redrawing {Path(init).name} at strength {strength}" if init else "generating"
    log(f"[image] {spec.backend} at {spec.base_url or spec.command}, {how}: {prompt!r} (seed {seed})")
    t0 = time.time()
    images = generate(prompt, spec, seed, count, init=init, strength=strength, mask=mask)
    for k, img in enumerate(images if count > 1 else []):
        img.save(out_dir / f"candidate_{k}.png")
    images[0].save(out_dir / "image.png")
    meta = {"prompt": prompt, "kind": spec.kind, "framed_prompt": frame(prompt, spec),
            "negative_prompt": negative_prompt(spec), "seed": seed, "backend": spec.backend, "model": spec.model,
            "size": [spec.width, spec.height], "seconds": round(time.time() - t0, 1)}
    if init:
        meta.update(init=Path(init).name, strength=strength, mask=Path(mask).name if mask else None)
    (out_dir / "image.json").write_text(json.dumps(meta, indent=2))
    log(f"[image] wrote {out_dir / 'image.png'} in {time.time() - t0:.0f}s")
    return out_dir / "image.png"


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", required=True, help="what to draw, e.g. 'a goblin with a lantern'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default=None, choices=sorted(STYLES), help="how to frame the prompt (default character)")
    ap.add_argument("--init", default=None, help="redraw this picture instead of starting from noise")
    ap.add_argument("--strength", type=float, default=0.6, help="with --init: 0 keeps the picture, 1 ignores it")
    ap.add_argument("--mask", default=None, help="with --init: white where the picture may change")
    ap.add_argument("--backend", default=None, help=f"one of {', '.join(BACKENDS)} (default sdcpp)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg_scale", type=float, default=None)
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--workflow", default=None)
    ap.add_argument("--command", default=None)
    ap.add_argument("--edit_workflow", default=None)
    ap.add_argument("--edit_command", default=None)
    ap.add_argument("--raw", action="store_true", help="send the prompt as written, without the framing")
    ap.add_argument("--keep_loaded", action="store_true", help="sdcpp: leave sd-server's model loaded afterwards")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--count", type=int, default=1)
    a = ap.parse_args(argv)
    spec = ImageSpec.from_env(backend=a.backend, url=a.url, model=a.model, width=a.width, height=a.height,
                              steps=a.steps, cfg_scale=a.cfg_scale, sampler=a.sampler, workflow=a.workflow,
                              command=a.command, style=False if a.raw else None,
                              keep_loaded=True if a.keep_loaded else None, kind=a.kind,
                              edit_workflow=a.edit_workflow, edit_command=a.edit_command)
    write(a.prompt, Path(a.out), spec, a.seed, a.count, init=Path(a.init) if a.init else None,
          strength=a.strength, mask=Path(a.mask) if a.mask else None)


if __name__ == "__main__":
    main()
