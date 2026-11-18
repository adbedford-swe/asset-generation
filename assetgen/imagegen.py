"""Text to character image, through whichever image generator you run.

The 3D stages need one picture of one character, whole and on a plain
background. This module frames a short description into such a prompt and asks
an image backend for it:

  sdcpp     stable-diffusion.cpp's sd-server (also A1111, Forge, SD.Next): /sdapi/v1/txt2img
  a1111     the same API, with A1111's default address
  openai    any OpenAI-compatible /v1/images/generations endpoint (OpenAI, LocalAI, sd-server)
  comfyui   a ComfyUI server running your workflow (API format, with {{prompt}} placeholders)
  command   any program that writes an image: "sd-cli -p {prompt} -o {out} --seed {seed}"

More backends register with :func:`register`. API keys are read from the environment
(``ASSETGEN_IMAGE_API_KEY``, or ``OPENAI_API_KEY`` for openai), never from arguments, so
they stay out of run manifests.

    python -m assetgen.imagegen --prompt "a goblin with a lantern" --out run/0_image --backend sdcpp
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
DEFAULT_URLS = {"sdcpp": "http://127.0.0.1:1234", "a1111": "http://127.0.0.1:7860",
                "openai": "https://api.openai.com/v1", "comfyui": "http://127.0.0.1:8188"}
TIMEOUT = float(os.environ.get("ASSETGEN_IMAGE_TIMEOUT", 900))


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
    negative: str = NEGATIVE
    workflow: str | None = None   # comfyui: path to an API-format workflow
    command: str | None = None    # command: program template
    style: bool = True            # frame the prompt as a full-body character on a plain background

    @property
    def base_url(self) -> str:
        return (self.url or DEFAULT_URLS.get(self.backend, "")).rstrip("/")

    @classmethod
    def from_env(cls, **overrides) -> "ImageSpec":
        env = {"backend": os.environ.get("ASSETGEN_IMAGE_BACKEND"), "url": os.environ.get("ASSETGEN_IMAGE_URL"),
               "model": os.environ.get("ASSETGEN_IMAGE_MODEL"), "workflow": os.environ.get("ASSETGEN_COMFY_WORKFLOW"),
               "command": os.environ.get("ASSETGEN_IMAGE_COMMAND")}
        values = {k: v for k, v in env.items() if v}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and k != "negative"}


def frame(prompt: str, spec: ImageSpec) -> str:
    return STYLE.format(prompt=prompt.strip().rstrip(".")) if spec.style else prompt


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


def sdapi(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """A1111's /sdapi/v1/txt2img, as served by sd-server, A1111, Forge and SD.Next."""
    import requests

    body = {"prompt": prompt, "negative_prompt": spec.negative, "width": spec.width, "height": spec.height,
            "seed": seed, "batch_size": count}
    if spec.steps:
        body["steps"] = spec.steps
    if spec.cfg_scale:
        body["cfg_scale"] = spec.cfg_scale
    if spec.sampler:
        body["sampler_name"] = spec.sampler
    if spec.model:
        body["override_settings"] = {"sd_model_checkpoint": spec.model}
    r = _check(requests.post(f"{spec.base_url}/sdapi/v1/txt2img", json=body, headers=_headers(spec.backend),
                             timeout=TIMEOUT))
    return [_decode(b) for b in r.json()["images"]][:count]


def openai_images(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """OpenAI-compatible /v1/images/generations (OpenAI, LocalAI, sd-server's OpenAI layer)."""
    import requests

    base = spec.base_url if spec.base_url.endswith("/v1") else f"{spec.base_url}/v1"
    body = {"prompt": prompt, "n": count, "size": f"{spec.width}x{spec.height}"}
    if spec.model:
        body["model"] = spec.model
    if (spec.model or "").startswith("dall-e"):
        body["response_format"] = "b64_json"
    r = _check(requests.post(f"{base}/images/generations", json=body, headers=_headers(spec.backend),
                             timeout=TIMEOUT))
    images = []
    for item in r.json()["data"]:
        if item.get("b64_json"):
            images.append(_decode(item["b64_json"]))
        elif item.get("url"):
            images.append(_check(requests.get(item["url"], timeout=TIMEOUT)).content)
    return images


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
    import requests

    if not spec.workflow:
        raise ValueError("comfyui needs a workflow: an API-format JSON with {{prompt}} (and optionally "
                         "{{negative}}, {{seed}}, {{width}}, {{height}}, {{steps}}) placeholders")
    template = json.loads(Path(spec.workflow).read_text())
    images = []
    for k in range(count):
        values = {"prompt": prompt, "negative": spec.negative, "seed": seed + k, "width": spec.width,
                  "height": spec.height, "steps": spec.steps or 30, "cfg": spec.cfg_scale or 5.0}
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
            raise RuntimeError(f"ComfyUI workflow {spec.workflow} saved no image (it needs a SaveImage node)")
        images.append(_check(requests.get(f"{spec.base_url}/view", params=saved[0], timeout=60)).content)
    return images


def split_command(template: str) -> list[str]:
    """Split a command line into arguments. On Windows, backslashes in paths stay as they are and
    quotes only group (POSIX shlex would read the backslashes as escapes)."""
    if os.name != "nt":
        return shlex.split(template)
    parts = shlex.split(template, posix=False)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


def command(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """Run a program once per image. The template's {prompt}, {negative}, {out}, {seed}, {width},
    {height} and {steps} are filled in; the program writes a PNG to {out}."""
    if not spec.command:
        raise ValueError('command needs a template, e.g. "sd-cli -m model.gguf -p {prompt} -o {out} -s {seed}"')
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(count):
            out = Path(tmp) / f"image_{k}.png"
            values = {"prompt": prompt, "negative": spec.negative, "out": str(out), "seed": seed + k,
                      "width": spec.width, "height": spec.height, "steps": spec.steps or 20}
            args = [part.format(**values) for part in split_command(spec.command)]
            subprocess.run(args, check=True, timeout=TIMEOUT)
            images.append(out.read_bytes())
    return images


BACKENDS: dict[str, Callable[[str, ImageSpec, int, int], list[bytes]]] = {
    "sdcpp": sdapi, "a1111": sdapi, "openai": openai_images, "comfyui": comfyui, "command": command,
}


def register(name: str, fn: Callable[[str, ImageSpec, int, int], list[bytes]], default_url: str | None = None):
    """Add a backend: ``fn(prompt, spec, seed, count)`` returns encoded images (PNG or JPEG bytes)."""
    BACKENDS[name] = fn
    if default_url:
        DEFAULT_URLS[name] = default_url


def probe(spec: ImageSpec) -> dict:
    """Whether a backend answers, and the models it reports. Never raises."""
    import requests

    try:
        if spec.backend in ("sdcpp", "a1111"):
            r = requests.get(f"{spec.base_url}/sdapi/v1/sd-models", headers=_headers(spec.backend), timeout=3)
            models = [m.get("model_name") or m.get("title") for m in r.json()] if r.ok else []
        elif spec.backend == "openai":
            base = spec.base_url if spec.base_url.endswith("/v1") else f"{spec.base_url}/v1"
            r = requests.get(f"{base}/models", headers=_headers(spec.backend), timeout=5)
            models = [m["id"] for m in r.json().get("data", [])][:50] if r.ok else []
        elif spec.backend == "comfyui":
            r = requests.get(f"{spec.base_url}/system_stats", timeout=3)
            models = []
        else:
            return {"ok": bool(spec.command), "models": []}
        return {"ok": r.ok, "status": r.status_code, "models": models}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def generate(prompt: str, spec: ImageSpec, seed: int | None = None, count: int = 1) -> list:
    """``count`` PIL images of the described character."""
    from PIL import Image

    if spec.backend not in BACKENDS:
        raise ValueError(f"unknown image backend {spec.backend!r}; known: {', '.join(sorted(BACKENDS))}")
    seed = random.randint(0, 2**31 - 1) if seed is None or seed < 0 else seed
    raw = BACKENDS[spec.backend](frame(prompt, spec), spec, seed, count)
    if not raw:
        raise RuntimeError(f"{spec.backend} returned no images")
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in raw]


def write(prompt: str, out_dir: Path, spec: ImageSpec, seed: int | None = None, count: int = 1,
          log=print) -> Path:
    """Generate into ``out_dir``: image.png (the first), candidate_<k>.png when count > 1, image.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = random.randint(0, 2**31 - 1) if seed is None or seed < 0 else seed
    log(f"[image] {spec.backend} at {spec.base_url or spec.command}: {prompt!r} (seed {seed})")
    t0 = time.time()
    images = generate(prompt, spec, seed, count)
    for k, img in enumerate(images if count > 1 else []):
        img.save(out_dir / f"candidate_{k}.png")
    images[0].save(out_dir / "image.png")
    (out_dir / "image.json").write_text(json.dumps({
        "prompt": prompt, "framed_prompt": frame(prompt, spec), "negative_prompt": spec.negative,
        "seed": seed, "backend": spec.backend, "model": spec.model, "size": [spec.width, spec.height],
        "seconds": round(time.time() - t0, 1)}, indent=2))
    log(f"[image] wrote {out_dir / 'image.png'} in {time.time() - t0:.0f}s")
    return out_dir / "image.png"


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", required=True, help="the character, e.g. 'a goblin with a lantern'")
    ap.add_argument("--out", required=True)
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
    ap.add_argument("--raw", action="store_true", help="send the prompt as written, without the character framing")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--count", type=int, default=1)
    a = ap.parse_args(argv)
    spec = ImageSpec.from_env(backend=a.backend, url=a.url, model=a.model, width=a.width, height=a.height,
                              steps=a.steps, cfg_scale=a.cfg_scale, sampler=a.sampler, workflow=a.workflow,
                              command=a.command, style=False if a.raw else None)
    write(a.prompt, Path(a.out), spec, a.seed, a.count)


if __name__ == "__main__":
    main()
