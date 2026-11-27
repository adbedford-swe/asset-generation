"""Videos of the character moving, from its start frame.

Backends:

- ``sdcpp``: stable-diffusion.cpp's sd-server with a MiniMax-H3 first/last-frame model
  (recipe ``minimax-h3-fl2va``). The start frame is the first frame, and the last too, so the
  video comes back to the rest pose: a one-shot then returns to its stance, and a loop has a
  cycle to be cut from its middle. H3 writes a soundtrack with the picture.
- ``command``: any program, from a template with ``{prompt}``, ``{init}``, ``{end}``, ``{out}``,
  ``{frames}``, ``{fps}``, ``{width}``, ``{height}`` and ``{seed}``; for example sd-cli run with
  the options your machine needs.
- ``file``: a video made elsewhere from the start frame (``assetgen capture start`` writes it).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

BACKENDS = ("sdcpp", "command", "file")
DEFAULT_RECIPE = "minimax-h3-fl2va"
FPS = 24


def h3_frames(seconds: float, fps: int = FPS) -> int:
    """The nearest frame count MiniMax-H3 accepts (17n + 5) at or above ``seconds``."""
    n = max(0, int(np.ceil((seconds * fps - 5) / 17)))
    return 17 * n + 5


@dataclass
class VideoSpec:
    backend: str = "sdcpp"
    url: str | None = None
    model: str | None = None         # sdcpp: a recipe id, or recipe:file as the image backend takes it
    command: str | None = None
    width: int = 640
    height: int = 640
    steps: int | None = None
    cfg_scale: float | None = None
    keep_loaded: bool = False

    @property
    def base_url(self) -> str:
        return (self.url or "http://127.0.0.1:1234").rstrip("/")

    @classmethod
    def from_env(cls, **overrides) -> "VideoSpec":
        spec = cls(backend=os.environ.get("ASSETGEN_VIDEO_BACKEND", "sdcpp"),
                   url=os.environ.get("ASSETGEN_VIDEO_URL") or os.environ.get("ASSETGEN_IMAGE_URL"),
                   model=os.environ.get("ASSETGEN_VIDEO_MODEL"),
                   command=os.environ.get("ASSETGEN_VIDEO_COMMAND"),
                   keep_loaded=os.environ.get("ASSETGEN_VIDEO_KEEP_LOADED") == "1")
        return replace(spec, **{k: v for k, v in overrides.items() if v is not None})


def video_prompt(motion: str, subject: str | None = None) -> str:
    """A motion prompt ("An object walks forward.") as a video prompt: the character, whole and in
    frame, against the start frame's plain background, seen by a camera that does not move."""
    text = motion.strip()
    who = (subject or "the character").strip().rstrip(".")
    if text.lower().startswith("an object"):
        text = who[0].upper() + who[1:] + text[len("an object"):]
    return (f"{text} The camera is static. The whole body stays in frame against a plain white background, "
            "with even, soft lighting.")


def _image_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _is_audio(file: dict) -> bool:
    return "audio" in file["id"].lower()


def _selection(spec: VideoSpec) -> dict:
    """The model selection for a video recipe: files from sd-server's catalog for each slot.

    MiniMax-H3's video and audio VAEs sit in the same folder, and its audio VAE is optional to the
    recipe: the video VAE slot never takes a file named for audio, and the audio VAE slot is filled
    when there is one, since the clip's sound marks its timing. sd.cpp cannot tell the first/last-frame
    weights from the reference weights, so for a first/last-frame recipe, reference weights are passed over."""
    from .. import sdcpp

    wanted = spec.model or DEFAULT_RECIPE
    recipe_id, _, file = wanted.partition(":")
    recipes, files = sdcpp.catalog(spec)
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if recipe is None:
        raise RuntimeError(f"sd-server at {spec.base_url} has no recipe {recipe_id!r}; its video recipes: "
                           + ", ".join(r["id"] for r in recipes if "video_frames" in (r.get("defaults") or {})))
    pool = [f for f in files if not _is_audio(f)]
    primaries = [f for f in pool if f["category"] == "diffusion_models" and (not file or f["id"].endswith(file))]
    if "fl2va" in recipe_id:
        primaries = [f for f in primaries if "ref2va" not in f["id"].lower()]
    found = [(sdcpp._name_score(recipe, p), p["id"], m) for p in primaries
             if (m := sdcpp._fill(recipe, p, pool)) is not None]
    if not found:
        raise RuntimeError(f"sd-server's model folders have no files that fill recipe {recipe_id!r}")
    found.sort(key=lambda x: (-x[0], x[1]))
    models = dict(found[0][2])
    slots = {s["id"] for s in recipe.get("slots", [])}
    if "audio_vae" in slots and "audio_vae" not in models:
        audio = sorted(f["id"] for f in files if _is_audio(f) and f["category"] in sdcpp.SLOT_CATEGORIES["audio_vae"])
        if audio:
            models["audio_vae"] = audio[0]
    return {"recipe": recipe_id, "models": models}


def sdcpp_generate(prompt: str, init: Path, end: Path | None, frames: int, spec: VideoSpec, seed: int,
                   out: Path, log=print, poll: float = 2.0, timeout: float = 7200) -> Path:
    """A video from sd-server's native video API, written to ``out`` (its container, with sound)."""
    from .. import sdcpp
    from ..imagegen import _check

    body = {"prompt": prompt, "width": spec.width, "height": spec.height, "seed": seed,
            "video_frames": frames, "fps": FPS, "init_image": _image_b64(init),
            "end_image": _image_b64(end) if end else None, "output_format": "webm"}
    sample = {}
    if spec.steps:
        sample["sample_steps"] = spec.steps
    if spec.cfg_scale is not None:
        sample["guidance"] = {"txt_cfg": spec.cfg_scale}
    if sample:
        body["sample_params"] = sample
    wanted = (spec.model or DEFAULT_RECIPE).partition(":")[0]
    status = sdcpp.status(spec) or {}
    if not (status.get("state") == "ready" and status.get("recipe") == wanted and ":" not in (spec.model or "")):
        body["model"] = _selection(spec)
    log(f"[capture] sd-server: {frames} frames at {spec.width}x{spec.height}"
        + (f" with {body['model']['recipe']}" if "model" in body else ""))
    job = _check(sdcpp._request(spec, "POST", "/sdcpp/v1/vid_gen", json=body, timeout=120)).json()
    started, last = time.time(), None
    try:
        while True:
            state = _check(sdcpp._request(spec, "GET", f"/sdcpp/v1/jobs/{job['id']}", timeout=60)).json()
            status_ = state.get("status")
            if status_ == "completed":
                break
            if status_ in ("failed", "cancelled"):
                raise RuntimeError(f"sd-server video job {status_}: {json.dumps(state.get('error'))[:500]}")
            progress = state.get("progress") or {}
            now = (progress.get("phase"), progress.get("step"))
            if now != last and progress.get("phase"):
                log(f"[capture]   {progress.get('phase')} {progress.get('step', '')}/{progress.get('total_steps', '')}")
                last = now
            if time.time() - started > timeout:
                sdcpp._request(spec, "POST", f"/sdcpp/v1/jobs/{job['id']}/cancel", timeout=30)
                raise TimeoutError(f"sd-server video job took over {timeout:.0f}s")
            time.sleep(poll)
    finally:
        if not spec.keep_loaded and "model" in body:
            try:
                sdcpp.unload(spec, log)
            except Exception as e:  # freeing VRAM is best effort; the video may still be fine
                log(f"[capture] could not unload sd-server's model: {e}")
    result = state["result"]
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = {"webm": ".webm", "webp": ".webp", "avi": ".avi"}.get(result.get("output_format", "webm"), ".webm")
    out = out.with_suffix(suffix)
    out.write_bytes(base64.b64decode(result["b64_json"]))
    return out


def command_generate(prompt: str, init: Path, end: Path | None, frames: int, spec: VideoSpec, seed: int,
                     out: Path, log=print) -> Path:
    from ..imagegen import split_command

    if not spec.command:
        raise ValueError('the command backend needs a template, e.g. "sd-cli -M vid_gen ... -i {init} --end-img {end} '
                         '-p {prompt} --video-frames {frames} -o {out}"')
    out.parent.mkdir(parents=True, exist_ok=True)
    values = {"prompt": prompt, "init": str(init), "end": str(end or init), "out": str(out), "frames": frames,
              "fps": FPS, "width": spec.width, "height": spec.height, "seed": seed}
    args = [part.format(**values) for part in split_command(spec.command)]
    log(f"[capture] running {args[0]} for {frames} frames")
    subprocess.run(args, check=True)
    if not out.exists():
        raise RuntimeError(f"the video command wrote nothing to {out}")
    return out


def generate(prompt: str, init: Path, end: Path | None, seconds: float, spec: VideoSpec, seed: int, out: Path,
             log=print) -> Path:
    frames = h3_frames(seconds)
    if spec.backend == "sdcpp":
        return sdcpp_generate(prompt, init, end, frames, spec, seed, out, log)
    if spec.backend == "command":
        return command_generate(prompt, init, end, frames, spec, seed, out, log)
    raise ValueError(f"backend {spec.backend!r} does not generate videos; one of sdcpp, command")


def read_frames(path: Path, width: int, height: int) -> tuple[np.ndarray, float]:
    """Every frame (T, height, width, 3) uint8, resized to the start frame's size, and the frame rate."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=["-vf", f"scale={width}:{height}"])
    meta = next(reader)
    frames = [np.frombuffer(f, dtype=np.uint8).reshape(height, width, 3) for f in reader]
    if not frames:
        raise RuntimeError(f"{path} has no frames")
    return np.stack(frames), float(meta.get("fps") or FPS)


def read_audio(path: Path, rate: int = 32000) -> np.ndarray | None:
    """The video's soundtrack, mono at ``rate``, or None when it has none."""
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-vn",
                              "-ac", "1", "-ar", str(rate), "-acodec", "pcm_s16le", str(wav)], capture_output=True)
        if run.returncode != 0 or not wav.exists() or wav.stat().st_size <= 44:
            return None
        with wave.open(str(wav)) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def write_audio(path: Path, samples: np.ndarray, rate: int = 32000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return path
