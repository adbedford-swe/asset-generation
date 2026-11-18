"""The assetgen server: an HTTP API and the web UI, sharing one job queue.

    python server.py                          # http://127.0.0.1:7860 (UI), /api/v1 (API), /docs (API reference)
    assetgen serve --host 0.0.0.0 --devices cuda:0,cuda:1 --image-backend sdcpp --image-url http://127.0.0.1:1234

Every job is a run folder under ``outputs/webui`` (``$ASSETGEN_OUT``), laid out like
``assetgen run``, so jobs survive restarts, the CLI can resume them, and the web UI,
the API, the Unity package and any script see the same list. A job runs on the next
free device; ``--devices`` lists them (one job per device at a time).

Set ``ASSETGEN_API_TOKEN`` (or ``--token``) to require ``Authorization: Bearer <token>``
on the API, which you want before listening on anything but localhost.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import character as character_file
from .animate import DEFAULT_PROMPTS
from .imagegen import BACKENDS, ImageSpec
from .pipeline import STAGES, Cancelled, Options, branch, completed_stages, run_inputs
from .pipeline import run as run_pipeline
from .thirdparty import ROOT

OUT_ROOT = Path(os.environ.get("ASSETGEN_OUT") or os.environ.get("ASSETGEN_WEBUI_OUT")
                or ROOT / "outputs" / "webui").resolve()
DRAFTS = "_drafts"
UPLOADS = "_uploads"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MAX_LOG_LINES = 5000


def version() -> str:
    try:
        from importlib.metadata import version as v

        return v("assetgen")
    except Exception:
        return "dev"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")[:48] or "asset"


def detect_devices() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return ["cpu"]


def best_glb(folder: Path, name: str) -> Path | None:
    for p in (folder / f"{name}.glb", folder / "2_rig" / "rigged.glb", folder / "1_mesh" / "mesh.glb"):
        if p.exists():
            return p
    return None


@dataclass
class Settings:
    """Everything about a run other than its input. The defaults are the CLI's."""
    motions: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    height: float | None = 1.8
    device: str = "auto"
    mode: str = "synthetic"
    seed: int = 42
    loop: str = "auto"
    root_motion: bool = False
    repetitions: int = 1
    faces: int = 40000
    texture: int = 2048
    voxels: int = 256
    num_beams: int = 10
    rig_candidates: int = 3
    rig_class: str | None = None
    cfg_scale: float | None = None
    llm_base_url: str | None = None

    @classmethod
    def parse(cls, data: dict | None) -> "Settings":
        data = dict(data or {})
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and v is not None}
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(unknown)}")
        if "loop" in known:
            known["loop"] = str(known["loop"])
        return cls(**known)

    def options(self, out: Path, device: str, **inputs) -> Options:
        return Options(out=out, device=device, mode=self.mode, seed=int(self.seed), prompts=list(self.motions),
                       repetitions=int(self.repetitions), height=self.height, rig_class=self.rig_class or None,
                       num_beams=int(self.num_beams), rig_candidates=int(self.rig_candidates),
                       mesh_args=["--target_faces", str(int(self.faces)), "--texture_size", str(int(self.texture)),
                                  "--resolution", str(int(self.voxels))],
                       llm_base_url=self.llm_base_url or None, cfg_scale=self.cfg_scale, loop=self.loop,
                       root_motion=bool(self.root_motion), **inputs)


class Job:
    """One run folder, queued, running or finished."""

    def __init__(self, folder: Path, name: str, options: Options | None = None, start: str | None = None,
                 settings: Settings | None = None, source: str | None = None, status: str = "queued"):
        self.folder = Path(folder)
        self.id = self.folder.name
        self.name = name
        self.options = options
        self.start = start
        self.settings = settings
        self.source = source
        self.status = status
        self.stage: str | None = None
        self.stages: dict[str, str] = {}
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.lines: list[str] = []
        self.seq = 0  # bumps on every change, so watchers can tell what is new
        self.cancel_event = threading.Event()
        self.lock = threading.Condition()

    @classmethod
    def from_folder(cls, folder: Path) -> "Job":
        """A job for a run folder made earlier (by this server, the old web UI or the CLI)."""
        try:
            inputs = run_inputs(folder)
        except Exception:
            inputs = {}
        name = inputs.get("name") or folder.name
        job = cls(folder, name, status="done")
        done = completed_stages(folder)
        job.stages = {s: "done" for s in done}
        finished = (folder / f"{name}.glb").exists() and "finalize" in done
        job.status = "done" if finished else "incomplete"
        job.created = job.finished = (folder / "manifest.json").stat().st_mtime if (folder / "manifest.json").exists() \
            else folder.stat().st_mtime
        return job

    def log(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > MAX_LOG_LINES:
                del self.lines[:len(self.lines) - MAX_LOG_LINES]
            self.seq += 1
            self.lock.notify_all()

    def update(self, **changes) -> None:
        with self.lock:
            for k, v in changes.items():
                setattr(self, k, v)
            self.seq += 1
            self.lock.notify_all()

    def on_stage(self, stage: str, state: str) -> None:
        with self.lock:
            self.stages[stage] = state
            if state == "running":
                self.stage = stage
            self.seq += 1
            self.lock.notify_all()

    def wait(self, seq: int, timeout: float) -> int:
        with self.lock:
            if self.seq == seq:
                self.lock.wait(timeout)
            return self.seq

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def planned_stages(self) -> list[str]:
        if self.options is None:
            return [s for s in STAGES if s in self.stages]
        inputs = self.options
        planned = [s for s in STAGES if not (s == "image" and not (inputs.text and not inputs.image))
                   and not (s == "generate" and inputs.ply)]
        if self.start:
            planned = planned[planned.index(self.start):] if self.start in planned else planned
        return planned

    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        planned = self.planned_stages()
        if not planned:
            return 0.0
        finished = sum(1 for s in planned if self.stages.get(s) in ("done", "skipped"))
        running = 0.5 if self.stage in planned and self.stages.get(self.stage) == "running" else 0.0
        return round((finished + running) / len(planned), 3)

    def character(self) -> dict | None:
        """The character file, written now for a run finalized before character files existed."""
        path = self.folder / f"{self.name}{character_file.EXTENSION}"
        glb = self.folder / f"{self.name}.glb"
        try:
            if not path.exists() and glb.exists() and self.status == "done":
                character_file.write(glb)
            return json.loads(path.read_text()) if path.exists() else None
        except Exception:
            return None

    def input_summary(self) -> dict:
        if self.options is not None:
            o = self.options
            inputs = {"text": o.text, "image": o.image, "ply": o.ply, "reference": o.reference}
        else:
            try:
                inputs = run_inputs(self.folder)
            except Exception:
                return {}
        out = {"prompt": inputs.get("text")}
        image_info = self.folder / "0_image" / "image.json"
        if image_info.exists() and not out["prompt"]:
            out["prompt"] = json.loads(image_info.read_text()).get("prompt")
        for key in ("image", "ply", "reference"):
            if inputs.get(key):
                out[key] = Path(inputs[key]).name
        return {k: v for k, v in out.items() if v}

    def to_dict(self, base: str = "/api/v1", detail: bool = True) -> dict:
        url = f"{base}/jobs/{self.id}"
        glb = best_glb(self.folder, self.name)
        files = {}
        if glb is not None:
            files["model"] = f"{url}/files/{glb.relative_to(self.folder).as_posix()}"
        for key, rel in (("preview", f"{self.name}.png"), ("image", "0_image/image.png"),
                         ("front", "1_mesh/front.png"), ("compare", "1_mesh/compare.png"),
                         ("skeleton", "2_rig/preview.png")):
            if (self.folder / rel).exists():
                files[key] = f"{url}/files/{rel}"
        data = {
            "id": self.id, "name": self.name, "status": self.status, "stage": self.stage,
            "progress": self.progress(), "created": self.created, "started": self.started,
            "finished": self.finished, "error": self.error, "input": self.input_summary(),
            "urls": {"self": url, "events": f"{url}/events", "log": f"{url}/log", "cancel": f"{url}/cancel",
                     "continue": f"{url}/continue", "character": f"{url}/character", **files,
                     "export": {e: f"{url}/export/{e}" for e in ("unity", "godot", "gltf")}},
        }
        if detail:
            data["stages"] = {s: self.stages.get(s, "pending") for s in self.planned_stages()}
            data["folder"] = str(self.folder)
            data["source"] = self.source
            data["character"] = self.character()
        return data


class JobManager:
    """Runs jobs on a pool of devices, one job per device at a time, and remembers them."""

    def __init__(self, root: Path = OUT_ROOT, devices: list[str] | None = None,
                 image_spec: ImageSpec | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.devices = devices or detect_devices()
        self.image_spec = image_spec or ImageSpec.from_env()
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job] = queue.Queue()
        self.free = queue.Queue()
        for d in self.devices:
            self.free.put(d)
        self.lock = threading.Lock()
        self.workers = [threading.Thread(target=self._work, daemon=True, name=f"assetgen-{d}") for d in self.devices]
        for w in self.workers:
            w.start()

    # -- bookkeeping -------------------------------------------------------

    def get(self, job_id: str) -> Job:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id or "") or job_id.startswith("_"):
            raise KeyError(job_id)
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            folder = self.root / job_id
            if not (folder / "manifest.json").exists() and not (folder / "run.json").exists():
                raise KeyError(job_id)
            job = Job.from_folder(folder)
            with self.lock:
                job = self.jobs.setdefault(job_id, job)
        return job

    def list(self, limit: int | None = None) -> list[Job]:
        folders = sorted((p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("_")
                          and ((p / "manifest.json").exists() or (p / "run.json").exists())), reverse=True)
        jobs = []
        for folder in folders[:limit] if limit else folders:
            try:
                jobs.append(self.get(folder.name))
            except KeyError:
                continue
        return jobs

    def new_folder(self, name: str, suffix: str = "") -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{slug(name)}{suffix}"
        folder, k = self.root / base, 1
        while folder.exists():
            k += 1
            folder = self.root / f"{base}-{k}"
        folder.mkdir(parents=True)
        return folder

    # -- inputs ------------------------------------------------------------

    def save_upload(self, filename: str, data: bytes) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in (*IMAGE_EXTENSIONS, ".ply"):
            raise ValueError(f"unsupported file type {ext or '(none)'}; images or .ply only")
        upload_id = f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
        folder = self.root / UPLOADS / upload_id
        folder.mkdir(parents=True)
        (folder / f"{slug(Path(filename).stem)}{ext}").write_bytes(data)
        return upload_id

    def upload_path(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9]+-[0-9a-f]+", upload_id or ""):
            raise KeyError(upload_id)
        files = list((self.root / UPLOADS / upload_id).glob("*"))
        if not files:
            raise KeyError(upload_id)
        return files[0]

    def draft_images(self, prompt: str, count: int = 4, seed: int | None = None,
                     spec: ImageSpec | None = None) -> dict:
        """Candidate images for a prompt, to pick one before the 3D stages."""
        from .imagegen import write

        draft_id = f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
        folder = self.root / DRAFTS / draft_id
        write(prompt, folder, spec or self.image_spec, seed, max(1, count), log=lambda *_: None)
        if count <= 1:
            shutil.copy2(folder / "image.png", folder / "candidate_0.png")
        n = len(list(folder.glob("candidate_*.png")))
        return {"id": draft_id, "prompt": prompt, "count": n,
                "images": [f"/api/v1/images/{draft_id}/{k}" for k in range(n)],
                "meta": json.loads((folder / "image.json").read_text())}

    def draft_path(self, draft_id: str, index: int = 0) -> Path:
        if not re.fullmatch(r"[0-9]+-[0-9a-f]+", draft_id or ""):
            raise KeyError(draft_id)
        path = self.root / DRAFTS / draft_id / f"candidate_{int(index)}.png"
        if not path.exists():
            raise KeyError(f"{draft_id}/{index}")
        return path

    # -- submitting --------------------------------------------------------

    def submit(self, *, prompt: str | None = None, image: Path | None = None, ply: Path | None = None,
               reference: Path | None = None, draft: tuple[str, int] | None = None, name: str | None = None,
               settings: Settings | None = None, image_spec: ImageSpec | None = None) -> Job:
        """Queue a new character. Give one of: ``prompt`` (text to image to character), ``image``,
        ``ply`` (with an optional ``reference`` image), or ``draft`` (a candidate from draft_images,
        which keeps its prompt)."""
        settings = settings or Settings()
        given = [x for x in (image, ply, draft) if x]
        if len(given) > 1:
            raise ValueError("give one of image, ply or draft")
        if not given and not prompt:
            raise ValueError("give a prompt, an image, a splat (.ply) or a draft image")
        if draft and not prompt:
            meta = self.draft_path(*draft).parent / "image.json"
            prompt = json.loads(meta.read_text()).get("prompt") if meta.exists() else None
        name = slug(name or (Path(image or ply).stem if (image or ply) else prompt or "asset"))
        folder = self.new_folder(name)
        inputs: dict = {"name": name}
        if draft:
            src = self.draft_path(*draft)
            (folder / "0_image").mkdir()
            shutil.copy2(src, folder / "0_image" / "image.png")
            meta = src.parent / "image.json"
            if meta.exists():
                shutil.copy2(meta, folder / "0_image" / "image.json")
                prompt = prompt or json.loads(meta.read_text()).get("prompt")
            inputs.update(text=prompt, image=folder / "0_image" / "image.png")
        elif image or ply:
            (folder / "input").mkdir()
            for key, path in (("image", image), ("ply", ply), ("reference", reference)):
                if path:
                    dst = folder / "input" / Path(path).name
                    shutil.copy2(path, dst)
                    inputs[key] = dst
            if prompt:
                inputs["text"] = prompt
        else:
            inputs.update(text=prompt, image_spec=image_spec or self.image_spec)
        o = settings.options(folder, settings.device, **inputs)
        job = Job(folder, name, o, None, settings, source=None)
        return self._enqueue(job)

    def continue_job(self, source_id: str, start: str, settings: Settings | None = None) -> Job:
        """A new job that keeps ``source_id``'s results before ``start`` and runs the rest."""
        if start not in STAGES or start in ("image", "generate"):
            raise ValueError(f"continue from one of mesh, rig, animate, finalize (not {start!r})")
        source = self.get(source_id)
        if source.active:
            raise ValueError(f"{source_id} is still {source.status}")
        settings = settings or source.settings or Settings()
        folder = self.root / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(source.name)}-from-{start}"
        branch(source.folder, start, folder)
        inputs = run_inputs(source.folder)

        def moved(p):
            if not p:
                return None
            p = Path(p)
            try:
                return folder / p.relative_to(source.folder)
            except ValueError:
                return p

        o = settings.options(folder, settings.device, name=inputs.get("name") or source.name,
                             text=inputs.get("text"), image=moved(inputs.get("image")), ply=moved(inputs.get("ply")),
                             reference=moved(inputs.get("reference")))
        o.mode = inputs.get("mode") or o.mode
        job = Job(folder, o.name, o, start, settings, source=source.id)
        return self._enqueue(job)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "queued":
            job.update(status="cancelled", finished=time.time(), error="cancelled before it started")
        elif job.status == "running":
            job.cancel_event.set()
        return job

    def _enqueue(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
        job.log(f"queued {job.id}")
        self.queue.put(job)
        return job

    # -- running -----------------------------------------------------------

    def _work(self) -> None:
        while True:
            job = self.queue.get()
            if job.status != "queued":
                continue
            device = self.free.get()
            try:
                self._run(job, device)
            finally:
                self.free.put(device)

    def _run(self, job: Job, device: str) -> None:
        o = job.options
        if o.device in ("auto", "", None):
            o = replace(o, device=device)
            job.options = o
        job.update(status="running", started=time.time())
        job.log(f"running on {o.device}" + (f", from {job.start}" if job.start else ""))
        try:
            run_pipeline(o, start=job.start, force=job.start is not None, log=job.log, cancel=job.cancel_event,
                         on_stage=job.on_stage)
            job.update(status="done", stage=None)
        except Cancelled as e:
            job.update(status="cancelled", error=str(e))
        except SystemExit as e:
            job.update(status="failed", error=str(e))
        except Exception as e:  # reported on the job, not swallowed
            job.update(status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            if job.stage and job.stages.get(job.stage) == "running":
                job.on_stage(job.stage, "cancelled" if job.status == "cancelled" else "failed")
            job.update(finished=time.time())
            job.log(f"{job.status}" + (f": {job.error}" if job.error else ""))


# -- HTTP API --------------------------------------------------------------


class ImageBackend(BaseModel):
    backend: str | None = Field(None, description=f"one of {', '.join(BACKENDS)}")
    url: str | None = None
    model: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str | None = None
    style: bool | None = Field(None, description="frame the prompt as a full-body character (default true)")


class Draft(BaseModel):
    id: str
    index: int = 0


class JobRequest(BaseModel):
    prompt: str | None = Field(None, description="Describe the character; an image is generated first "
                                                 "unless image, upload or draft is given.",
                               examples=["a goblin shaman with a lantern and a hooded cloak"])
    image: str | None = Field(None, description="An image of the character, as a data URL or base64.")
    upload: str | None = Field(None, description="An id from POST /uploads (an image or a .ply splat).")
    reference: str | None = Field(None, description="With a .ply upload: an upload id of Idol's input image.")
    draft: Draft | None = Field(None, description="A candidate from POST /images.")
    name: str | None = None
    motions: list[str] | None = Field(None, description='One per clip: "An object <does one thing>." '
                                                         'Clips loop unless the prompt ends in "[once]".')
    height: float | None = Field(None, description="Final height in metres (default 1.8).")
    settings: dict | None = Field(None, description="Any of: " + ", ".join(Settings.__dataclass_fields__))
    image_backend: ImageBackend | None = None


class ContinueRequest(BaseModel):
    start: str = Field(..., alias="from", description="mesh, rig, animate or finalize")
    motions: list[str] | None = None
    height: float | None = None
    settings: dict | None = None


class ImagesRequest(BaseModel):
    prompt: str
    count: int = Field(4, ge=1, le=8)
    seed: int | None = None
    image_backend: ImageBackend | None = None


def decode_image(data: str) -> tuple[str, bytes]:
    """A data URL or plain base64 image -> (extension, bytes)."""
    ext = ".png"
    if data.startswith("data:"):
        header, data = data.split(",", 1)
        m = re.match(r"data:image/(\w+)", header)
        if m:
            ext = "." + m.group(1).replace("jpeg", "jpg")
    return ext, base64.b64decode(data)


def create_api(manager: JobManager, token: str | None = None):
    def check_token(authorization: str | None = Header(default=None)):
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token", headers={"WWW-Authenticate": "Bearer"})

    router = APIRouter(prefix="/api/v1", dependencies=[Depends(check_token)])

    def settings_of(motions, height, settings) -> Settings:
        try:
            s = Settings.parse(settings)
        except (TypeError, ValueError) as e:
            raise HTTPException(422, str(e))
        if motions:
            s.motions = [m.strip() for m in motions if m.strip()]
        if height is not None:
            s.height = height
        return s

    def spec_of(backend: ImageBackend | None) -> ImageSpec:
        if backend is None:
            return manager.image_spec
        values = {k: v for k, v in backend.model_dump().items() if v is not None}
        spec = replace(manager.image_spec, **values)
        if spec.backend not in BACKENDS:
            raise HTTPException(422, f"unknown image backend {spec.backend!r}")
        return spec

    def job_or_404(job_id: str) -> Job:
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}")

    @router.get("/info", summary="Server version, devices, image backend and defaults")
    def info():
        from .imagegen import probe

        return {"name": "assetgen", "version": version(), "devices": manager.devices,
                "stages": STAGES, "engines": ["unity", "godot", "gltf"], "default_motions": DEFAULT_PROMPTS,
                "image_backend": {**manager.image_spec.public(), "url": manager.image_spec.base_url,
                                  **probe(manager.image_spec)},
                "image_backends": sorted(BACKENDS), "settings": Settings().__dict__,
                "busy": sum(1 for j in manager.jobs.values() if j.status == "running"),
                "queued": sum(1 for j in manager.jobs.values() if j.status == "queued")}

    @router.post("/uploads", summary="Upload an image or a .ply splat; returns an id for POST /jobs")
    async def upload(file: UploadFile = File(...)):
        try:
            return {"id": manager.save_upload(file.filename or "upload.png", await file.read())}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @router.post("/images", summary="Generate candidate images for a prompt, to pick one for POST /jobs")
    def images(req: ImagesRequest):
        try:
            return manager.draft_images(req.prompt, req.count, req.seed, spec_of(req.image_backend))
        except Exception as e:
            raise HTTPException(502, f"image backend failed: {e}")

    @router.get("/images/{draft_id}/{index}", summary="One candidate image")
    def draft_image(draft_id: str, index: int):
        try:
            return FileResponse(manager.draft_path(draft_id, index), media_type="image/png")
        except KeyError:
            raise HTTPException(404, "no such image")

    @router.post("/jobs", status_code=202, summary="Make a character: prompt, image, upload or draft -> job")
    def create_job(req: JobRequest):
        s = settings_of(req.motions, req.height, req.settings)
        image = ply = reference = None
        try:
            if req.image:
                ext, data = decode_image(req.image)
                image = manager.upload_path(manager.save_upload(f"{req.name or 'image'}{ext}", data))
            if req.upload:
                path = manager.upload_path(req.upload)
                if path.suffix == ".ply":
                    ply = path
                    reference = manager.upload_path(req.reference) if req.reference else None
                else:
                    image = path
            job = manager.submit(prompt=req.prompt, image=image, ply=ply, reference=reference,
                                 draft=(req.draft.id, req.draft.index) if req.draft else None, name=req.name,
                                 settings=s, image_spec=spec_of(req.image_backend))
        except KeyError as e:
            raise HTTPException(404, f"not found: {e}")
        except ValueError as e:
            raise HTTPException(422, str(e))
        return job.to_dict()

    @router.get("/jobs", summary="Every job, newest first")
    def list_jobs(limit: int = Query(50, ge=1, le=1000)):
        return {"jobs": [j.to_dict(detail=False) for j in manager.list(limit)]}

    @router.get("/jobs/{job_id}", summary="A job's status, stages, files and (once done) its character")
    def get_job(job_id: str):
        return job_or_404(job_id).to_dict()

    @router.get("/jobs/{job_id}/log", response_class=PlainTextResponse, summary="The job's log")
    def job_log(job_id: str, tail: int = Query(400, ge=1, le=MAX_LOG_LINES)):
        job = job_or_404(job_id)
        if job.lines:
            return "\n".join(job.lines[-tail:])
        logs = sorted((job.folder / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
        text = "\n".join(f"== {p.stem}\n{p.read_text(errors='replace')}" for p in logs)
        return "\n".join(text.splitlines()[-tail:])

    @router.get("/jobs/{job_id}/events", summary="Server-sent events: log lines and status until the job ends")
    def job_events(job_id: str):
        job = job_or_404(job_id)

        def stream():
            sent, seq, last_state = 0, -1, None
            while True:
                seq = job.wait(seq, 15.0)
                lines = job.lines[sent:]
                sent = len(job.lines)
                for line in lines:
                    yield f"event: log\ndata: {json.dumps(line)}\n\n"
                state = (job.status, job.stage, job.progress())
                if state != last_state:
                    last_state = state
                    yield f"event: status\ndata: {json.dumps(job.to_dict(detail=False))}\n\n"
                if not job.active:
                    yield f"event: end\ndata: {json.dumps(job.to_dict())}\n\n"
                    return
                if not lines:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.post("/jobs/{job_id}/cancel", summary="Stop a queued or running job")
    def cancel_job(job_id: str):
        job_or_404(job_id)
        return manager.cancel(job_id).to_dict(detail=False)

    @router.post("/jobs/{job_id}/continue", status_code=202,
                 summary="Re-run from a stage (new motions, a new rig, ...) in a new job; the original is kept")
    def continue_job(job_id: str, req: ContinueRequest):
        job = job_or_404(job_id)
        s = settings_of(req.motions, req.height, req.settings) if (req.motions or req.height or req.settings) \
            else (job.settings or Settings())
        try:
            return manager.continue_job(job_id, req.start, s).to_dict()
        except (ValueError, SystemExit) as e:
            raise HTTPException(422, str(e))

    @router.get("/jobs/{job_id}/character", summary="The character file (.assetgen): skeleton, clips, locomotion")
    def job_character(job_id: str):
        data = job_or_404(job_id).character()
        if data is None:
            raise HTTPException(404, "not finished yet")
        return data

    @router.get("/jobs/{job_id}/files/{path:path}", summary="Any file in the job's folder")
    def job_file(job_id: str, path: str):
        job = job_or_404(job_id)
        target = (job.folder / path).resolve()
        if not target.is_relative_to(job.folder.resolve()) or not target.is_file():
            raise HTTPException(404, "no such file")
        media = {".glb": "model/gltf-binary", ".assetgen": "application/json"}.get(target.suffix)
        return FileResponse(target, media_type=media)

    @router.get("/jobs/{job_id}/export/{engine}", summary="A zip ready for unity, godot or gltf")
    def job_export(job_id: str, engine: str):
        from .export import ENGINES, export_zip

        job = job_or_404(job_id)
        if engine not in ENGINES:
            raise HTTPException(404, f"engine is one of {', '.join(ENGINES)}")
        if not (job.folder / f"{job.name}.glb").exists():
            raise HTTPException(409, "the character is not finished")
        exports = job.folder / "exports"
        exports.mkdir(exist_ok=True)
        path = export_zip(job.folder / f"{job.name}.glb", engine, exports / f"{job.name}-{engine}.zip")
        return FileResponse(path, media_type="application/zip", filename=path.name)

    return router


def create_app(manager: JobManager | None = None, token: str | None = None, ui: bool = True):
    """The FastAPI app: the API under /api/v1 and, with ``ui``, the web UI at /."""
    from fastapi import FastAPI

    manager = manager or JobManager()
    app = FastAPI(title="assetgen", version=version(),
                  description="Text or image to a textured, rigged, animated character, ready for Unity "
                              "and Godot. Make a job with POST /api/v1/jobs, follow it with "
                              "GET /api/v1/jobs/{id}/events, then fetch /export/unity or /export/godot.")
    app.state.manager = manager
    app.include_router(create_api(manager, token))
    if ui:
        import gradio as gr

        from .webui import CSS, MODEL_VIEWER, build

        demo = build(manager)
        demo.queue(default_concurrency_limit=16)
        app = gr.mount_gradio_app(app, demo, path="/", allowed_paths=[str(manager.root)], head=MODEL_VIEWER,
                                  css=CSS, footer_links=[{"label": "API", "url": "/docs"}])
    return app


def main(argv=None) -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="assetgen serve", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--devices", default=None, help="comma-separated, e.g. cuda:0,cuda:1 (default: every GPU)")
    ap.add_argument("--out", default=None, help=f"where jobs are kept (default {OUT_ROOT})")
    ap.add_argument("--token", default=os.environ.get("ASSETGEN_API_TOKEN"), help="require this bearer token on /api")
    ap.add_argument("--no-ui", action="store_true", help="the API only")
    ap.add_argument("--image-backend", default=None, help=f"text-to-image backend: {', '.join(BACKENDS)}")
    ap.add_argument("--image-url", default=None, help="its address, e.g. http://127.0.0.1:1234 for sd-server")
    ap.add_argument("--image-model", default=None)
    ap.add_argument("--image-size", default=None, help="WIDTHxHEIGHT (default 1024x1024)")
    ap.add_argument("--image-steps", type=int, default=None)
    ap.add_argument("--comfy-workflow", default=None, help="comfyui: an API-format workflow with {{prompt}}")
    ap.add_argument("--image-command", default=None, help='command: e.g. "sd-cli -p {prompt} -o {out}"')
    a = ap.parse_args(argv)
    width = height = None
    if a.image_size:
        width, height = (int(v) for v in a.image_size.lower().split("x"))
    spec = ImageSpec.from_env(backend=a.image_backend, url=a.image_url, model=a.image_model, width=width,
                              height=height, steps=a.image_steps, workflow=a.comfy_workflow, command=a.image_command)
    devices = [d.strip() for d in a.devices.split(",")] if a.devices else None
    manager = JobManager(Path(a.out).resolve() if a.out else OUT_ROOT, devices, spec)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.token:
        print("warning: listening beyond localhost without --token; anyone who can reach it can run jobs")
    app = create_app(manager, a.token, ui=not a.no_ui)
    print(f"assetgen {version()}: devices {', '.join(manager.devices)}; jobs in {manager.root}")
    print(f"image backend: {spec.backend} at {spec.base_url or spec.command or '(not set)'}")
    shown = "127.0.0.1" if a.host == "0.0.0.0" else a.host
    print(f"web UI http://{shown}:{a.port}/   API http://{shown}:{a.port}/api/v1   reference http://{shown}:{a.port}/docs")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
