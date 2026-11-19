import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from assetgen import imagegen
from assetgen.imagegen import ImageSpec


def png(color, size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class StubImageServer(BaseHTTPRequestHandler):
    """sd-server's A1111 and OpenAI routes, answering with flat-colour images."""
    requests: list = []

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/sdapi/v1/sd-models":
            self._json([{"title": "flux-schnell", "model_name": "flux-schnell"}])
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "sd-cpp-local"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubImageServer.requests.append((self.path, body, self.headers.get("Authorization")))
        if self.path in ("/sdapi/v1/txt2img", "/sdapi/v1/img2img"):
            n = body.get("batch_size", 1)
            self._json({"images": [base64.b64encode(png((40 * k, 80, 120))).decode() for k in range(n)],
                        "parameters": body, "info": ""})
        elif self.path == "/v1/images/generations":
            self._json({"created": 0, "data": [{"b64_json": base64.b64encode(png((200, 10, 10))).decode()}
                                               for _ in range(body.get("n", 1))]})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


@pytest.fixture
def image_server():
    StubImageServer.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubImageServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_sdapi_frames_the_prompt_and_writes_candidates(image_server, tmp_path):
    spec = ImageSpec(backend="sdcpp", url=image_server, width=512, height=768, steps=8)
    imagegen.write("a goblin with a lantern.", tmp_path, spec, seed=7, count=3, log=lambda *_: None)
    path, body, _ = StubImageServer.requests[0]
    assert path == "/sdapi/v1/txt2img"
    assert body["prompt"].startswith("a goblin with a lantern, full body character")
    assert (body["width"], body["height"], body["steps"], body["seed"], body["batch_size"]) == (512, 768, 8, 7, 3)
    assert body["negative_prompt"] == imagegen.NEGATIVE
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["candidate_0.png", "candidate_1.png",
                                                             "candidate_2.png", "image.png"]
    meta = json.loads((tmp_path / "image.json").read_text())
    assert meta["prompt"] == "a goblin with a lantern." and meta["seed"] == 7 and meta["backend"] == "sdcpp"


def test_raw_prompt_is_sent_as_written(image_server):
    spec = ImageSpec(backend="a1111", url=image_server, style=False)
    imagegen.generate("pixel art knight", spec, seed=1)
    assert StubImageServer.requests[0][1]["prompt"] == "pixel art knight"


def test_openai_backend_reads_key_from_environment(image_server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ASSETGEN_IMAGE_API_KEY", raising=False)
    images = imagegen.generate("a robot", ImageSpec(backend="openai", url=image_server, model="m"), seed=1, count=2)
    path, body, auth = StubImageServer.requests[0]
    assert path == "/v1/images/generations" and body["n"] == 2 and body["size"] == "1024x1024"
    assert auth == "Bearer sk-test" and len(images) == 2 and images[0].getpixel((0, 0)) == (200, 10, 10)


def test_probe_reports_models_and_unreachable_servers(image_server):
    assert imagegen.probe(ImageSpec(backend="sdcpp", url=image_server)) == {
        "ok": True, "status": 200, "models": ["flux-schnell"]}
    assert imagegen.probe(ImageSpec(backend="sdcpp", url="http://127.0.0.1:9"))["ok"] is False


def test_command_backend_fills_the_template(tmp_path):
    script = tmp_path / "draw.py"
    script.write_text("import sys\nfrom PIL import Image\nImage.new('RGB', (8, 8), (1, 2, 3)).save(sys.argv[2])\n")
    import sys

    spec = ImageSpec(backend="command", command=f'"{sys.executable}" "{script}" {{prompt}} {{out}}')
    images = imagegen.generate("a cat", spec, seed=3)
    assert images[0].getpixel((0, 0)) == (1, 2, 3)


def test_comfyui_placeholders_keep_types():
    workflow = {"3": {"inputs": {"seed": "{{seed}}", "text": "x {{prompt}} y"}}}
    filled = imagegen._fill(workflow, {"seed": 5, "prompt": "fox"})
    assert filled == {"3": {"inputs": {"seed": 5, "text": "x fox y"}}}


def test_register_adds_a_backend(tmp_path):
    imagegen.register("flat", lambda prompt, spec, seed, count: [png((9, 9, 9))] * count)
    try:
        out = imagegen.write("anything", tmp_path, ImageSpec(backend="flat"), seed=0, log=lambda *_: None)
        assert Image.open(out).getpixel((0, 0)) == (9, 9, 9)
    finally:
        imagegen.BACKENDS.pop("flat")


def test_unknown_backend_is_an_error():
    with pytest.raises(ValueError, match="unknown image backend"):
        imagegen.generate("x", ImageSpec(backend="nope"))


def test_image_stage_runs_before_generate(tmp_path):
    from assetgen.pipeline import Options, stage_commands

    o = Options(out=tmp_path, text="a goblin", image_spec=ImageSpec(backend="sdcpp", url="http://h:1", steps=4))
    cmds = stage_commands(o)
    assert list(cmds)[:2] == ["image", "generate"]
    args = cmds["image"][0]
    assert args[:3] == ["assetgen.imagegen", "--prompt", "a goblin"] and "--steps" in args and "http://h:1" in args
    assert cmds["generate"][0][2] == str(Path(tmp_path).resolve() / "0_image" / "image.png")


RECIPES = json.loads((Path(__file__).parent / "data" / "sdcpp_recipes.json").read_text())  # from sd.cpp's built-ins


def catalog_files(*ids):
    return [{"id": i, "name": i.split("/")[-1], "category": i.split("/")[0], "relative_path": i, "root": "@r"}
            for i in ids]


class StubSdServer(StubImageServer):
    """sd-server with its model API and catalog: models load by recipe and files, and unload."""
    state: dict = {}
    files: list = []

    def do_GET(self):
        if self.path == "/sdcpp/v1/models/status":
            self._json(StubSdServer.state)
        elif self.path == "/sdcpp/v1/recipes":
            self._json({**RECIPES, "errors": []})
        elif self.path == "/sdcpp/v1/model-files":
            self._json({"files": StubSdServer.files, "errors": []})
        else:
            super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        StubImageServer.requests.append((self.path, body, None))
        state = StubSdServer.state
        if self.path == "/sdcpp/v1/models/unload":
            StubSdServer.state = {"state": "empty", "recipe": None, "models": {}}
            self._json(StubSdServer.state)
        elif self.path == "/sdcpp/v1/models/load":
            known = {f["id"] for f in StubSdServer.files}
            if any(v not in known for v in body.get("models", {}).values()):
                self._json({"error": "unknown model or incorrect category", "code": "model_load_failed"}, 422)
                return
            StubSdServer.state = {"state": "ready", "recipe": body["recipe"], "models": body.get("models", {})}
            self._json(StubSdServer.state)
        elif self.path == "/sdapi/v1/txt2img":
            if state.get("state") != "ready":
                self._json({"error": "No model is loaded", "code": "model_not_loaded"}, 503)
                return
            if state.pop("unload_before_next", False):  # another job unloads it first
                StubSdServer.state = {"state": "empty", "recipe": None, "models": {}}
                self._json({"error": "No model is loaded", "code": "model_not_loaded"}, 503)
                return
            self._json({"images": [base64.b64encode(png((1, 2, 3))).decode()], "parameters": body, "info": ""})
        else:
            self._json({"error": "not found"}, 404)


SD15 = {"recipe": "sd1-checkpoint", "models": {"model": "checkpoints/sd15-q4_0.gguf"}}
EMPTY = {"state": "empty", "recipe": None, "models": {}}


@pytest.fixture
def sd_server(monkeypatch, tmp_path):
    monkeypatch.setattr(imagegen, "STATE_DIR", tmp_path / "state")
    StubImageServer.requests = []
    StubSdServer.state = {"state": "ready", **SD15}
    StubSdServer.files = catalog_files("checkpoints/sd15-q4_0.gguf")
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubSdServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def calls():
    return [path for path, _, _ in StubImageServer.requests]


def test_sdcpp_unloads_after_the_picture_and_loads_it_back(sd_server, tmp_path):
    spec = ImageSpec(backend="sdcpp", url=sd_server)
    imagegen.generate("a fox", spec, seed=1)
    assert calls() == ["/sdapi/v1/txt2img", "/sdcpp/v1/models/unload"]
    assert StubSdServer.state["state"] == "empty"
    assert json.loads((tmp_path / "state" / "sdcpp_models.json").read_text()) == {sd_server: SD15}

    StubImageServer.requests = []
    imagegen.generate("a fox", spec, seed=2)  # a new run finds it unloaded
    assert calls() == ["/sdcpp/v1/models/load", "/sdapi/v1/txt2img", "/sdcpp/v1/models/unload"]
    assert StubImageServer.requests[0][1] == SD15


def test_sdcpp_loads_a_model_on_an_empty_server_by_itself(sd_server):
    """A freshly started sd-server with nothing loaded and nothing remembered: the one model there is."""
    StubSdServer.state = dict(EMPTY)
    images = imagegen.generate("a fox", ImageSpec(backend="sdcpp", url=sd_server), seed=1)
    assert len(images) == 1
    assert calls() == ["/sdcpp/v1/models/load", "/sdapi/v1/txt2img", "/sdcpp/v1/models/unload"]
    assert StubImageServer.requests[0][1] == SD15


def test_sdcpp_matches_a_flux_model_with_its_text_encoders_and_vae(sd_server):
    from assetgen import sdcpp

    StubSdServer.state = dict(EMPTY)
    StubSdServer.files = catalog_files("diffusion_models/flux1-schnell-q4_k.gguf", "text_encoders/clip_l.safetensors",
                                       "text_encoders/t5xxl_fp8_e4m3fn.safetensors", "vae/ae.safetensors")
    spec = ImageSpec(backend="sdcpp", url=sd_server)
    found = sdcpp.options(spec)
    assert found[0]["recipe"] == "flux1-schnell" and not any(o["recipe"] == "flux1-kontext" for o in found)
    imagegen.generate("a fox", spec, seed=1)
    assert StubImageServer.requests[0][1] == {"recipe": "flux1-schnell", "models": {
        "diffusion_model": "diffusion_models/flux1-schnell-q4_k.gguf", "clip_l": "text_encoders/clip_l.safetensors",
        "t5xxl": "text_encoders/t5xxl_fp8_e4m3fn.safetensors", "vae": "vae/ae.safetensors"}}


def test_sdcpp_names_the_choices_when_it_cannot_tell(sd_server):
    StubSdServer.state = dict(EMPTY)
    StubSdServer.files = catalog_files("checkpoints/sd15-q4_0.gguf", "checkpoints/sdxl-base-1.0.safetensors")
    with pytest.raises(RuntimeError, match="which to load") as e:
        imagegen.generate("a fox", ImageSpec(backend="sdcpp", url=sd_server), seed=1)
    assert "sd1-checkpoint:checkpoints/sd15-q4_0.gguf" in str(e.value)
    assert "sdxl-checkpoint:checkpoints/sdxl-base-1.0.safetensors" in str(e.value)


@pytest.mark.parametrize("model", ["sdxl-checkpoint:checkpoints/sdxl-base-1.0.safetensors",
                                   "sdxl-base-1.0.safetensors", "checkpoints/sdxl-base-1.0.safetensors",
                                   "sdxl-checkpoint"])
def test_sdcpp_loads_the_model_asked_for(sd_server, model):
    StubSdServer.files = catalog_files("checkpoints/sd15-q4_0.gguf", "checkpoints/sdxl-base-1.0.safetensors")
    imagegen.generate("a fox", ImageSpec(backend="sdcpp", url=sd_server, model=model), seed=1)
    assert calls() == ["/sdcpp/v1/models/load", "/sdapi/v1/txt2img", "/sdcpp/v1/models/unload"]
    assert StubImageServer.requests[0][1] == {"recipe": "sdxl-checkpoint",
                                             "models": {"model": "checkpoints/sdxl-base-1.0.safetensors"}}


def test_sdcpp_chooses_again_when_the_remembered_model_is_gone(sd_server, tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "sdcpp_models.json").write_text(json.dumps(
        {sd_server: {"recipe": "sd1-checkpoint", "models": {"model": "checkpoints/deleted.gguf"}}}))
    StubSdServer.state = dict(EMPTY)
    imagegen.generate("a fox", ImageSpec(backend="sdcpp", url=sd_server), seed=1)
    assert [c for c in calls() if c.endswith("/load")] == ["/sdcpp/v1/models/load"] * 2
    assert StubImageServer.requests[1][1] == SD15


def test_sdcpp_keep_loaded(sd_server):
    imagegen.generate("a fox", ImageSpec(backend="sdcpp", url=sd_server, keep_loaded=True), seed=1)
    assert calls() == ["/sdapi/v1/txt2img"] and StubSdServer.state["state"] == "ready"


def test_sdcpp_leaves_a_command_line_model_loaded(sd_server, capsys):
    StubSdServer.state = {"state": "ready", "origin": "command_line", "recipe": None, "models": {}}
    imagegen.generate("a fox", ImageSpec(backend="sdcpp", url=sd_server), seed=1)
    assert calls() == ["/sdapi/v1/txt2img"] and StubSdServer.state["state"] == "ready"
    assert "command line" in capsys.readouterr().out


def test_sdcpp_reloads_when_another_job_unloads_it_first(sd_server):
    StubSdServer.state["unload_before_next"] = True
    images = imagegen.generate("a fox", ImageSpec(backend="sdcpp", url=sd_server), seed=1)
    assert len(images) == 1
    assert calls() == ["/sdapi/v1/txt2img", "/sdcpp/v1/models/unload", "/sdcpp/v1/models/load",
                       "/sdapi/v1/txt2img", "/sdcpp/v1/models/unload"]


def test_image_models_endpoint(sd_server, tmp_path):
    from fastapi.testclient import TestClient

    from assetgen import server

    StubSdServer.state = dict(EMPTY)
    StubSdServer.files = catalog_files("checkpoints/sd15-q4_0.gguf", "checkpoints/sdxl-base-1.0.safetensors")
    manager = server.JobManager(tmp_path / "jobs", ["cpu"], ImageSpec(backend="sdcpp", url=sd_server))
    data = TestClient(server.create_app(manager, ui=False)).get("/api/v1/image-models").json()
    assert {o["id"] for o in data["options"]} == {"sd1-checkpoint:checkpoints/sd15-q4_0.gguf",
                                                 "sdxl-checkpoint:checkpoints/sdxl-base-1.0.safetensors"}
    assert data["automatic"] is None and data["loaded"] is None and data["error"] is None


def test_each_kind_is_framed_for_its_use(image_server):
    imagegen.generate("a wooden barrel.", ImageSpec(backend="a1111", url=image_server, kind="prop"), seed=1)
    body = StubImageServer.requests[0][1]
    assert body["prompt"].startswith("a wooden barrel, a single game prop")
    assert body["negative_prompt"] == imagegen.NEGATIVES["prop"]
    with pytest.raises(ValueError, match="unknown kind"):
        imagegen.frame("x", ImageSpec(kind="vehicle"))


def test_redraw_sends_the_picture_fitted_to_the_canvas(image_server, tmp_path):
    crop = Image.new("RGBA", (300, 100), (0, 0, 0, 0))
    crop.paste((200, 30, 30, 255), (100, 0, 200, 100))
    crop.save(tmp_path / "crop.png")
    Image.new("L", (300, 100), 255).save(tmp_path / "mask.png")
    spec = ImageSpec(backend="a1111", url=image_server, kind="prop", width=512, height=512)
    imagegen.write("a red crate", tmp_path / "out", spec, seed=3, init=tmp_path / "crop.png", strength=0.45,
                   mask=tmp_path / "mask.png", log=lambda *_: None)
    path, body, _ = StubImageServer.requests[0]
    assert path == "/sdapi/v1/img2img" and body["denoising_strength"] == 0.45
    init = Image.open(io.BytesIO(base64.b64decode(body["init_images"][0])))
    assert init.size == (512, 512)
    assert init.getpixel((256, 256)) == (200, 30, 30) and init.getpixel((5, 5)) == (255, 255, 255)
    mask = Image.open(io.BytesIO(base64.b64decode(body["mask"]))).convert("L")
    assert mask.getpixel((256, 256)) == 255 and mask.getpixel((256, 5)) == 0  # outside the picture stays
    meta = json.loads((tmp_path / "out" / "image.json").read_text())
    assert meta["init"] == "crop.png" and meta["strength"] == 0.45 and meta["kind"] == "prop"


def test_command_backend_redraws_with_its_edit_template(tmp_path):
    script = tmp_path / "redraw.py"
    script.write_text("import sys\nfrom PIL import Image\n"
                      "Image.open(sys.argv[1]).point(lambda v: 255 - v).save(sys.argv[2])\n")
    import sys

    spec = ImageSpec(backend="command", command="unused {prompt} {out}", width=32, height=32,
                     edit_command=f'"{sys.executable}" "{script}" {{init}} {{out}}')
    images = imagegen.generate("a cat", spec, seed=3, init=Image.new("RGB", (32, 32), (10, 20, 30)))
    assert images[0].getpixel((0, 0)) == (245, 235, 225)
    with pytest.raises(ValueError, match=r"with \{init\}"):
        imagegen.generate("a cat", ImageSpec(backend="command", command="x {out}"), init=Image.new("RGB", (4, 4)))
