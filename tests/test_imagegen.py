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
        if self.path == "/sdapi/v1/txt2img":
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
