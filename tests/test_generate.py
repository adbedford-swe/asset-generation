import json
import sys
import types
from pathlib import Path

from PIL import Image

from assetgen import generate


def fake_hub_cache(root: Path, repo: str, files: list[str]) -> Path:
    """The on-disk layout huggingface_hub uses: models--org--name/{refs/main, snapshots/<sha>/...}."""
    base = root / ("models--" + repo.replace("/", "--"))
    sha = "0123456789abcdef0123456789abcdef01234567"
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text(sha)
    snap = base / "snapshots" / sha
    for f in files:
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        (snap / f).write_text("{}")
    return snap


def test_encoders_resolve_to_cached_snapshots(tmp_path, monkeypatch):
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    dino = fake_hub_cache(tmp_path, "facebook/dinov3-vith16plus-pretrain-lvd1689m", ["config.json"])
    flux = fake_hub_cache(tmp_path, "black-forest-labs/FLUX.2-dev", ["vae/config.json"])
    assert generate.resolve_encoders() == {"dinov3": str(dino), "flux2_vae": str(flux)}


def test_encoders_fall_back_to_repo_ids(tmp_path, monkeypatch):
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    assert generate.resolve_encoders() == {k: repo for k, (repo, _) in generate.ENCODERS.items()}


def test_generate_writes_splat_input_and_camera(tmp_path, monkeypatch):
    calls = {}

    class Gaussian:
        def save_ply(self, path):
            Path(path).write_text("ply")

    class Pipeline:
        @classmethod
        def from_pretrained(cls, repo, image_encoders=None, device=None):
            calls["load"] = (repo, image_encoders, device)
            return cls()

        def __call__(self, image, mode, seed, steps, num_gaussians):
            calls["run"] = (image, mode, seed, steps, num_gaussians)
            return types.SimpleNamespace(gaussian=Gaussian(), preprocessed_image=Image.new("RGB", (8, 8)),
                                         camera={"yaw": 0.1, "pitch": 0.2, "radius": 2.0, "fov": 30.0})

    monkeypatch.setitem(sys.modules, "idol", types.SimpleNamespace(IdolPipeline=Pipeline))
    monkeypatch.setattr(generate, "resolve_encoders", lambda: {"dinov3": "/d", "flux2_vae": "/f"})
    camera = generate.generate("fig.png", tmp_path / "out", device="cuda:1", mode="real", seed=7, steps=20,
                               num_gaussians=65536, log=lambda *_: None)
    assert calls["load"] == ("speridlabs/idol", {"dinov3": "/d", "flux2_vae": "/f"}, "cuda:1")
    assert calls["run"] == ("fig.png", "real", 7, 20, 65536)
    out = tmp_path / "out"
    assert (out / "splat.ply").read_text() == "ply"
    assert Image.open(out / "input.png").size == (8, 8)
    assert json.loads((out / "camera.json").read_text()) == camera
