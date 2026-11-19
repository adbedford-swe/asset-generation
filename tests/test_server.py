import base64
import io
import json
import shutil
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from assetgen import server
from assetgen.imagegen import ImageSpec
from tests.test_character import finished  # noqa: F401  (fixture)


def wait_for(job, status=("done", "failed", "cancelled"), timeout=10.0):
    end = time.time() + timeout
    while job.status not in status and time.time() < end:
        time.sleep(0.02)
    return job.status


@pytest.fixture
def fake_pipeline(monkeypatch, finished):  # noqa: F811
    """Stands in for the stages: reports each one and leaves a finished character behind."""
    calls = []

    def run(o, start=None, stop=None, force=False, log=print, cancel=None, on_stage=None):
        calls.append(o)
        (o.out / "run.json").write_text(json.dumps({"text": o.text, "name": o.name, "kind": o.kind}))
        for stage in server.Job(o.out, o.name, o, start).planned_stages():
            on_stage(stage, "running")
            log(f"working on {stage}")
            if cancel is not None and cancel.wait(0.05 if o.name != "slow" else 5):
                raise server.Cancelled(f"cancelled during {stage}")
            on_stage(stage, "done")
        for suffix in (".glb", ".assetgen"):
            shutil.copy2(finished.with_suffix(suffix), o.out / f"{o.name}{suffix}")
        return o.out / f"{o.name}.glb"

    monkeypatch.setattr(server, "run_pipeline", run)
    return calls


@pytest.fixture
def api(tmp_path, fake_pipeline):
    manager = server.JobManager(tmp_path / "jobs", ["cpu"], ImageSpec(backend="sdcpp", url="http://127.0.0.1:9"))
    return TestClient(server.create_app(manager, ui=False)), manager


def png_data_url():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_info(api):
    client, _ = api
    info = client.get("/api/v1/info").json()
    assert info["devices"] == ["cpu"] and info["stages"][0] == "image"
    assert info["image_backend"]["backend"] == "sdcpp" and info["image_backend"]["ok"] is False


def test_text_job_runs_the_image_stage_first(api, fake_pipeline):
    client, manager = api
    r = client.post("/api/v1/jobs", json={"prompt": "a goblin with a lantern", "motions": ["An object walks forward."],
                                          "height": 1.2, "settings": {"num_beams": 2}})
    assert r.status_code == 202
    job = r.json()
    assert job["name"] == "a_goblin_with_a_lantern" and job["input"] == {"prompt": "a goblin with a lantern"}
    assert list(job["stages"]) == ["image", "generate", "mesh", "rig", "animate", "finalize"]
    assert wait_for(manager.get(job["id"])) == "done"
    o = fake_pipeline[0]
    assert o.text == "a goblin with a lantern" and o.height == 1.2 and o.num_beams == 2 and o.device == "cpu"
    assert o.prompts == ["An object walks forward."] and o.image_spec.backend == "sdcpp"
    done = client.get(f"/api/v1/jobs/{job['id']}").json()
    assert done["progress"] == 1.0 and done["character"]["name"] == "hero"
    assert set(done["stages"].values()) == {"done"}


def test_image_job_and_exports(api):
    client, manager = api
    job = client.post("/api/v1/jobs", json={"image": png_data_url(), "name": "knight"}).json()
    assert "image" not in job["stages"] and job["input"]["image"].endswith(".png")
    assert wait_for(manager.get(job["id"])) == "done"
    character = client.get(f"/api/v1/jobs/{job['id']}/character")
    assert character.status_code == 200 and character.json()["format"] == "assetgen.character"
    r = client.get(f"/api/v1/jobs/{job['id']}/export/godot")
    assert r.status_code == 200
    assert "knight/knight.glb.import" in zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    model = client.get(job["urls"]["self"] + "/files/knight.glb")
    assert model.status_code == 200 and model.headers["content-type"] == "model/gltf-binary"
    assert client.get(job["urls"]["self"] + "/files/../../secret").status_code == 404
    assert client.get("/api/v1/jobs/..%2F..").status_code == 404


def test_upload_and_bad_requests(api):
    client, _ = api
    r = client.post("/api/v1/uploads", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 422
    r = client.post("/api/v1/uploads", files={"file": ("splat.ply", b"ply\n")})
    assert r.status_code == 200 and r.json()["id"]
    assert client.post("/api/v1/jobs", json={}).status_code == 422
    assert client.post("/api/v1/jobs", json={"prompt": "x", "settings": {"bogus": 1}}).status_code == 422
    assert client.post("/api/v1/jobs", json={"upload": "1-abc"}).status_code == 404
    assert client.get("/api/v1/jobs/nope").status_code == 404


def test_cancel_a_running_job(api):
    client, manager = api
    job = client.post("/api/v1/jobs", json={"prompt": "x", "name": "slow"}).json()
    j = manager.get(job["id"])
    assert wait_for(j, ("running",)) == "running"
    while not j.stages:
        time.sleep(0.01)
    client.post(f"/api/v1/jobs/{job['id']}/cancel")
    assert wait_for(j) == "cancelled"
    assert "cancelled" in j.stages.values() and "running" not in j.stages.values()


def test_continue_keeps_the_source(api, fake_pipeline):
    client, manager = api
    first = client.post("/api/v1/jobs", json={"prompt": "a hero"}).json()
    wait_for(manager.get(first["id"]))
    folder = manager.root / first["id"]
    for stage_dir in ("1_mesh", "2_rig", "3_motion"):
        (folder / stage_dir).mkdir()
    (folder / "manifest.json").write_text(json.dumps(
        {"stages": {s: {"args": [], "ok": True} for s in ("mesh", "rig", "animate", "finalize")}}))
    r = client.post(f"/api/v1/jobs/{first['id']}/continue",
                    json={"from": "animate", "motions": ["An object waves. [once]"]})
    assert r.status_code == 202
    second = r.json()
    assert second["id"].endswith("-from-animate") and list(second["stages"]) == ["animate", "finalize"]
    assert wait_for(manager.get(second["id"])) == "done"
    assert fake_pipeline[-1].prompts == ["An object waves. [once]"] and fake_pipeline[-1].text == "a hero"
    assert client.post(f"/api/v1/jobs/{first['id']}/continue", json={"from": "image"}).status_code == 422


def test_runs_made_elsewhere_are_listed(api, finished):  # noqa: F811
    client, manager = api
    old = manager.root / "20250101-000000-hero"
    old.mkdir(parents=True)
    shutil.copy2(finished, old / "hero.glb")
    (old / "run.json").write_text(json.dumps({"name": "hero"}))
    (old / "manifest.json").write_text(json.dumps({"stages": {"finalize": {"ok": True}}}))
    listed = client.get("/api/v1/jobs").json()["jobs"]
    assert [j["id"] for j in listed] == ["20250101-000000-hero"] and listed[0]["status"] == "done"
    # A run finalized before character files existed gets one on first request.
    assert client.get("/api/v1/jobs/20250101-000000-hero/character").json()["name"] == "hero"


def test_token(tmp_path):
    manager = server.JobManager(tmp_path, ["cpu"])
    client = TestClient(server.create_app(manager, token="s3cret", ui=False))
    assert client.get("/api/v1/info").status_code == 401
    assert client.get("/api/v1/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_settings_parse():
    s = server.Settings.parse({"loop": 120, "num_beams": 4})
    assert s.loop == "120" and s.num_beams == 4
    with pytest.raises(ValueError, match="unknown settings: nope"):
        server.Settings.parse({"nope": 1})


def test_step_settings_reach_the_stages(api, fake_pipeline):
    from assetgen.pipeline import stage_commands

    client, manager = api
    job = client.post("/api/v1/jobs", json={"image": png_data_url(), "settings": {"idol_steps": 24, "flow_steps": 16}})
    assert wait_for(manager.get(job.json()["id"])) == "done"
    o = fake_pipeline[-1]
    assert (o.idol_steps, o.flow_steps) == (24, 16)
    cmds = stage_commands(o)
    assert cmds["generate"][0][-2:] == ["--steps", "24"]
    assert cmds["animate"][0][cmds["animate"][0].index("--flow_steps") + 1] == "16"
    assert client.get("/api/v1/info").json()["settings"]["idol_steps"] == 50


def test_prop_jobs_stop_after_the_mesh(api, fake_pipeline):
    client, manager = api
    job = client.post("/api/v1/jobs", json={"prompt": "a wooden barrel", "kind": "prop"}).json()
    assert job["kind"] == "prop"
    assert list(job["stages"]) == ["image", "generate", "mesh", "finalize"]
    assert wait_for(manager.get(job["id"])) == "done"
    o = fake_pipeline[-1]
    assert o.kind == "prop" and o.height == 1.0 and o.image_spec.kind == "prop"
    sized = client.post("/api/v1/jobs", json={"prompt": "a pine", "kind": "prop", "height": 9}).json()
    assert wait_for(manager.get(sized["id"])) == "done" and fake_pipeline[-1].height == 9
    assert client.post("/api/v1/jobs", json={"prompt": "x", "kind": "vehicle"}).status_code == 422
    folder = manager.root / job["id"]
    (folder / "1_mesh").mkdir()
    (folder / "manifest.json").write_text(json.dumps(
        {"stages": {s: {"args": [], "ok": True} for s in ("mesh", "finalize")}}))
    r = client.post(f"/api/v1/jobs/{job['id']}/continue", json={"from": "rig"})
    assert r.status_code == 422 and "mesh, finalize" in r.json()["detail"]
    again = client.post(f"/api/v1/jobs/{job['id']}/continue", json={"from": "finalize"}).json()
    assert again["kind"] == "prop" and list(again["stages"]) == ["finalize"]
    assert wait_for(manager.get(again["id"])) == "done" and fake_pipeline[-1].kind == "prop"


@pytest.fixture
def fake_scene(monkeypatch, finished):  # noqa: F811
    """Stands in for the scene build: reports each step and leaves a finished scene behind."""
    import assetgen.scene.build
    import assetgen.scene.plan

    calls = []

    def build(spec, out, devices=None, image_spec=None, log=print, cancel=None, on_stage=None, **kwargs):
        calls.append((spec, out, devices, kwargs))
        for step in server.SCENE_STEPS:
            on_stage(step, "running")
            on_stage(step, "done")
        shutil.copy2(finished, out / f"{spec.name}.glb")
        (out / f"{spec.name}.assetgen").write_text(json.dumps({"format": "assetgen.scene", "version": 1,
                                                               "name": spec.name,
                                                               "model": {"file": f"{spec.name}.glb"}}))
        return out / f"{spec.name}.assetgen"

    def plan(description, url, model="local", size=None, log=print, **kwargs):
        calls.append(("plan", description, url, model))
        return assetgen.scene.plan.EXAMPLE

    monkeypatch.setattr(assetgen.scene.build, "build", build)
    monkeypatch.setattr(assetgen.scene.plan, "plan", plan)
    return calls


def test_scene_jobs(api, fake_scene):
    from assetgen.scene.plan import EXAMPLE

    client, manager = api
    assert client.get("/api/v1/scenes/example").json() == EXAMPLE
    job = client.post("/api/v1/scenes", json={"spec": EXAMPLE, "settings": {"faces": 8000}}).json()
    assert job["kind"] == "scene" and list(job["stages"]) == ["props", "ground", "layout", "assemble"]
    assert job["input"]["objects"] == [o["id"] for o in EXAMPLE["objects"]]
    assert wait_for(manager.get(job["id"])) == "done"
    spec, out, devices, kwargs = fake_scene[-1]
    assert spec.name == "forest_camp" and devices == ["cpu"] and kwargs["mesh_args"][1] == "8000"
    assert client.get(f"/api/v1/jobs/{job['id']}/export/godot").status_code == 200

    bad = dict(EXAMPLE, objects=[{"id": "x", "prompt": "x", "place": [[0, 0]]}])
    r = client.post("/api/v1/scenes", json={"spec": bad})
    assert r.status_code == 422 and "height" in r.json()["detail"]
    r = client.post("/api/v1/scenes", json={"prompt": "a forest camp"})
    assert r.status_code == 422 and "language model" in r.json()["detail"]

    planned = client.post("/api/v1/scenes", json={"prompt": "a forest camp", "name": "camp",
                                                  "llm": {"base_url": "http://127.0.0.1:9/v1"}}).json()
    assert list(planned["stages"]) == ["plan", "props", "ground", "layout", "assemble"]
    assert wait_for(manager.get(planned["id"])) == "done"
    assert fake_scene[-2] == ("plan", "a forest camp", "http://127.0.0.1:9/v1", "local")
    assert fake_scene[-1][0].name == "camp"

    (manager.root / job["id"] / "props" / "pine").mkdir(parents=True)
    again = client.post("/api/v1/scenes", json={"spec": EXAMPLE, "from": job["id"]}).json()
    assert (manager.root / again["id"] / "props" / "pine").is_dir() and again["source"] == job["id"]
    assert client.post(f"/api/v1/jobs/{job['id']}/continue", json={"from": "mesh"}).status_code == 422

    manager.jobs.clear()  # as after a restart: scenes are found again from their folders
    listed = {j["id"]: j for j in client.get("/api/v1/jobs").json()["jobs"]}
    assert listed[job["id"]]["kind"] == "scene" and listed[job["id"]]["status"] == "done"


def test_decompose_returns_a_spec_the_server_can_build(api, fake_scene, monkeypatch):
    import assetgen.scene.decompose as dec

    def decompose(image, names, out, name=None, style="", *args, **kwargs):
        (out.parent / f"{out.stem}_parts").mkdir(parents=True)
        Image.new("RGBA", (8, 8)).save(out.parent / f"{out.stem}_parts" / "barn.png")
        Image.new("RGB", (8, 8)).save(out.parent / f"{out.stem}_found.png")
        spec = {"name": name, "style": style, "ground": {"size": [10, 10]},
                "objects": [{"id": "barn", "image": f"{out.stem}_parts/barn.png", "height": 6,
                             "place": [{"position": [0, 0], "yaw": {"seen_from": 12.0}}]}]}
        out.write_text(json.dumps(spec))
        return spec

    monkeypatch.setattr(dec, "decompose", decompose)
    client, manager = api
    r = client.post("/api/v1/scenes/decompose", json={"image": png_data_url(), "objects": ["red barn"],
                                                      "name": "farm", "known": {"red barn": 6}})
    assert r.status_code == 200, r.text
    found = r.json()
    image = found["spec"]["objects"][0]["image"]
    assert image == f"draft:{found['id']}/farm_parts/barn.png"
    assert client.get(found["found"]).status_code == 200 and client.get(found["parts"][0]).status_code == 200
    assert client.get(f"/api/v1/drafts/{found['id']}/../../secret").status_code == 404
    job = client.post("/api/v1/scenes", json={"spec": found["spec"]}).json()
    assert wait_for(manager.get(job["id"])) == "done"
    built = fake_scene[-1][0]
    assert built.objects[0].image == str(manager.root / "_drafts" / found["id"] / "farm_parts" / "barn.png")
    assert client.post("/api/v1/scenes/decompose", json={"objects": []}).status_code == 422
