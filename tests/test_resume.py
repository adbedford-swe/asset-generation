import json
from pathlib import Path

import pytest

from assetgen import server
from assetgen.pipeline import branch, completed_stages, run_inputs, stage_commands
from assetgen.webui import list_runs


def fake_run(root: Path, name: str, done: list[str], from_image: bool = False) -> Path:
    """A run folder as the pipeline leaves it, with the given stages finished."""
    out = root / name
    for d in ("input", "0_splat", "1_mesh", "2_rig", "3_motion"):
        (out / d).mkdir(parents=True)
        (out / d / "marker.txt").write_text(d)
    (out / "input" / "splat.ply").write_text("ply")
    stages = {}
    for s in done:
        stages[s] = {"args": [f"assetgen.{s}"], "ok": True, "seconds": 1.0}
    if from_image:
        stages.setdefault("generate", {"args": ["assetgen.generate", "--image", "/pics/fox.png", "--mode", "real"],
                                       "ok": True, "seconds": 1.0})
    (out / "manifest.json").write_text(json.dumps({"stages": stages}))
    if not from_image:
        (out / "run.json").write_text(json.dumps({"image": None, "ply": str(out / "input" / "splat.ply"),
                                                  "reference": None, "name": "fox", "mode": "synthetic"}))
    return out


def test_branch_copies_only_the_stages_before_start(tmp_path):
    src = fake_run(tmp_path, "a", ["mesh", "rig", "animate", "finalize"])
    dst = branch(src, "animate", tmp_path / "b")
    assert sorted(p.name for p in dst.iterdir()) == ["1_mesh", "2_rig", "input", "manifest.json"]
    assert completed_stages(dst) == ["mesh", "rig"]
    assert (src / "3_motion" / "marker.txt").exists()  # the source is untouched


def test_branch_refuses_a_stage_whose_inputs_are_unfinished(tmp_path):
    src = fake_run(tmp_path, "a", ["mesh"])
    with pytest.raises(SystemExit, match="start from rig"):
        branch(src, "animate", tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_continued_runs_point_inside_the_new_folder(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(server, "run_pipeline", lambda o, **kw: ran.append((o, kw)))
    manager = server.JobManager(tmp_path, ["cuda:1"])
    src = fake_run(tmp_path, "20261101-000000-a", ["mesh", "rig"])
    job = manager.continue_job(src.name, "rig", server.Settings(motions=["An object waves."]))
    while job.active:
        job.wait(job.seq, 0.1)
    o, kw = ran[0]
    dst = job.folder
    assert o.ply == dst / "input" / "splat.ply" and o.name == "fox" and o.device == "cuda:1"
    assert kw["start"] == "rig" and kw["force"] and (src / "2_rig" / "marker.txt").exists()
    cmds = stage_commands(o)
    assert str(dst / "1_mesh" / "mesh.glb") in cmds["rig"][0]  # rigs the copied mesh


def test_run_inputs_of_older_runs_come_from_the_manifest(tmp_path):
    src = fake_run(tmp_path, "old", ["mesh"], from_image=True)
    inputs = run_inputs(src)
    assert inputs["image"] == "/pics/fox.png" and inputs["mode"] == "real" and inputs["ply"] is None


def test_list_runs_newest_first_with_progress(tmp_path):
    fake_run(tmp_path, "20261101-000000-a", ["mesh"])
    fake_run(tmp_path, "20261102-000000-b", ["mesh", "rig"])
    labels = [label for label, _ in list_runs(server.JobManager(tmp_path, ["cpu"]))]
    assert labels == ["20261102-000000-b  ·  mesh, rig", "20261101-000000-a  ·  mesh"]
