import json
import zipfile

import numpy as np
import pygltflib as gl
import pytest

from assetgen import character, export
from assetgen.finalize import append_accessor, ground, merge_clips
from assetgen.gltf import write_rigged_glb
from assetgen.gltf_anim import AnimatedGLB
from tests.test_gltf import add_rotation_clip, column_asset


def add_travel_clip(path, distance=1.0, seconds=1.0, lift=0.0):
    """Move the root forward (+z) by ``distance`` over ``seconds``, raised by ``lift``, and turn 'upper'."""
    doc = gl.GLTF2.load_binary(str(path))
    blob = bytearray(doc.binary_blob())
    root = next(i for i, n in enumerate(doc.nodes) if n.name == "root")
    upper = next(i for i, n in enumerate(doc.nodes) if n.name == "upper")
    t = np.linspace(0, seconds, 31, dtype=np.float32)[:, None]
    pos = np.stack([np.zeros(31), np.full(31, lift), np.linspace(0, distance, 31)], 1).astype(np.float32)
    angle = np.sin(np.linspace(0, 2 * np.pi, 31))
    q = np.stack([np.zeros(31), np.zeros(31), np.sin(angle / 4), np.cos(angle / 4)], 1).astype(np.float32)
    ti = append_accessor(doc, blob, t, gl.Accessor(componentType=gl.FLOAT, type=gl.SCALAR, min=[0.0], max=[seconds]))
    pi = append_accessor(doc, blob, pos, gl.Accessor(componentType=gl.FLOAT, type=gl.VEC3))
    qi = append_accessor(doc, blob, q, gl.Accessor(componentType=gl.FLOAT, type=gl.VEC4))
    doc.animations = [gl.Animation(name="move", samplers=[gl.AnimationSampler(input=ti, output=pi),
                                                          gl.AnimationSampler(input=ti, output=qi)],
                                   channels=[gl.AnimationChannel(sampler=0, target=gl.AnimationChannelTarget(
                                       node=root, path="translation")),
                                       gl.AnimationChannel(sampler=1, target=gl.AnimationChannelTarget(
                                           node=upper, path="rotation"))])]
    doc.buffers[0].byteLength = len(blob)
    doc.set_binary_blob(bytes(blob))
    doc.save_binary(str(path))


@pytest.fixture
def finished(tmp_path):
    """A finalized asset with an idle, a walk (0.5 m/s), a run (1.5 m/s) and a one-shot, like finalize writes."""
    mesh, rig = column_asset()
    clips = []
    for name, prompt, dist, loop in (("idle", "An object stands idle and breathes.", 0.0, True),
                                     ("walk", "An object walks forward.", 0.5, True),
                                     ("run", "An object runs forward.", 1.5, True),
                                     ("attack", "An object attacks forward. [once]", None, False)):
        path = tmp_path / f"{name}.glb"
        write_rigged_glb(mesh, rig, path)
        if dist is None:
            add_rotation_clip(path, np.pi / 3)
        else:
            add_travel_clip(path, dist, lift=-0.25)
        clips.append({"glb": str(path), "prompt": prompt.replace(" [once]", ""), "loop": loop,
                      "loop_frames": 30 if loop else None})
    out = tmp_path / "run" / "hero.glb"
    info = merge_clips(clips, out, log=lambda *_: None)
    info["ground_offset"] = ground(out, log=lambda *_: None)
    (tmp_path / "run" / "run.json").write_text(json.dumps({"text": "a hero", "name": "hero"}))
    character.write(out, info, {"prompt": "a hero"})
    return out


def test_clip_roles():
    assert character.clip_role("An object stands idle and breathes.", True, 0.0, 1.8) == "idle"
    assert character.clip_role("An object trots forward.", True, 1.0, 1.8) == "run"
    assert character.clip_role("An object walks forward.", True, 0.6, 1.8) == "walk"
    assert character.clip_role("An object crawls forward.", True, 0.6, 1.8) == "move"
    assert character.clip_role("An object dances.", True, 0.0, 1.8) == "loop"
    assert character.clip_role("An object jumps.", False, 0.0, 1.8) == "action"


def test_character_file_describes_clips_and_locomotion(finished):
    data = character.load(finished.with_suffix(".assetgen"))
    assert data["format"] == "assetgen.character" and data["version"] == 1 and data["name"] == "hero"
    assert data["model"]["file"] == "hero.glb" and data["model"]["up"] == "+Y" and data["model"]["forward"] == "+Z"
    clips = {c["name"]: c for c in data["clips"]}
    assert set(clips) == {"stands_idle_and_breathes", "walks_forward", "runs_forward", "attacks_forward"}
    assert clips["walks_forward"]["role"] == "walk" and clips["walks_forward"]["loop"]
    assert clips["walks_forward"]["speed"] == pytest.approx(0.5, abs=1e-3)
    assert clips["runs_forward"]["speed"] == pytest.approx(1.5, abs=1e-3)
    assert clips["attacks_forward"]["role"] == "action" and not clips["attacks_forward"]["loop"]
    assert "speed" not in clips["attacks_forward"]
    assert data["locomotion"] == {"idle": "stands_idle_and_breathes", "blend": [
        {"clip": "stands_idle_and_breathes", "speed": 0.0}, {"clip": "walks_forward", "speed": 0.5},
        {"clip": "runs_forward", "speed": 1.5}]}
    assert [j["name"] for j in data["skeleton"]["joints"]] == ["root", "upper"]
    assert data["skeleton"]["joints"][1]["parent"] == "root"
    assert data["source"] == {"prompt": "a hero"} and data["license"]["motion"] == "CC-BY-NC-4.0"


def test_ground_puts_the_feet_on_zero(finished):
    g = AnimatedGLB(finished)
    lows = [g.posed(1, t)[:, 1].min() for t in np.linspace(0, g.duration(1), 9)]
    assert np.median(lows) == pytest.approx(0.0, abs=1e-4)


def test_load_rejects_other_files(tmp_path):
    (tmp_path / "x.assetgen").write_text(json.dumps({"format": "something"}))
    with pytest.raises(ValueError, match="not an assetgen character"):
        character.load(tmp_path / "x.assetgen")
    (tmp_path / "y.assetgen").write_text(json.dumps({"format": character.FORMAT, "version": 99}))
    with pytest.raises(ValueError, match="v99"):
        character.load(tmp_path / "y.assetgen")


def test_schema_matches_a_written_file(finished):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((export.ROOT / "schema" / "character.schema.json").read_text())
    jsonschema.validate(json.loads(finished.with_suffix(".assetgen").read_text()), schema)


@pytest.mark.parametrize("engine", ["unity", "godot", "gltf"])
def test_export_zip(finished, engine, tmp_path):
    path = export.export_zip(finished.parent, engine, tmp_path / f"hero-{engine}.zip")
    names = set(zipfile.ZipFile(path).namelist())
    assert {"hero/hero.glb", "hero/hero.assetgen"} <= names
    assert ("hero/hero.glb.import" in names) == (engine == "godot")


def test_export_writes_character_file_for_older_runs(finished, tmp_path):
    finished.with_suffix(".assetgen").unlink()
    folder = export.export(finished, "gltf", tmp_path / "out")
    assert character.load(folder / "hero.assetgen")["name"] == "hero"


def test_install_into_unity_project(finished, tmp_path):
    project = tmp_path / "Game"
    (project / "Assets").mkdir(parents=True)
    (project / "Packages").mkdir()
    (project / "Packages" / "manifest.json").write_text(json.dumps({"dependencies": {"com.unity.ugui": "2.0.0"}}))
    dest = export.install(finished, "unity", project, log=lambda *_: None)
    assert dest == project / "Assets" / "Characters" / "hero" and (dest / "hero.assetgen").exists()
    deps = json.loads((project / "Packages" / "manifest.json").read_text())["dependencies"]
    assert deps[export.UNITY_PACKAGE].startswith("file:") and deps["com.unity.ugui"] == "2.0.0"
    deps[export.UNITY_PACKAGE] = "1.2.3"  # a pinned version the user chose stays
    (project / "Packages" / "manifest.json").write_text(json.dumps({"dependencies": deps}))
    export.install(finished, "unity", project, log=lambda *_: None)
    deps = json.loads((project / "Packages" / "manifest.json").read_text())["dependencies"]
    assert deps[export.UNITY_PACKAGE] == "1.2.3"


def test_install_into_godot_project(finished, tmp_path):
    project = tmp_path / "game"
    project.mkdir()
    (project / "project.godot").write_text('config_version=5\n\n[editor_plugins]\n\n'
                                           'enabled=PackedStringArray("res://addons/other/plugin.cfg")\n')
    dest = export.install(finished, "godot", project, log=lambda *_: None)
    assert (dest / "hero.glb.import").read_text().count("character_import.gd") == 1
    assert (project / "addons" / "assetgen" / "plugin.cfg").exists()
    text = (project / "project.godot").read_text()
    assert 'enabled=PackedStringArray("res://addons/other/plugin.cfg", "res://addons/assetgen/plugin.cfg")' in text
    export.install(finished, "godot", project, log=lambda *_: None)
    assert (project / "project.godot").read_text() == text


def test_not_an_engine_project(finished, tmp_path):
    with pytest.raises(ValueError, match="not a Unity project"):
        export.install(finished, "unity", tmp_path, log=lambda *_: None)
    with pytest.raises(ValueError, match="not a Godot project"):
        export.install(finished, "godot", tmp_path, log=lambda *_: None)
