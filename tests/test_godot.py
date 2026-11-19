"""The Godot addon on a real Godot, when ASSETGEN_GODOT names its binary (e.g. Godot_v4.7.2-stable_linux.x86_64)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from assetgen import export, prop
from assetgen.mesh import export_glb
from tests.test_character import finished  # noqa: F401  (fixture)
from tests.test_prop import box, textured

GODOT = os.environ.get("ASSETGEN_GODOT")


def godot_project(path: Path) -> Path:
    path.mkdir()
    (path / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="assetgen test"\n')
    return path


def run_check(project: Path, script: str, *assets: str) -> list[str]:
    """Imports the project, runs one of godot/tests' checks on it and returns its PASS/FAIL lines."""
    shutil.copy2(export.ROOT / "godot" / "tests" / script, project / script)
    subprocess.run([GODOT, "--headless", "--path", str(project), "--import"], check=True, capture_output=True,
                   timeout=600)
    run = subprocess.run([GODOT, "--headless", "--path", str(project), "--script", f"res://{script}", "--", *assets],
                         capture_output=True, text=True, timeout=600)
    report = [line for line in run.stdout.splitlines() if line.startswith(("PASS", "FAIL"))]
    assert run.returncode == 0 and report and not any(line.startswith("FAIL") for line in report), run.stdout
    return report


@pytest.mark.skipif(not GODOT, reason="set ASSETGEN_GODOT to a Godot 4 binary")
def test_character_imports_ready_to_play(finished, tmp_path):  # noqa: F811
    project = godot_project(tmp_path / "game")
    export.install(finished, "godot", project, log=lambda *_: None)
    report = run_check(project, "test_character.gd", "res://characters/hero/hero.glb")
    assert any("returns to Locomotion" in line for line in report)
    assert Path(project / "characters" / "hero" / "hero.glb.import").read_text().count("character_import.gd") == 1


@pytest.mark.skipif(not GODOT, reason="set ASSETGEN_GODOT to a Godot 4 binary")
def test_props_import_with_their_colliders(tmp_path):
    # A crate floating off-centre: finalize stands it on the ground at the origin.
    mesh = tmp_path / "mesh.glb"
    export_glb(textured(*box((1.0, 0.3, -0.5), (1.6, 1.1, 0.1))), mesh)
    project = godot_project(tmp_path / "game")
    assets = {"crate": ("prop", "auto"), "ruin": ("diorama", "auto"), "block": ("prop", "box"),
              "ghost": ("prop", "none")}
    paths = []
    for name, (kind, collider) in assets.items():
        glb = tmp_path / "runs" / name / f"{name}.glb"
        prop.finalize(mesh, glb, kind, height=1.5, collider=collider, source={"prompt": f"a {name}"},
                      log=lambda *_: None)
        folder = export.install(glb, "godot", project, log=lambda *_: None)
        paths.append(f"res://{(folder / glb.name).relative_to(project).as_posix()}")
    report = run_check(project, "test_prop.gd", *paths)
    for line in ("crate's collision shape is a ConvexPolygonShape3D",
                 "ruin's collision shape is a ConcavePolygonShape3D",
                 "block's collision shape is a BoxShape3D", "ghost has no collider", "crate sits on y = 0"):
        assert any(line in r for r in report), line
    assert sum("a ray from above lands on its top" in r for r in report) == 3


@pytest.mark.skipif(not GODOT, reason="set ASSETGEN_GODOT to a Godot 4 binary")
def test_scenes_import_with_a_collider_per_placement(tmp_path):
    from assetgen.scene import parse
    from assetgen.scene.build import build

    def asset(name, parts):
        v, f = [], []
        for lo, hi in parts:
            bv, bf = box(lo, hi)
            f += [[a + len(v), b + len(v), c + len(v)] for a, b, c in bf]
            v += bv
        export_glb(textured(v, f), tmp_path / f"{name}.glb")
        return str(tmp_path / f"{name}.glb")

    # One object per collider type: convex (the default for props), box, none, and mesh (a diorama's default).
    spec = parse({"name": "camp", "seed": 3, "ground": {"size": [24, 20], "relief": 1.0},
                  "objects": [{"id": "campfire", "asset": asset("fire", [((-.4, 0, -.4), (.4, .3, .4))]),
                               "height": 0.4, "place": [[0, 0]]},
                              {"id": "pine",
                               "asset": asset("tree", [((-.1, 0, -.1), (.1, 2, .1)), ((-1, 2, -1), (1, 5, 1))]),
                               "height": 6, "collider": "box", "scatter": {"count": 4}},
                              {"id": "rock", "asset": asset("rock", [((-.5, 0, -.4), (.5, .6, .4))]),
                               "height": 0.6, "collider": "none", "scatter": {"count": 3}},
                              {"id": "ruin", "kind": "diorama", "height": 2,
                               "asset": asset("ruin", [((-2, 0, -2), (2, .5, 2)), ((-.3, .5, -.3), (.3, 2, .3))]),
                               "place": [[-6, 5]]}]})
    out = build(spec, tmp_path / "out", devices=["cpu"], log=lambda *_: None).parent
    project = godot_project(tmp_path / "game")
    folder = export.install(out, "godot", project, log=lambda *_: None)
    scene = f"res://{(folder / 'camp.glb').relative_to(project).as_posix()}"
    report = run_check(project, "test_scene.gd", scene)
    for line in ("the ground has a trimesh shape", "campfire: every placement has a ConvexPolygonShape3D",
                 "pine: every placement has a BoxShape3D", "rock: every placement has no collider",
                 "ruin: every placement has a ConcavePolygonShape3D", "a ray from above lands on the ground",
                 "a ray from above lands on a campfire", "a ray from above lands on a pine",
                 "a ray from above lands on a ruin"):
        assert any(line in r for r in report), line

    # The objects come along as props of their own.
    props = [f"res://{p.relative_to(project).as_posix()}" for p in sorted((folder / "assets").glob("*.glb"))]
    assert len(props) == 4
    report = run_check(project, "test_prop.gd", *props)
    assert sum("the root is an AssetgenProp" in r for r in report) == 4
