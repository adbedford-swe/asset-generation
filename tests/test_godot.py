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
    for line in ("crate's collision shape is a ConvexPolygonShape3D", "ruin's collision shape is a ConcavePolygonShape3D",
                 "block's collision shape is a BoxShape3D", "ghost has no collider", "crate sits on y = 0"):
        assert any(line in r for r in report), line
    assert sum("a ray from above lands on its top" in r for r in report) == 3
