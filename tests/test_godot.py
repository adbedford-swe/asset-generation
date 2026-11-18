"""The Godot addon on a real Godot, when ASSETGEN_GODOT names its binary (e.g. Godot_v4.7.2-stable_linux.x86_64)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from assetgen import export
from tests.test_character import finished  # noqa: F401  (fixture)

GODOT = os.environ.get("ASSETGEN_GODOT")


@pytest.mark.skipif(not GODOT, reason="set ASSETGEN_GODOT to a Godot 4 binary")
def test_character_imports_ready_to_play(finished, tmp_path):  # noqa: F811
    project = tmp_path / "game"
    project.mkdir()
    (project / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="assetgen test"\n')
    export.install(finished, "godot", project, log=lambda *_: None)
    shutil.copy2(export.ROOT / "godot" / "tests" / "test_character.gd", project / "test_character.gd")
    subprocess.run([GODOT, "--headless", "--path", str(project), "--import"], check=True, capture_output=True,
                   timeout=600)
    run = subprocess.run([GODOT, "--headless", "--path", str(project), "--script", "res://test_character.gd", "--",
                          "res://characters/hero/hero.glb"], capture_output=True, text=True, timeout=600)
    report = [line for line in run.stdout.splitlines() if line.startswith(("PASS", "FAIL"))]
    assert run.returncode == 0 and report and not any(line.startswith("FAIL") for line in report), run.stdout
    assert any("returns to Locomotion" in line for line in report)
    assert Path(project / "characters" / "hero" / "hero.glb.import").read_text().count("character_import.gd") == 1
