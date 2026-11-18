"""Package a finished character for an engine.

    assetgen export outputs/goblin --engine unity --project ~/MyGame      # straight into a Unity project
    assetgen export outputs/goblin --engine godot --project ~/my_godot_game
    assetgen export outputs/goblin --engine godot --zip                    # goblin-godot.zip to hand over

Every export is the same few files: ``<name>.glb`` (mesh, skeleton, every clip),
``<name>.assetgen`` (the character file), and the clip contact sheet. The engine
integrations do the rest on import:

  unity  the com.assetgen.unity package (in this repository's ``unity/`` folder) builds an
         Animator Controller with a locomotion blend tree and a ready-to-use prefab.
  godot  the ``assetgen`` addon (``godot/addons/assetgen``) sets each clip's loop mode and builds
         an AnimationTree; the export writes the ``.glb.import`` that points Godot at it.

With ``--project`` the files are copied into the project, and the integration is added to it
when it is missing (Unity: a ``Packages/manifest.json`` entry; Godot: the addon, enabled).
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from . import character
from .thirdparty import ROOT

ENGINES = ("unity", "godot", "gltf")
UNITY_PACKAGE = "com.assetgen.unity"
UNITY_PACKAGE_DIR = ROOT / "unity" / UNITY_PACKAGE
UNITY_GIT_URL = f"https://github.com/adbedford-swe/asset-generation.git?path=/unity/{UNITY_PACKAGE}"
GODOT_ADDON_DIR = ROOT / "godot" / "addons" / "assetgen"
GODOT_IMPORT_SCRIPT = "res://addons/assetgen/character_import.gd"


def find_asset(source: Path) -> Path:
    """The finished ``<name>.glb`` of a run folder, or ``source`` itself when it is a .glb."""
    source = Path(source)
    if source.suffix.lower() == ".glb":
        if not source.exists():
            raise FileNotFoundError(source)
        return source
    if source.suffix == character.EXTENSION:
        return source.with_suffix(".glb")
    from .pipeline import run_inputs

    try:
        name = run_inputs(source).get("name")
    except Exception:
        name = None
    candidates = [source / f"{name}.glb"] if name else []
    candidates += sorted(p for p in source.glob("*.glb"))
    for glb in candidates:
        if glb.exists():
            return glb
    raise FileNotFoundError(f"no finished asset in {source}: run the finalize stage first")


def ensure_character(glb: Path) -> Path:
    """The character file beside ``glb``, written now for runs finalized before it existed."""
    path = character.character_path(glb)
    if not path.exists():
        character.write(glb)
    return path


def _godot_import_file(data: dict) -> str:
    """A ``.glb.import`` that runs the addon's import script. Godot fills in the rest on first import."""
    return "\n".join([
        "[remap]", "", 'importer="scene"', 'type="PackedScene"', "", "[params]", "",
        'nodes/root_type="Node3D"', f'nodes/root_name="{data["name"]}"',
        f'import_script/path="{GODOT_IMPORT_SCRIPT}"', "animation/import=true", "animation/fps=30",
        "animation/trimming=false", "animation/remove_immutable_tracks=true", ""])


def export(source: Path, engine: str, out_dir: Path) -> Path:
    """Write ``out_dir/<name>/`` for ``engine`` and return it."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; one of {', '.join(ENGINES)}")
    glb = find_asset(source)
    data = character.load(ensure_character(glb))
    name = glb.stem
    data["model"]["file"] = glb.name  # the files travel together under the model's name
    folder = Path(out_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(glb, folder / glb.name)
    (folder / f"{name}{character.EXTENSION}").write_text(json.dumps(data, indent=2))
    preview = glb.with_suffix(".png")
    if preview.exists():
        shutil.copy2(preview, folder / f"{name}_clips.png")
    if engine == "godot":
        (folder / f"{glb.name}.import").write_text(_godot_import_file(data))
    return folder


def export_zip(source: Path, engine: str, zip_path: Path | None = None) -> Path:
    glb = find_asset(source)
    zip_path = Path(zip_path or glb.with_name(f"{glb.stem}-{engine}.zip"))
    with tempfile.TemporaryDirectory() as tmp:
        folder = export(glb, engine, Path(tmp))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(folder.rglob("*")):
                z.write(f, Path(folder.name) / f.relative_to(folder))
    return zip_path


def unity_project(path: Path) -> Path:
    path = Path(path)
    if not (path / "Assets").is_dir() or not (path / "Packages" / "manifest.json").exists():
        raise ValueError(f"{path} is not a Unity project (no Assets/ and Packages/manifest.json)")
    return path


def add_unity_package(project: Path, source: str = "auto") -> str | None:
    """Reference com.assetgen.unity from the project's manifest, if it is not there yet.
    ``source``: "local" (this checkout's copy, as a file: path), "git", or "auto" (local when present)."""
    manifest_path = Path(project) / "Packages" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    deps = manifest.setdefault("dependencies", {})
    if UNITY_PACKAGE in deps:
        return None
    use_local = source == "local" or (source == "auto" and UNITY_PACKAGE_DIR.exists())
    deps[UNITY_PACKAGE] = f"file:{UNITY_PACKAGE_DIR.resolve().as_posix()}" if use_local else UNITY_GIT_URL
    manifest["dependencies"] = dict(sorted(deps.items()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return deps[UNITY_PACKAGE]


def godot_project(path: Path) -> Path:
    path = Path(path)
    if not (path / "project.godot").exists():
        raise ValueError(f"{path} is not a Godot project (no project.godot)")
    return path


def enable_godot_addon(project: Path) -> bool:
    """Copy the addon into ``addons/assetgen`` (replacing an older copy) and enable it in project.godot.
    Returns True when project.godot changed."""
    project = Path(project)
    target = project / "addons" / "assetgen"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(GODOT_ADDON_DIR, target)
    cfg = project / "project.godot"
    text = cfg.read_text()
    entry = '"res://addons/assetgen/plugin.cfg"'
    if entry in text:
        return False
    m = re.search(r"^enabled=PackedStringArray\((.*)\)$", text, re.MULTILINE)
    if m:
        inner = m.group(1).strip()
        text = text[:m.start(1)] + (f"{inner}, {entry}" if inner else entry) + text[m.end(1):]
    elif "[editor_plugins]" in text:
        text = text.replace("[editor_plugins]", f"[editor_plugins]\n\nenabled=PackedStringArray({entry})", 1)
    else:
        text = text.rstrip("\n") + f"\n\n[editor_plugins]\n\nenabled=PackedStringArray({entry})\n"
    cfg.write_text(text)
    return True


def install(source: Path, engine: str, project: Path, folder: str | None = None, package_source: str = "auto",
            log=print) -> Path:
    """Export ``source`` straight into an engine project. Returns the character's folder there."""
    if engine == "unity":
        project = unity_project(project)
        dest = export(source, engine, project / "Assets" / (folder or "Characters"))
        added = add_unity_package(project, package_source)
        if added:
            log(f"added {UNITY_PACKAGE} to {project / 'Packages' / 'manifest.json'} ({added})")
    elif engine == "godot":
        project = godot_project(project)
        dest = export(source, engine, project / (folder or "characters"))
        if enable_godot_addon(project):
            log(f"enabled the assetgen addon in {project / 'project.godot'}")
    else:
        dest = export(source, engine, Path(project) / (folder or ""))
    log(f"exported {dest.name} to {dest}")
    return dest
