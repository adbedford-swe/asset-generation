"""The character file: an engine-neutral description of a finished asset.

``<name>.assetgen`` sits next to ``<name>.glb`` and says what an engine needs to
set the asset up without guessing: its units and axes, its skeleton, and for
every clip whether it loops, how long it is, how fast it travels and what it is
for (idle, walk, run, a one-shot action). The Unity and Godot integrations read
it to build animator controllers, locomotion blend trees and prefabs or scenes;
anything else can read it as plain JSON (schema: ``schema/character.schema.json``).

The glTF stays the single source of the geometry and motion; this file only
describes it, so either can be regenerated from a run folder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

FORMAT = "assetgen.character"
VERSION = 1
EXTENSION = ".assetgen"
SCHEMA_URL = "https://raw.githubusercontent.com/adbedford-swe/asset-generation/main/schema/character.schema.json"
FPS = 30

# Prompt words that say what a clip is for. Checked in this order; the first match wins.
ROLE_WORDS = [
    ("idle", r"\b(idle|idles|stands?|standing|breath\w*|waits?|waiting|rests?|resting|sways?)\b"),
    ("run", r"\b(runs?|running|jogs?|jogging|sprints?|sprinting|trots?|trotting|gallops?|galloping|dash\w*"
            r"|charg\w*)\b"),
    ("walk", r"\b(walks?|walking|strolls?|sneaks?|sneaking|creeps?|creeping|limps?|limping|march\w*|paces?)\b"),
]


def character_path(glb: Path) -> Path:
    return Path(glb).with_suffix(EXTENSION)


def clip_role(prompt: str, loop: bool, speed: float, height: float) -> str:
    """idle, walk, run or move (other travelling loops) for locomotion; loop or action otherwise."""
    text = prompt.lower()
    if loop:
        for role, pattern in ROLE_WORDS:
            if re.search(pattern, text):
                return role
        return "move" if speed > 0.15 * height else "loop"
    return "action"


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("assetgen")
    except Exception:
        return "dev"


def _skeleton(g) -> dict:
    """Joints in rest pose, world space (metres, Y up, facing +Z), parents by name."""
    nodes = g.doc.nodes
    joints = list(g.skin.joints)
    names = [nodes[j].name or f"joint_{k}" for k, j in enumerate(joints)]
    in_skin = {j: names[k] for k, j in enumerate(joints)}
    positions = g.joint_positions(None)
    out = []
    for k, j in enumerate(joints):
        parent = g.parent.get(j)
        out.append({"name": names[k], "parent": in_skin.get(parent),
                    "position": [round(float(v), 5) for v in positions[k]]})
    roots = [j["name"] for j in out if j["parent"] is None]
    armature = g.parent.get(joints[0])
    while armature is not None and armature in in_skin:
        armature = g.parent.get(armature)
    return {"root": roots[0] if roots else None,
            "armature": nodes[armature].name if armature is not None else None,
            "joints": out}


def build(glb: Path, info: dict | None = None, source: dict | None = None) -> dict:
    """The character description of a finished ``<name>.glb``.

    ``info`` is what finalize measured (``<name>.json``: seams, scale); ``source`` says what the
    asset was made from (prompt, image, backend). Both are optional, so any glb this pipeline
    finalized can be described after the fact."""
    from .finalize import read_accessor
    from .gltf_anim import AnimatedGLB

    glb = Path(glb)
    g = AnimatedGLB(glb)
    doc = g.doc
    rest = g.posed(None)
    lo, hi = rest.min(0), rest.max(0)
    height = float(hi[1] - lo[1])
    measured = (info or {}).get("animations", {})

    clips = []
    for i, anim in enumerate(doc.animations):
        extras = dict(anim.extras or {})
        extras.update({k: v for k, v in measured.get(anim.name, {}).items() if k not in extras})
        times = np.concatenate([read_accessor(doc, g.blob, s.input)[:, 0] for s in anim.samplers])
        duration = float(times.max() - times.min())
        loop = bool(extras.get("loop", False))
        velocity = [float(v) for v in extras.get("root_velocity", [0.0, 0.0, 0.0])]
        speed = float(np.hypot(velocity[0], velocity[2]))
        prompt = extras.get("prompt", anim.name or f"clip_{i}")
        fps = float(extras.get("fps") or FPS)
        clip = {"name": anim.name or f"clip_{i}", "prompt": prompt, "role": clip_role(prompt, loop, speed, height),
                "loop": loop, "duration": round(duration, 4), "frames": int(round(duration * fps)) + 1,
                "fps": int(fps) if fps.is_integer() else fps, "source": extras.get("source", "unimate")}
        if loop:
            clip["root_motion"] = extras.get("root_motion", "in_place")
            clip["root_velocity"] = [round(v, 5) for v in velocity]
            clip["speed"] = round(speed, 5)
            if "seam" in extras:
                clip["seam"] = {k: float(f"{v:.3g}") for k, v in extras["seam"].items()}
        for key in ("events", "phases", "sound"):
            if extras.get(key):
                clip[key] = extras[key]
        clips.append(clip)

    character = {
        "$schema": SCHEMA_URL,
        "format": FORMAT,
        "version": VERSION,
        "name": glb.stem,
        "generator": {"name": "assetgen", "version": _version()},
        "model": {
            "file": glb.name,
            "units": "meters",
            "up": "+Y",
            "forward": "+Z",
            "height": round(height, 4),
            "bounds": {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi]},
            "vertices": int(len(g.positions)),
            "triangles": int(len(g.faces)),
            "texture": list(g.texture.shape[1::-1]) if g.texture is not None else None,
        },
        "skeleton": _skeleton(g),
        "clips": clips,
        "locomotion": locomotion(clips),
        "license": motion_license({c["source"] for c in clips}),
    }
    if source:
        character["source"] = {k: v for k, v in source.items() if v not in (None, "", [])}
    preview = glb.with_suffix(".png")
    if preview.exists():
        character["preview"] = preview.name
    return character


def motion_license(sources: set[str]) -> dict:
    made_with = []
    if not sources or "unimate" in sources:
        made_with.append("UniMate's released checkpoints")
    if "video" in sources:
        made_with.append("CoTracker3 (motion from video)")
    return {"motion": "CC-BY-NC-4.0",
            "note": f"Animated with {' and '.join(made_with)}, both CC BY-NC 4.0: non-commercial use only."
            if len(made_with) > 1 else f"Animated with {made_with[0]} (CC BY-NC 4.0): non-commercial use only."}


def locomotion(clips: list[dict]) -> dict:
    """The clips an engine should blend by ground speed: one idle at 0 m/s, then travelling loops
    in order of speed (one per distinct speed). Empty when there is nothing to blend."""
    idle = next((c for c in clips if c["role"] == "idle"), None)
    moving = sorted((c for c in clips if c["loop"] and c["role"] in ("walk", "run", "move")),
                    key=lambda c: c["speed"])
    blend, last = [], 0.0
    if idle:
        blend.append({"clip": idle["name"], "speed": 0.0})
    for c in moving:
        if c["speed"] > last + 0.05:
            blend.append({"clip": c["name"], "speed": c["speed"]})
            last = c["speed"]
    if len(blend) < 2:
        return {"idle": idle["name"] if idle else None, "blend": []}
    return {"idle": idle["name"] if idle else None, "blend": blend}


def run_source(run_dir: Path) -> dict:
    """What a run folder's asset was made from, for the character's ``source`` block."""
    run_dir = Path(run_dir)
    source = {}
    image_info = run_dir / "0_image" / "image.json"
    if image_info.exists():
        meta = json.loads(image_info.read_text())
        source.update({"prompt": meta.get("prompt"), "image_backend": meta.get("backend"),
                       "image_model": meta.get("model"), "seed": meta.get("seed")})
    run = run_dir / "run.json"
    if run.exists():
        inputs = json.loads(run.read_text())
        if inputs.get("image") and not source.get("prompt"):
            source["image"] = Path(inputs["image"]).name
        if inputs.get("ply"):
            source["splat"] = Path(inputs["ply"]).name
    return source


def write(glb: Path, info: dict | None = None, source: dict | None = None) -> Path:
    glb = Path(glb)
    if info is None and glb.with_suffix(".json").exists():
        info = json.loads(glb.with_suffix(".json").read_text())
    if source is None:
        source = run_source(glb.parent)
    path = character_path(glb)
    path.write_text(json.dumps(build(glb, info, source), indent=2))
    return path


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not an assetgen character file")
    if data.get("version", 0) > VERSION:
        raise ValueError(f"{path} is character format v{data['version']}; this assetgen reads up to v{VERSION}")
    return data


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="write the character file for a finished .glb")
    ap.add_argument("glb")
    a = ap.parse_args(argv)
    print(write(Path(a.glb)))


if __name__ == "__main__":
    main()
