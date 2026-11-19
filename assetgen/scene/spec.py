"""The scene spec: what an environment holds and where.

A scene is a ground and a few kinds of object, each generated once as a prop (or a
diorama) and placed as many times as the scene needs. The spec is JSON::

    {
      "name": "forest_camp",
      "style": "stylised low-poly, hand-painted textures, warm evening light",
      "seed": 7,
      "ground": {"size": [24, 24], "prompt": "mossy forest floor with pine needles", "relief": 0.6},
      "objects": [
        {"id": "tent", "prompt": "a canvas tent with wooden poles", "height": 2.2,
         "place": [{"position": [2, -1], "yaw": {"face": "campfire"}}]},
        {"id": "campfire", "prompt": "a ring of stones around burning logs", "height": 0.5,
         "place": [{"position": [0, 0]}]},
        {"id": "pine", "prompt": "a tall pine tree", "height": 9,
         "scatter": {"count": 14, "region": "edge", "scale": [0.8, 1.2]}}
      ]
    }

Every object comes from one of: ``prompt`` (drawn by the image backend in the scene's
style), ``image`` (a picture of it; ``redraw`` > 0 first redraws it in the scene's
style), ``ply`` (a splat) or ``asset`` (a finished prop, its .glb or run folder). The
scene's ``style`` is added to every prompt, so separately generated props share a
look; :mod:`assetgen.scene.harmonize` evens out the rest.

Positions are (x, z) in metres on the ground, or (x, y, z) to set the height instead
of standing the object on the ground. Yaw is in degrees about +Y (0 faces +Z), or
"random", or ``{"face": <object id or [x, z]>}`` to turn the object's front towards
something, or ``{"seen_from": <degrees>}`` to turn it so that a viewer in that
direction sees the side its picture showed (scenes taken from a concept picture use
this; see :mod:`assetgen.scene.decompose`). Scale multiplies the object's height.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = ("prop", "diorama")
REGIONS = ("all", "edge", "centre", "center")


class SpecError(ValueError):
    """A scene spec that cannot be built, with what is wrong and where."""


@dataclass
class Placement:
    position: list[float]               # [x, z] or [x, y, z]
    yaw: float | str | dict = 0.0       # degrees, "random", or {"face": id | [x, z]}
    scale: float = 1.0
    tilt: bool = False                  # lean with the ground's slope (rocks, logs) instead of standing upright


@dataclass
class Scatter:
    count: int
    region: str | list = "all"          # all, edge, centre, or [[x0, z0], [x1, z1]]
    spacing: float | None = None        # least distance between their centres (default: from their footprint)
    scale: list[float] = field(default_factory=lambda: [0.85, 1.15])
    yaw: float | str | dict = "random"
    tilt: bool = False
    clearance: float = 0.5              # metres kept free around explicitly placed objects


@dataclass
class ObjectSpec:
    id: str
    height: float
    prompt: str | None = None
    image: str | None = None
    ply: str | None = None
    reference: str | None = None
    asset: str | None = None
    kind: str = "prop"
    redraw: float = 0.0                 # with image: redraw it in the scene's style first (image to image strength)
    style: bool = True                  # add the scene's style to the prompt
    collider: str = "auto"
    seed: int | None = None
    place: list[Placement] = field(default_factory=list)
    scatter: Scatter | None = None

    @property
    def source(self) -> str:
        return next(k for k in ("asset", "ply", "image", "prompt") if getattr(self, k))


@dataclass
class GroundSpec:
    size: list[float] = field(default_factory=lambda: [20.0, 20.0])
    prompt: str | None = None           # drawn as a tileable texture in the scene's style
    texture: str | None = None          # or a picture of the surface
    color: list[float] = field(default_factory=lambda: [0.36, 0.42, 0.27])
    tile: float = 4.0                   # metres covered by one repeat of the texture
    relief: float = 0.0                 # height of hills, metres
    feature: float = 8.0                # size of hills, metres
    resolution: float = 0.5             # metres between ground vertices
    flatten: float = 1.5                # flat pad around each explicitly placed object: this times its base radius


@dataclass
class SceneSpec:
    name: str
    objects: list[ObjectSpec]
    style: str = ""
    seed: int = 0
    ground: GroundSpec = field(default_factory=GroundSpec)
    harmonize: float = 0.5              # 0 leaves each prop's colours alone; 1 matches their lighting fully
    concept: str | None = None          # a picture of the whole scene: the colour reference, and see decompose

    def object(self, object_id: str) -> ObjectSpec:
        for o in self.objects:
            if o.id == object_id:
                return o
        raise KeyError(object_id)

    def to_dict(self) -> dict:
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items() if v is not None}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value

        return clean(asdict(self))


def _yaw(value, where: str):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value == "random":
        return value
    if isinstance(value, dict) and set(value) == {"face"}:
        target = value["face"]
        if isinstance(target, str) or (isinstance(target, list) and len(target) == 2):
            return {"face": target}
    if isinstance(value, dict) and set(value) == {"seen_from"} and isinstance(value["seen_from"], (int, float)):
        return {"seen_from": float(value["seen_from"])}
    raise SpecError(f'{where}: yaw is degrees, "random", {{"face": <object id or [x, z]>}} or '
                    f'{{"seen_from": <degrees>}}, not {value!r}')


def _number(data: dict, key: str, where: str, default=None, positive: bool = False):
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{where}: {key} must be a number, not {value!r}")
    if positive and value <= 0:
        raise SpecError(f"{where}: {key} must be more than 0")
    return float(value)


def _known(data: dict, allowed, where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise SpecError(f"{where}: unknown {'key' if len(unknown) == 1 else 'keys'} {', '.join(unknown)}; "
                        f"expected {', '.join(sorted(allowed))}")


def _placement(data, where: str) -> Placement:
    if isinstance(data, list):
        data = {"position": data}
    if not isinstance(data, dict):
        raise SpecError(f"{where}: a placement is {{\"position\": [x, z], ...}} or just [x, z]")
    _known(data, Placement.__dataclass_fields__, where)
    position = data.get("position")
    if not (isinstance(position, list) and len(position) in (2, 3)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in position)):
        raise SpecError(f"{where}: position is [x, z] or [x, y, z] in metres")
    return Placement(position=[float(v) for v in position], yaw=_yaw(data.get("yaw", 0.0), where),
                     scale=_number(data, "scale", where, 1.0, positive=True), tilt=bool(data.get("tilt", False)))


def _scatter(data, where: str) -> Scatter:
    if not isinstance(data, dict):
        raise SpecError(f"{where}: scatter is an object with at least a count")
    _known(data, Scatter.__dataclass_fields__, where)
    count = data.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise SpecError(f"{where}: count must be a whole number of at least 1")
    region = data.get("region", "all")
    if isinstance(region, str):
        if region not in REGIONS:
            raise SpecError(f"{where}: region is one of {', '.join(REGIONS)} or [[x0, z0], [x1, z1]]")
        region = "centre" if region == "center" else region
    elif not (isinstance(region, list) and len(region) == 2 and all(isinstance(c, list) and len(c) == 2
                                                                    for c in region)):
        raise SpecError(f"{where}: region is one of {', '.join(REGIONS)} or [[x0, z0], [x1, z1]]")
    scale = data.get("scale", [0.85, 1.15])
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        scale = [float(scale), float(scale)]
    if not (isinstance(scale, list) and len(scale) == 2 and 0 < scale[0] <= scale[1]):
        raise SpecError(f"{where}: scale is a number or [smallest, largest], both above 0")
    return Scatter(count=count, region=region, spacing=_number(data, "spacing", where, positive=True),
                   scale=[float(s) for s in scale], yaw=_yaw(data.get("yaw", "random"), where),
                   tilt=bool(data.get("tilt", False)), clearance=_number(data, "clearance", where, 0.5))


def _object(data, k: int, base: Path | None) -> ObjectSpec:
    where = f"objects[{k}]"
    if not isinstance(data, dict):
        raise SpecError(f"{where}: each object is a JSON object")
    _known(data, ObjectSpec.__dataclass_fields__, where)
    object_id = data.get("id")
    if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", object_id):
        raise SpecError(f"{where}: id must start with a letter and hold only letters, digits and _ (got {object_id!r})")
    where = f"object {object_id!r}"
    sources = [s for s in ("prompt", "image", "ply", "asset") if data.get(s)]
    if not sources:
        raise SpecError(f"{where}: give a prompt, an image, a ply or an asset to make it from")
    if len([s for s in sources if s != "prompt"]) > 1:
        raise SpecError(f"{where}: give only one of image, ply and asset")
    kind = data.get("kind", "prop")
    if kind not in KINDS:
        raise SpecError(f"{where}: kind is one of {', '.join(KINDS)}")
    height = _number(data, "height", where, positive=True)
    if height is None:
        raise SpecError(f"{where}: height (metres) is needed to make it the right size")

    def path(value):
        if not value:
            return None
        p = Path(value)
        return str((base / p) if base is not None and not p.is_absolute() else p)

    place = data.get("place", [])
    if not isinstance(place, list):
        raise SpecError(f"{where}: place is a list of placements")
    obj = ObjectSpec(
        id=object_id, height=height, prompt=data.get("prompt"), image=path(data.get("image")),
        ply=path(data.get("ply")), reference=path(data.get("reference")), asset=path(data.get("asset")), kind=kind,
        redraw=_number(data, "redraw", where, 0.0), style=bool(data.get("style", True)),
        collider=data.get("collider", "auto"), seed=data.get("seed"),
        place=[_placement(p, f"{where} place[{i}]") for i, p in enumerate(place)],
        scatter=_scatter(data["scatter"], f"{where} scatter") if data.get("scatter") is not None else None)
    if not obj.place and obj.scatter is None:
        raise SpecError(f"{where}: say where it goes, with place or scatter")
    if not 0 <= obj.redraw <= 1:
        raise SpecError(f"{where}: redraw is an image-to-image strength from 0 to 1")
    return obj


def parse(data: dict, base: Path | None = None) -> SceneSpec:
    """A checked scene spec. Relative file paths are taken from ``base`` (the spec's folder)."""
    if not isinstance(data, dict):
        raise SpecError("a scene spec is a JSON object")
    _known(data, {*SceneSpec.__dataclass_fields__, "$schema"}, "scene")
    name = data.get("name") or "scene"
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
        raise SpecError(f"scene: name {name!r} must be usable as a file name (letters, digits, _ . -)")
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise SpecError("scene: objects is a list with at least one object")
    parsed = [_object(o, k, base) for k, o in enumerate(objects)]
    ids = [o.id for o in parsed]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SpecError(f"scene: object ids must be unique ({', '.join(duplicates)} repeated)")
    ground = data.get("ground", {})
    if not isinstance(ground, dict):
        raise SpecError("scene: ground is an object")
    _known(ground, GroundSpec.__dataclass_fields__, "ground")
    size = ground.get("size", [20.0, 20.0])
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        size = [size, size]
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in size)):
        raise SpecError("ground: size is [width, depth] in metres")
    g = GroundSpec(size=[float(v) for v in size], prompt=ground.get("prompt"),
                   texture=str(base / ground["texture"]) if ground.get("texture") and base is not None
                   and not Path(ground["texture"]).is_absolute() else ground.get("texture"),
                   color=[float(c) for c in ground.get("color", GroundSpec().color)],
                   tile=_number(ground, "tile", "ground", 4.0, positive=True),
                   relief=_number(ground, "relief", "ground", 0.0),
                   feature=_number(ground, "feature", "ground", 8.0, positive=True),
                   resolution=_number(ground, "resolution", "ground", 0.5, positive=True),
                   flatten=_number(ground, "flatten", "ground", 1.5))
    for o in parsed:
        for k, p in enumerate(o.place):
            if isinstance(p.yaw, dict) and isinstance(p.yaw.get("face"), str) and p.yaw["face"] not in ids:
                raise SpecError(f"object {o.id!r} place[{k}]: faces {p.yaw['face']!r}, which is not an object id")
    harmonize = _number(data, "harmonize", "scene", 0.5)
    if not 0 <= harmonize <= 1:
        raise SpecError("scene: harmonize runs from 0 (off) to 1")
    concept = data.get("concept")
    if concept and base is not None and not Path(concept).is_absolute():
        concept = str(base / concept)
    return SceneSpec(name=name, objects=parsed, style=str(data.get("style", "")), seed=int(data.get("seed", 0)),
                     ground=g, harmonize=harmonize, concept=concept)


def load(path: Path) -> SceneSpec:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SpecError(f"{path}: not valid JSON ({e})")
    return parse(data, path.parent.resolve())
