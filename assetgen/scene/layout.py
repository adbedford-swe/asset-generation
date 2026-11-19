"""Where every object in a scene goes.

Explicitly placed objects come first: the ground is levelled under each so it sits
flat, and they turn as asked (a number of degrees, at random, or to face another
object). Scattered objects then fill their region one at a time, each kept far
enough from everything already placed that footprints do not overlap: scattered
neighbours may brush canopies (they keep apart by the average of their reach and
their base), and placed objects keep a clearance around their full reach, so paths
around a tent or a well stay open.

Every object stands on the ground: its base is lowered until no part of it floats
on a slope, or, with ``tilt``, it leans with the slope like a rock or a log would.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .spec import SceneSpec
from .terrain import Heightfield, flatten


@dataclass
class Footprint:
    """What layout needs to know about a generated object, at its generated height. ``view`` is where the
    camera of the picture it was made from stood, as a yaw from its front (0: the picture showed its front)."""
    height: float
    radius: float
    base_radius: float
    view: float = 0.0

    @classmethod
    def of(cls, prop_file: dict, view: float = 0.0) -> "Footprint":
        fp = prop_file["footprint"]
        return cls(prop_file["model"]["height"], fp["radius"], fp["base_radius"], view)


@dataclass
class Instance:
    object: str
    index: int
    position: list[float]
    yaw: float                # degrees about +Y; 0 faces +Z
    scale: float
    rotation: list[float]     # quaternion (x, y, z, w): the yaw, then any lean with the ground
    placed: bool              # explicitly placed (else scattered)

    @property
    def node(self) -> str:
        return f"{self.object}_{self.index}"


@dataclass
class Layout:
    instances: list[Instance]
    ground: Heightfield
    warnings: list[str] = field(default_factory=list)


def _seed(scene_seed: int, name: str) -> int:
    return (scene_seed * 1_000_003 + sum((k + 1) * ord(c) for k, c in enumerate(name))) % (2**32)


def yaw_quaternion(yaw_deg: float) -> np.ndarray:
    a = math.radians(yaw_deg) / 2
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def lean_quaternion(normal: np.ndarray) -> np.ndarray:
    """The rotation taking +Y onto ``normal``."""
    n = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    axis = np.cross([0.0, 1.0, 0.0], n)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    angle = math.atan2(s, n[1])
    axis /= s
    return np.array([*(axis * math.sin(angle / 2)), math.cos(angle / 2)])


def facing(origin, target) -> float:
    """The yaw that turns an object at ``origin`` (x, z) to face ``target`` (x, z)."""
    dx, dz = target[0] - origin[0], target[1] - origin[1]
    return math.degrees(math.atan2(dx, dz)) % 360 if (dx or dz) else 0.0


def standing_height(ground: Heightfield, x: float, z: float, base_radius: float) -> float:
    """The height to put an object's base at so none of it floats: the lowest ground under its base."""
    if base_radius <= 0:
        return float(ground.height(x, z))
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    xs = np.concatenate([[x], x + base_radius * np.cos(angles)])
    zs = np.concatenate([[z], z + base_radius * np.sin(angles)])
    return float(np.min(ground.height(xs, zs)))


def _region(region, size, margin: float, band: float):
    """A sampler of (x, z) points in a scatter region, and whether a point is in it."""
    hw, hd = size[0] / 2 - margin, size[1] / 2 - margin
    if isinstance(region, list):
        (x0, z0), (x1, z1) = region
        lo = (max(min(x0, x1), -hw), max(min(z0, z1), -hd))
        hi = (min(max(x0, x1), hw), min(max(z0, z1), hd))
        return lambda rng: (rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])), lambda p: True
    if region == "centre":
        return lambda rng: (rng.uniform(-hw / 2, hw / 2), rng.uniform(-hd / 2, hd / 2)), lambda p: True
    if region == "edge":
        band = min(band, hw, hd)
        return (lambda rng: (rng.uniform(-hw, hw), rng.uniform(-hd, hd)),
                lambda p: hw - abs(p[0]) < band or hd - abs(p[1]) < band)
    return lambda rng: (rng.uniform(-hw, hw), rng.uniform(-hd, hd)), lambda p: True


def _yaw(value, position, rng, anchors: dict, view: float = 0.0) -> float:
    if isinstance(value, dict) and "seen_from" in value:
        # Turn the side the object's picture showed towards the direction the scene's picture saw it from.
        return (value["seen_from"] - view) % 360
    if isinstance(value, dict):
        target = value["face"]
        target = anchors.get(target, position) if isinstance(target, str) else target
        return facing(position, target)
    if value == "random":
        return float(rng.uniform(0, 360))
    return float(value) % 360


def _rotation(ground: Heightfield, x: float, z: float, yaw: float, tilt: bool) -> np.ndarray:
    q = yaw_quaternion(yaw)
    if tilt:
        q = quat_multiply(lean_quaternion(ground.normal(x, z)), q)
    return q


def separate(points: list, radii: list, movable: list, seed: int = 0, gap: float = 0.1,
             iterations: int = 200) -> tuple[list[list[float]], list[float]]:
    """Push apart circles (x, z, radius) that overlap, moving each pair by as little as it takes and leaving the
    unmovable ones where they are. Returns the new centres and how far each moved."""
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    start = pts.copy()
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        overlapping = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if not (movable[i] or movable[j]):
                    continue
                d = pts[j] - pts[i]
                dist = float(np.hypot(*d))
                need = radii[i] + radii[j] + gap
                if dist >= need - 1e-6:
                    continue
                overlapping = True
                if dist > 1e-9:
                    direction = d / dist
                else:  # on top of each other: any way apart will do
                    angle = rng.uniform(0, 2 * math.pi)
                    direction = np.array([math.cos(angle), math.sin(angle)])
                push = need - dist
                if movable[i] and movable[j]:
                    pts[i] -= direction * push / 2
                    pts[j] += direction * push / 2
                elif movable[i]:
                    pts[i] -= direction * push
                else:
                    pts[j] += direction * push
        if not overlapping:
            break
    return pts.tolist(), np.hypot(*(pts - start).T).tolist()


def plan(spec: SceneSpec, footprints: dict[str, Footprint], ground: Heightfield) -> Layout:
    """Lay out every object of ``spec`` on ``ground``; ``footprints`` holds each object's measurements."""
    warnings: list[str] = []
    hw, hd = spec.ground.size[0] / 2, spec.ground.size[1] / 2

    # Objects placed by hand: moved apart where their bases overlap, a level pad under each, and the anchors
    # "face" can point at. Objects given a height (stacked on something on purpose) stay where they are.
    placed = [(o, p) for o in spec.objects for p in o.place]
    spots, moved = separate([[p.position[0], p.position[-1]] for _, p in placed],
                            [max(footprints[o.id].base_radius, 0.5 * footprints[o.id].radius) * p.scale
                             for o, p in placed],
                            [len(p.position) == 2 for _, p in placed], spec.seed)
    anchors: dict[str, list[float]] = {}
    pads = []
    for (o, p), (x, z), shift in zip(placed, spots, moved):
        fp = footprints[o.id]
        if shift > 0.01:
            warnings.append(f"{o.id} moved {shift:.2f} m from ({p.position[0]:g}, {p.position[-1]:g}) so it does "
                            f"not overlap its neighbours")
        if abs(x) > hw or abs(z) > hd:
            warnings.append(f"{o.id} at ({x:.2f}, {z:.2f}) is off the {spec.ground.size[0]:g} x "
                            f"{spec.ground.size[1]:g} m ground")
        anchors.setdefault(o.id, [x, z])
        if len(p.position) == 2 and not p.tilt and spec.ground.flatten > 0:
            pads.append((x, z, spec.ground.flatten * max(fp.base_radius, 0.25 * fp.radius) * p.scale))
    ground = flatten(ground, pads)

    instances: list[Instance] = []
    occupied: list[tuple[float, float, float, bool]] = []  # x, z, keep-out radius, placed by hand
    counts: dict[str, int] = {}

    def add(o, x, z, y, yaw, scale, tilt, by_hand):
        fp = footprints[o.id]
        if y is None:
            y = (float(ground.height(x, z)) - 0.05 * fp.base_radius * scale if tilt
                 else standing_height(ground, x, z, fp.base_radius * scale))
        k = counts.get(o.id, 0)
        counts[o.id] = k + 1
        q = _rotation(ground, x, z, yaw, tilt)
        instances.append(Instance(o.id, k, [round(x, 4), round(y, 4), round(z, 4)], round(yaw, 3), round(scale, 4),
                                  [round(float(v), 6) for v in q], by_hand))

    for (o, p), (x, z) in zip(placed, spots):
        fp = footprints[o.id]
        rng = np.random.default_rng(_seed(spec.seed, f"{o.id}/{len(instances)}"))
        add(o, x, z, p.position[1] if len(p.position) == 3 else None, _yaw(p.yaw, (x, z), rng, anchors, fp.view),
            p.scale, p.tilt, True)
        occupied.append((x, z, fp.radius * p.scale, True))

    for o in spec.objects:
        s = o.scatter
        if s is None:
            continue
        fp = footprints[o.id]
        rng = np.random.default_rng(_seed(spec.seed, o.id))
        mean_scale = (s.scale[0] + s.scale[1]) / 2
        reach = 0.5 * (fp.radius + fp.base_radius)
        spacing = s.spacing or 2 * reach * mean_scale
        band = max(1.5 * spacing, 0.15 * min(spec.ground.size))
        sample, inside = _region(s.region, spec.ground.size, fp.radius * s.scale[1], band)
        made, tries = 0, 0
        while made < s.count and tries < 200 * s.count:
            tries += 1
            x, z = sample(rng)
            if not inside((x, z)):
                continue
            scale = float(rng.uniform(*s.scale))
            mine = reach * scale

            def too_close(ox, oz, r, by_hand):
                need = r + s.clearance + fp.base_radius * scale if by_hand else (s.spacing or r + mine)
                return math.hypot(x - ox, z - oz) < need

            if any(too_close(*other) for other in occupied):
                continue
            add(o, x, z, None, _yaw(s.yaw, (x, z), rng, anchors, fp.view), scale, s.tilt, False)
            occupied.append((x, z, mine, False))
            made += 1
        if made < s.count:
            warnings.append(f"{o.id}: room for {made} of {s.count} without overlapping; enlarge the ground or the "
                            f"region, or lower the count or the spacing")
    return Layout(instances, ground, warnings)
